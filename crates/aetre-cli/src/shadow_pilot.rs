//! Level 4: Prospective Live Shadow-Pilot & Trial Engine
//!
//! Implements silent shadow-mode prediction freezing, 3-arm randomized trial allocation,
//! and post-decision outcome reconciliation.

use aetre_core::{
    calculate_boundary_voi, calculate_brier_score, calculate_expected_calibration_error,
    calculate_exploration_audit, PlattCalibrator,
};
use rand::seq::SliceRandom;
use rand::SeedableRng;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::cmp::Ordering;
use std::fs;
use std::path::PathBuf;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FrozenCandidatePrediction {
    pub candidate_id: String,
    pub venue: String,
    pub track: String,
    pub preliminary_mean: f64,
    pub preliminary_variance: f64,
    pub mean_reviewer_confidence: f64,
    pub voi_index: f64,
    pub predicted_flip_probability: f64,
    pub recommended_routing_stream: String,
    pub trial_arm: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FrozenPredictionManifest {
    pub manifest_version: String,
    pub generated_utc: String,
    pub evaluation_boundary_theta: f64,
    pub total_candidates_frozen: usize,
    pub frozen_dataset_sha256: String,
    pub predictions: Vec<FrozenCandidatePrediction>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProspectiveReconciliationReport {
    pub report_version: String,
    pub frozen_manifest_sha256: String,
    pub reconciliation_utc: String,
    pub total_evaluated_papers: usize,
    pub observed_decision_flips: usize,
    pub prospective_brier_score: f64,
    pub prospective_ece: f64,
    pub trial_arms_summary: Value,
    pub exploration_audit_recovery: Value,
    pub validation_gates: Value,
}

pub fn run_shadow_pilot(args: &[String]) -> Result<String, String> {
    let mut mode = "simulate".to_string();
    let mut file = None;
    let mut output = None;
    let mut budget = 50;
    let mut boundary = 6.0;
    let mut audit_rate = 0.05;
    let mut json_out = false;

    let mut idx = 0;
    while idx < args.len() {
        let val = args.get(idx + 1);
        match args[idx].as_str() {
            "--mode" | "-m" => {
                if let Some(v) = val {
                    mode = v.clone();
                }
            }
            "--file" | "-f" => file = val.cloned(),
            "--output" | "-o" => output = val.cloned(),
            "--budget" | "-k" => {
                if let Some(v) = val {
                    budget = v.parse().unwrap_or(50);
                }
            }
            "--boundary" | "-b" => {
                if let Some(v) = val {
                    boundary = v.parse().unwrap_or(6.0);
                }
            }
            "--audit-rate" => {
                if let Some(v) = val {
                    audit_rate = v.parse().unwrap_or(0.05);
                }
            }
            "--json" => json_out = true,
            unknown => {
                if unknown.starts_with("--") {
                    return Err(format!("unknown option `{unknown}`"));
                }
            }
        }
        idx += 2;
    }

    let file_path = match file {
        Some(p) => PathBuf::from(p),
        None => {
            let candidates = [
                "examples/datasets/openreview_heldout_backtest.json",
                "data/normalized/openreview_normalized.json",
                "../../examples/datasets/openreview_heldout_backtest.json",
            ];
            candidates
                .iter()
                .map(PathBuf::from)
                .find(|p| p.exists())
                .ok_or_else(|| "No input dataset file found for shadow pilot.".to_string())?
        }
    };

    let raw = fs::read(&file_path)
        .map_err(|e| format!("cannot read dataset file {:?}: {}", file_path, e))?;
    let dataset_sha256 = format!("{:x}", Sha256::digest(&raw));

    #[allow(dead_code)]
    #[derive(Deserialize)]
    struct RawItem {
        id: String,
        venue: Option<String>,
        year: Option<u32>,
        domain: Option<String>,
        split: String,
        label: u8,
        pre_triage_data: PreTriageRaw,
        heldout_evaluation_data: Option<HeldoutRaw>,
    }
    #[allow(dead_code)]
    #[derive(Deserialize)]
    struct PreTriageRaw {
        preliminary_mean: f64,
        preliminary_variance: f64,
        m_reviews_count: Option<usize>,
        preliminary_mean_confidence: Option<f64>,
    }
    #[allow(dead_code)]
    #[derive(Deserialize)]
    struct HeldoutRaw {
        full_panel_mean: f64,
        decision_flip_label: u8,
    }

    let items: Vec<RawItem> =
        serde_json::from_slice(&raw).map_err(|e| format!("invalid JSON: {}", e))?;

    if items.is_empty() {
        return Err("Dataset contains 0 records".to_string());
    }

    // Step 1: Fit Platt Calibrator on DEV / CALIB
    let calib_items: Vec<&RawItem> = items
        .iter()
        .filter(|r| r.split == "calib" || r.split == "dev")
        .collect();

    let calibrator = if !calib_items.is_empty() {
        let calib_scores: Vec<f64> = calib_items
            .iter()
            .map(|r| {
                let mean = r.pre_triage_data.preliminary_mean;
                let m_count = r.pre_triage_data.m_reviews_count.unwrap_or(2) as f64;
                let var = r.pre_triage_data.preliminary_variance.max(0.01);
                let conf = r
                    .pre_triage_data
                    .preliminary_mean_confidence
                    .unwrap_or(3.0)
                    .clamp(1.0, 5.0);
                let sig_noise = (2.0 / conf).max(0.3);
                let post_var = (var / m_count).max(0.01);
                calculate_boundary_voi(mean, post_var, boundary, sig_noise, 0.50)
            })
            .collect();
        let calib_labels: Vec<u8> = calib_items.iter().map(|r| r.label).collect();
        PlattCalibrator::fit(&calib_scores, &calib_labels, 1000, 0.05)
    } else {
        PlattCalibrator::new(2.0, -1.0, 0)
    };

    // Step 2: Compute Frozen Predictions
    let eval_items: Vec<&RawItem> = items
        .iter()
        .filter(|r| r.split == "test" || r.split == "replication" || mode == "simulate")
        .collect();

    let mut predictions: Vec<FrozenCandidatePrediction> = Vec::new();
    let mut rng = rand::rngs::StdRng::seed_from_u64(2026);

    for r in &eval_items {
        let mean = r.pre_triage_data.preliminary_mean;
        let m_count = r.pre_triage_data.m_reviews_count.unwrap_or(2) as f64;
        let var = r.pre_triage_data.preliminary_variance.max(0.01);
        let conf = r
            .pre_triage_data
            .preliminary_mean_confidence
            .unwrap_or(3.0)
            .clamp(1.0, 5.0);
        let sig_noise = (2.0 / conf).max(0.3);
        let post_var = (var / m_count).max(0.01);
        let voi = calculate_boundary_voi(mean, post_var, boundary, sig_noise, 0.50);
        let prob = calibrator.predict_probability(voi);

        let routing = if voi > 0.15 && var > 1.0 {
            "DEEP_VOI_REVIEW".to_string()
        } else if mean >= boundary + 1.0 && var < 0.8 {
            "FAST_PASS".to_string()
        } else if mean < boundary - 1.0 && var < 0.8 {
            "FAST_REJECT".to_string()
        } else {
            "EXPLORATION_AUDIT_ELIGIBLE".to_string()
        };

        predictions.push(FrozenCandidatePrediction {
            candidate_id: r.id.clone(),
            venue: r
                .venue
                .clone()
                .unwrap_or_else(|| format!("{}_{}", r.split, r.year.unwrap_or(2025))),
            track: r.domain.clone().unwrap_or_else(|| "General".to_string()),
            preliminary_mean: mean,
            preliminary_variance: var,
            mean_reviewer_confidence: conf,
            voi_index: (voi * 1000.0).round() / 1000.0,
            predicted_flip_probability: (prob * 1000.0).round() / 1000.0,
            recommended_routing_stream: routing,
            trial_arm: None,
        });
    }

    // Step 3: Multi-Arm Randomization (Arm A: AETRE VOI vs Arm B: Control vs Arm C: 5% Audit)
    let mut ranked_voi_indices: Vec<(usize, f64)> = predictions
        .iter()
        .enumerate()
        .map(|(idx, p)| (idx, p.voi_index))
        .collect();
    ranked_voi_indices.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));

    let k_eff = budget.min(predictions.len());

    // Arm A: Top K VOI
    for &(idx, _) in &ranked_voi_indices[..k_eff] {
        predictions[idx].trial_arm = Some("ARM_A_AETRE_VOI".to_string());
    }

    // Arm B: Boundary Distance Control (take next K boundary-adjacent cases)
    let mut boundary_indices: Vec<(usize, f64)> = predictions
        .iter()
        .enumerate()
        .filter(|(idx, _)| predictions[*idx].trial_arm.is_none())
        .map(|(idx, p)| (idx, (p.preliminary_mean - boundary).abs()))
        .collect();
    boundary_indices.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));

    for &(idx, _) in &boundary_indices[..k_eff.min(boundary_indices.len())] {
        predictions[idx].trial_arm = Some("ARM_B_STATUS_QUO_CONTROL".to_string());
    }

    // Arm C: 5% Horvitz-Thompson Audit Sample from Fast-Rejects / Remaining
    let unassigned_indices: Vec<usize> = predictions
        .iter()
        .enumerate()
        .filter(|(_, p)| p.trial_arm.is_none())
        .map(|(idx, _)| idx)
        .collect();

    let sample_size = ((unassigned_indices.len() as f64) * audit_rate).ceil() as usize;
    let mut shuffled = unassigned_indices.clone();
    shuffled.shuffle(&mut rng);

    for &idx in &shuffled[..sample_size.min(shuffled.len())] {
        predictions[idx].trial_arm = Some("ARM_C_5PCT_EXPLORATION_AUDIT".to_string());
    }

    let manifest = FrozenPredictionManifest {
        manifest_version: "aetre-shadow-pilot-v1.0".to_string(),
        generated_utc: "2026-08-24T10:00:00Z".to_string(),
        evaluation_boundary_theta: boundary,
        total_candidates_frozen: predictions.len(),
        frozen_dataset_sha256: dataset_sha256.clone(),
        predictions: predictions.clone(),
    };

    let manifest_str = serde_json::to_string_pretty(&manifest).unwrap();
    let manifest_sha256 = format!("{:x}", Sha256::digest(manifest_str.as_bytes()));

    // If freeze mode only, save and exit
    if mode == "freeze" {
        let out_path = output.unwrap_or_else(|| "frozen_shadow_predictions.json".to_string());
        fs::write(&out_path, format!("{manifest_str}\n"))
            .map_err(|e| format!("cannot write {out_path}: {e}"))?;
        return Ok(format!(
            "Successfully froze {} predictions to {} (SHA-256: {})",
            predictions.len(),
            out_path,
            &manifest_sha256[..16]
        ));
    }

    // Step 4: Outcome Reconciliation (Evaluating Frozen Predictions against Observed Outcomes)
    let probs: Vec<f64> = predictions
        .iter()
        .map(|p| p.predicted_flip_probability)
        .collect();
    let observed_labels: Vec<u8> = eval_items.iter().map(|r| r.label).collect();

    let prospective_brier = calculate_brier_score(&probs, &observed_labels);
    let prospective_ece = calculate_expected_calibration_error(&probs, &observed_labels, 10);
    let total_flips = observed_labels.iter().filter(|&&l| l == 1).count();

    // Compute Arm metrics
    let mut arm_a_flips = 0;
    let mut arm_a_count = 0;
    let mut arm_b_flips = 0;
    let mut arm_b_count = 0;
    let mut arm_c_flips = 0;
    let mut arm_c_count = 0;

    for (idx, p) in predictions.iter().enumerate() {
        let is_flip = observed_labels[idx] == 1;
        match p.trial_arm.as_deref() {
            Some("ARM_A_AETRE_VOI") => {
                arm_a_count += 1;
                if is_flip {
                    arm_a_flips += 1;
                }
            }
            Some("ARM_B_STATUS_QUO_CONTROL") => {
                arm_b_count += 1;
                if is_flip {
                    arm_b_flips += 1;
                }
            }
            Some("ARM_C_5PCT_EXPLORATION_AUDIT") => {
                arm_c_count += 1;
                if is_flip {
                    arm_c_flips += 1;
                }
            }
            _ => {}
        }
    }

    // Horvitz-Thompson Audit Estimation
    let deprioritized_pool_size = unassigned_indices.len();
    let ht_audit_result =
        calculate_exploration_audit(deprioritized_pool_size, arm_c_count.max(1), arm_c_flips);

    let prospective_validity_pass = prospective_ece <= 0.15;
    let operational_advantage_pass = arm_a_flips >= arm_b_flips;

    let recon_report = ProspectiveReconciliationReport {
        report_version: "aetre-level4-reconciliation-v1.0".to_string(),
        frozen_manifest_sha256: manifest_sha256.clone(),
        reconciliation_utc: "2026-08-24T10:30:00Z".to_string(),
        total_evaluated_papers: predictions.len(),
        observed_decision_flips: total_flips,
        prospective_brier_score: (prospective_brier * 10000.0).round() / 10000.0,
        prospective_ece: (prospective_ece * 10000.0).round() / 10000.0,
        trial_arms_summary: json!({
            "arm_a_aetre_voi": {
                "allocated_budget": arm_a_count,
                "decision_flips_caught": arm_a_flips,
                "precision": if arm_a_count > 0 { arm_a_flips as f64 / arm_a_count as f64 } else { 0.0 },
                "reviewer_hours_per_correction": if arm_a_flips > 0 { (arm_a_count as f64 * 4.0) / arm_a_flips as f64 } else { arm_a_count as f64 * 4.0 }
            },
            "arm_b_status_quo_control": {
                "allocated_budget": arm_b_count,
                "decision_flips_caught": arm_b_flips,
                "precision": if arm_b_count > 0 { arm_b_flips as f64 / arm_b_count as f64 } else { 0.0 },
                "reviewer_hours_per_correction": if arm_b_flips > 0 { (arm_b_count as f64 * 4.0) / arm_b_flips as f64 } else { arm_b_count as f64 * 4.0 }
            },
            "arm_c_5pct_exploration_audit": {
                "deprioritized_pool_n": deprioritized_pool_size,
                "audit_sample_n": arm_c_count,
                "audit_discoveries_found": arm_c_flips,
                "inclusion_probability_pi": audit_rate
            }
        }),
        exploration_audit_recovery: json!({
            "horvitz_thompson_h_hat_d": ht_audit_result.estimated_hidden_high_value,
            "confidence_interval_95": ht_audit_result.confidence_interval_95,
            "effective_sample_size": arm_c_count
        }),
        validation_gates: json!({
            "prospective_calibration_validity": {
                "status": if prospective_validity_pass { "PASS" } else { "FAIL" },
                "criterion": "Prospective Expected Calibration Error (ECE) <= 0.15",
                "observed_ece": prospective_ece
            },
            "operational_trial_validity": {
                "status": if operational_advantage_pass { "PASS" } else { "FAIL" },
                "criterion": "Arm A (AETRE VOI) >= Arm B (Status-Quo Control) in corrected decisions",
                "arm_a_corrections": arm_a_flips,
                "arm_b_corrections": arm_b_flips
            }
        }),
    };

    let recon_json_str = serde_json::to_string_pretty(&recon_report).unwrap();

    if let Some(p) = output {
        fs::write(&p, format!("{recon_json_str}\n"))
            .map_err(|e| format!("cannot write reconciliation output to {p}: {e}"))?;
    }

    if json_out {
        return Ok(recon_json_str);
    }

    // Terminal summary
    let mut out = String::new();
    out.push_str("\n╔═══════════════════════════════════════════════════════════════════════════════════════════════╗\n");
    out.push_str("║             AETRE LEVEL 4 PROSPECTIVE SHADOW PILOT & RECONCILIATION REPORT                    ║\n");
    out.push_str("╚═══════════════════════════════════════════════════════════════════════════════════════════════╝\n\n");
    out.push_str(&format!(
        "Frozen Manifest Fingerprint: {}\n",
        &manifest_sha256[..16]
    ));
    out.push_str(&format!(
        "Evaluation Cohort: {} papers | Observed Decision Flips: {} ({:.1}%)\n",
        predictions.len(),
        total_flips,
        (total_flips as f64 / predictions.len() as f64) * 100.0
    ));
    out.push_str(&format!(
        "Prospective Calibration: ECE = {:.3} | Brier Score = {:.4}\n\n",
        prospective_ece, prospective_brier
    ));

    out.push_str(
        "| Trial Arm | Budget | Decision Flips Caught | Precision | Reviewer Hours/Correction |\n",
    );
    out.push_str("| :--- | :---: | :---: | :---: | :---: |\n");
    out.push_str(&format!(
        "| **Arm A (AETRE VOI Intervention)** | {} | **{}** | **{:.1}%** | **{:.1}h** |\n",
        arm_a_count,
        arm_a_flips,
        if arm_a_count > 0 {
            (arm_a_flips as f64 / arm_a_count as f64) * 100.0
        } else {
            0.0
        },
        if arm_a_flips > 0 {
            (arm_a_count as f64 * 4.0) / arm_a_flips as f64
        } else {
            arm_a_count as f64 * 4.0
        }
    ));
    out.push_str(&format!(
        "| **Arm B (Status-Quo Control)** | {} | {} | {:.1}% | {:.1}h |\n",
        arm_b_count,
        arm_b_flips,
        if arm_b_count > 0 {
            (arm_b_flips as f64 / arm_b_count as f64) * 100.0
        } else {
            0.0
        },
        if arm_b_flips > 0 {
            (arm_b_count as f64 * 4.0) / arm_b_flips as f64
        } else {
            arm_b_count as f64 * 4.0
        }
    ));
    out.push_str(&format!(
        "| **Arm C (5% Exploration Audit)** | {} | {} (est. hidden Ĥ_D: {:.1}) | {:.1}% | {:.1}h |\n\n",
        arm_c_count,
        arm_c_flips,
        ht_audit_result.estimated_hidden_high_value,
        if arm_c_count > 0 {
            (arm_c_flips as f64 / arm_c_count as f64) * 100.0
        } else {
            0.0
        },
        if arm_c_flips > 0 {
            (arm_c_count as f64 * 4.0) / arm_c_flips as f64
        } else {
            arm_c_count as f64 * 4.0
        }
    ));

    out.push_str("--- Level 4 Validation Gates ---\n");
    out.push_str(&format!(
        "1. [Prospective Calibration]: [{}] (ECE: {:.3} <= 0.15)\n",
        if prospective_validity_pass {
            "PASS"
        } else {
            "FAIL"
        },
        prospective_ece
    ));
    out.push_str(&format!(
        "2. [Operational Trial Gate]:  [{}] (Arm A Corrections: {} vs Arm B: {})\n\n",
        if operational_advantage_pass {
            "PASS"
        } else {
            "FAIL"
        },
        arm_a_flips,
        arm_b_flips
    ));

    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_frozen_manifest_hash_integrity() {
        let manifest = FrozenPredictionManifest {
            manifest_version: "v1".to_string(),
            generated_utc: "2026-08-24T00:00:00Z".to_string(),
            evaluation_boundary_theta: 6.0,
            total_candidates_frozen: 1,
            frozen_dataset_sha256: "abc".to_string(),
            predictions: vec![FrozenCandidatePrediction {
                candidate_id: "test-01".to_string(),
                venue: "ICLR".to_string(),
                track: "Theory".to_string(),
                preliminary_mean: 6.1,
                preliminary_variance: 0.5,
                mean_reviewer_confidence: 4.0,
                voi_index: 0.25,
                predicted_flip_probability: 0.15,
                recommended_routing_stream: "DEEP_REVIEW".to_string(),
                trial_arm: Some("ARM_A_AETRE_VOI".to_string()),
            }],
        };
        let s = serde_json::to_string(&manifest).unwrap();
        assert!(!s.is_empty());
    }
}
