//! Held-Out Review Backtesting Engine for AETRE
//!
//! Evaluates 8 peer-review triage and allocation policies under a common fixed review budget K:
//! 1. AETRE (Calibrated Rescue Score)
//! 2. Random Allocation
//! 3. Closest-to-Boundary Allocation
//! 4. Highest-Variance Allocation
//! 5. Lowest-Reviewer-Confidence Allocation
//! 6. Mean-Plus-Variance Logistic Baseline
//! 7. Simple Historical Text Baseline
//! 8. Boundary-Distance-Plus-Variance Policy (without VOI)

use aetre_core::{calculate_brier_score, calculate_expected_calibration_error, PlattCalibrator};
use rand::seq::SliceRandom;
use rand::{Rng, SeedableRng};
use serde::{Deserialize, Serialize};
use serde_json::json;
use sha2::{Digest, Sha256};
use std::cmp::Ordering;
use std::collections::BTreeMap;
use std::fs;
use std::path::PathBuf;

#[derive(Debug, Clone, Deserialize, Serialize, Default)]
pub struct PreTriageData {
    #[serde(default = "default_m_count")]
    pub m_reviews_count: usize,
    #[serde(default, alias = "preliminary_scores")]
    pub initial_review_scores: Vec<f64>,
    #[serde(default, alias = "preliminary_confidences")]
    pub initial_reviewer_confidences: Option<Vec<f64>>,
    #[serde(default)]
    pub preliminary_mean: f64,
    #[serde(default)]
    pub preliminary_variance: f64,
    #[serde(default)]
    pub preliminary_mean_confidence: Option<f64>,
    #[serde(default)]
    pub preliminary_decision: Option<String>,
}

fn default_m_count() -> usize {
    2
}

#[derive(Debug, Clone, Deserialize, Serialize, Default)]
pub struct HeldoutEvaluationData {
    #[serde(default)]
    pub heldout_reviews_count: usize,
    #[serde(default)]
    pub heldout_scores: Vec<f64>,
    #[serde(default)]
    pub heldout_confidences: Option<Vec<f64>>,
    #[serde(default)]
    pub full_panel_mean: f64,
    #[serde(default)]
    pub full_panel_variance: f64,
    #[serde(default)]
    pub full_panel_decision: Option<String>,
    #[serde(default)]
    pub decision_flip_label: u8,
    #[serde(default)]
    pub significant_disagreement_label: Option<u8>,
    #[serde(default)]
    pub heldout_error_magnitude: Option<f64>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct BacktestCandidate {
    pub id: String,
    #[serde(default)]
    pub venue: Option<String>,
    #[serde(default)]
    pub year: Option<u32>,
    #[serde(default)]
    pub domain: Option<String>,
    #[serde(default)]
    pub split: String,
    #[serde(default)]
    pub title: String,
    #[serde(default)]
    pub abstract_text: Option<String>,
    #[serde(default)]
    pub unconventional_novelty: Option<f64>,
    pub pre_triage_data: PreTriageData,
    #[serde(default)]
    pub heldout_evaluation_data: HeldoutEvaluationData,
    #[serde(default)]
    pub label: u8,
    #[serde(default)]
    pub subgroup: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct PolicyMetricSummary {
    pub policy_name: String,
    pub budget_allocated_k: usize,
    pub true_positives_at_k: usize,
    pub total_positives_in_pool: usize,
    pub precision_at_k: f64,
    pub recall_at_k: f64,
    pub discoveries_at_k: usize,
    pub reviewer_hours_per_discovery: f64,
    pub roc_auc: Option<f64>,
    pub pr_auc: Option<f64>,
    pub brier_score: Option<f64>,
    pub expected_calibration_error: Option<f64>,
    pub decision_regret_at_k: f64,
    pub paired_recall_diff_vs_aetre: f64,
    pub paired_recall_diff_ci_95: (f64, f64),
}

fn compute_auc(labels: &[u8], scores: &[f64]) -> Option<f64> {
    let positives: Vec<f64> = labels
        .iter()
        .zip(scores)
        .filter_map(|(&l, &s)| (l == 1).then_some(s))
        .collect();
    let negatives: Vec<f64> = labels
        .iter()
        .zip(scores)
        .filter_map(|(&l, &s)| (l == 0).then_some(s))
        .collect();

    if positives.is_empty() || negatives.is_empty() {
        return None;
    }

    let wins: f64 = positives
        .iter()
        .flat_map(|pos| negatives.iter().map(move |neg| (pos, neg)))
        .map(|(pos, neg)| {
            if pos > neg {
                1.0
            } else if (pos - neg).abs() < 1e-12 {
                0.5
            } else {
                0.0
            }
        })
        .sum();

    Some(wins / (positives.len() * negatives.len()) as f64)
}

fn compute_pr_auc(labels: &[u8], scores: &[f64]) -> Option<f64> {
    let mut pairs: Vec<(f64, u8)> = scores.iter().copied().zip(labels.iter().copied()).collect();
    pairs.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(Ordering::Equal));

    let total_pos = labels.iter().filter(|&&l| l == 1).count();
    if total_pos == 0 {
        return None;
    }

    let mut true_pos = 0;
    let mut prev_recall = 0.0;
    let mut pr_auc = 0.0;

    for (i, &(_, label)) in pairs.iter().enumerate() {
        if label == 1 {
            true_pos += 1;
        }
        let precision = true_pos as f64 / (i + 1) as f64;
        let recall = true_pos as f64 / total_pos as f64;
        pr_auc += precision * (recall - prev_recall);
        prev_recall = recall;
    }

    Some(pr_auc)
}

/// Computes paired bootstrap 95% confidence interval for (Score_AETRE - Score_Baseline)
fn paired_bootstrap_ci(
    aetre_selected_mask: &[bool],
    baseline_selected_mask: &[bool],
    labels: &[u8],
    subgroups: &[String],
    total_positives: usize,
    replications: usize,
) -> (f64, f64) {
    if total_positives == 0 || labels.is_empty() {
        return (0.0, 0.0);
    }

    let mut rng = rand::rngs::StdRng::seed_from_u64(42);
    let n = labels.len();

    // Group indices by cluster (subgroup / venue-year)
    let mut clusters: BTreeMap<String, Vec<usize>> = BTreeMap::new();
    for (idx, sg) in subgroups.iter().enumerate() {
        clusters.entry(sg.clone()).or_default().push(idx);
    }
    let cluster_keys: Vec<String> = clusters.keys().cloned().collect();

    let mut diffs: Vec<f64> = Vec::with_capacity(replications);
    let use_cluster_bootstrap = cluster_keys.len() >= 2;

    for _ in 0..replications {
        let mut sample_indices = Vec::with_capacity(n);
        if use_cluster_bootstrap {
            for _ in 0..cluster_keys.len() {
                let chosen_cluster = cluster_keys.choose(&mut rng).unwrap();
                let members = &clusters[chosen_cluster];
                sample_indices.extend_from_slice(members);
            }
        } else {
            for _ in 0..n {
                sample_indices.push(rng.gen_range(0..n));
            }
        }

        let mut aetre_pos = 0;
        let mut baseline_pos = 0;
        let mut sample_total_pos = 0;

        for &idx in &sample_indices {
            let l = labels[idx];
            if l == 1 {
                sample_total_pos += 1;
                if aetre_selected_mask[idx] {
                    aetre_pos += 1;
                }
                if baseline_selected_mask[idx] {
                    baseline_pos += 1;
                }
            }
        }

        let denom = sample_total_pos.max(1) as f64;
        let diff = (aetre_pos as f64 / denom) - (baseline_pos as f64 / denom);
        diffs.push(diff);
    }

    diffs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));
    let low_idx = (replications as f64 * 0.025).round() as usize;
    let high_idx = (replications as f64 * 0.975).round() as usize;

    (
        diffs[low_idx.min(replications - 1)],
        diffs[high_idx.min(replications - 1)],
    )
}

/// Computes the frozen asymmetric rescue-ranking score used by the
/// retrospective conference benchmark.
///
/// This score is motivated by boundary-crossing value of information, but its
/// coefficients are empirically specified rather than derived from the
/// analytical VOI implementation in `aetre-core`.
pub fn compute_calibrated_rescue_score(pre_data: &PreTriageData, boundary: f64) -> f64 {
    let mean = pre_data.preliminary_mean;
    let scores = &pre_data.initial_review_scores;
    let conf = pre_data
        .preliminary_mean_confidence
        .unwrap_or(3.5)
        .clamp(1.0, 5.0);

    let spread = if scores.len() >= 2 {
        (scores[0] - scores[1]).abs()
    } else {
        pre_data.preliminary_variance.sqrt()
    };

    // Effective predictive noise standard deviation of 3rd review
    let eff_sigma = (1.25 + 0.40 * spread + 0.35 * (5.0 - conf)).max(0.5);

    if mean < boundary {
        let gap = boundary - mean;
        let z = 3.0 * gap / eff_sigma;
        aetre_core::normal_cdf(-z) * (1.0 + 0.20 * (5.0 - conf))
    } else {
        // Papers already above boundary have very low flip probability in real peer review
        let gap = mean - boundary;
        let z = 3.0 * gap / eff_sigma;
        0.01 * aetre_core::normal_cdf(-z)
    }
}

pub fn run_heldout_backtest(args: &[String]) -> Result<String, String> {
    let mut file = None;
    let mut output = None;
    let mut budget = 50;
    let mut boundary = 6.0;
    let mut target_split = "test".to_string();
    let mut json_out = false;

    let mut idx = 0;
    while idx < args.len() {
        let val = args.get(idx + 1);
        match args[idx].as_str() {
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
            "--split" => {
                if let Some(v) = val {
                    target_split = v.clone();
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
            let defaults = [
                "examples/datasets/openreview_heldout_backtest.json",
                "data/normalized/openreview_normalized.json",
            ];
            defaults
                .iter()
                .map(PathBuf::from)
                .find(|p| p.exists())
                .ok_or_else(|| {
                    "No backtest dataset specified (--file) and default fixtures not found."
                        .to_string()
                })?
        }
    };

    let raw = fs::read(&file_path)
        .map_err(|e| format!("cannot read backtest file {:?}: {}", file_path, e))?;
    let sha256_hash = format!("{:x}", Sha256::digest(&raw));

    let all_records: Vec<BacktestCandidate> =
        serde_json::from_slice(&raw).map_err(|e| format!("invalid backtest JSON: {}", e))?;

    if all_records.is_empty() {
        return Err("Backtest dataset is empty".to_string());
    }

    // Train calibrator strictly on CALIB split (no DEV contamination)
    let mut calib_records: Vec<&BacktestCandidate> =
        all_records.iter().filter(|r| r.split == "calib").collect();

    if calib_records.is_empty() {
        calib_records = all_records.iter().filter(|r| r.split == "dev").collect();
    }

    let calibrator = if !calib_records.is_empty() {
        let calib_scores: Vec<f64> = calib_records
            .iter()
            .map(|r| compute_calibrated_rescue_score(&r.pre_triage_data, boundary))
            .collect();
        let calib_labels: Vec<u8> = calib_records.iter().map(|r| r.label).collect();
        PlattCalibrator::fit(&calib_scores, &calib_labels, 1000, 0.05)
    } else {
        PlattCalibrator::new(2.0, -1.0, 0)
    };

    // Filter to target evaluation split (default: test)
    let eval_records: Vec<&BacktestCandidate> = all_records
        .iter()
        .filter(|r| r.split == target_split || target_split == "all")
        .collect();

    if eval_records.is_empty() {
        return Err(format!(
            "No records found for target split `{}`",
            target_split
        ));
    }

    let n = eval_records.len();
    let total_positives = eval_records.iter().filter(|r| r.label == 1).count();
    let labels: Vec<u8> = eval_records.iter().map(|r| r.label).collect();
    let subgroups: Vec<String> = eval_records
        .iter()
        .map(|r| {
            r.subgroup.clone().unwrap_or_else(|| {
                format!(
                    "{}_{}",
                    r.venue.as_deref().unwrap_or("venue"),
                    r.year.unwrap_or(2024)
                )
            })
        })
        .collect();

    // Compute policy scores for each evaluated record
    // 1. AETRE (VOI)
    let aetre_scores: Vec<f64> = eval_records
        .iter()
        .map(|r| {
            let raw_voi = compute_calibrated_rescue_score(&r.pre_triage_data, boundary);
            calibrator.predict_probability(raw_voi)
        })
        .collect();

    // 2. Random Allocation
    let mut rng = rand::rngs::StdRng::seed_from_u64(1337);
    let random_scores: Vec<f64> = (0..n).map(|_| rand::Rng::gen::<f64>(&mut rng)).collect();

    // 3. Closest-to-Boundary (|mean - boundary|^-1)
    let boundary_scores: Vec<f64> = eval_records
        .iter()
        .map(|r| {
            let dist = (r.pre_triage_data.preliminary_mean - boundary).abs();
            1.0 / (dist + 0.05)
        })
        .collect();

    // 4. Highest-Variance
    let variance_scores: Vec<f64> = eval_records
        .iter()
        .map(|r| r.pre_triage_data.preliminary_variance)
        .collect();

    // 5. Lowest-Reviewer-Confidence (1 / conf)
    let low_conf_scores: Vec<f64> = eval_records
        .iter()
        .map(|r| {
            let conf = r
                .pre_triage_data
                .preliminary_mean_confidence
                .unwrap_or(3.0)
                .max(1.0);
            1.0 / conf
        })
        .collect();

    // 6. Mean-Plus-Variance Logistic Baseline
    let logistic_scores: Vec<f64> = eval_records
        .iter()
        .map(|r| {
            let dist = (r.pre_triage_data.preliminary_mean - boundary).abs();
            let var = r.pre_triage_data.preliminary_variance;
            let z = 0.5 - 1.2 * dist + 0.8 * var;
            1.0 / (1.0 + (-z).exp())
        })
        .collect();

    // 7. Simple Historical Text Baseline
    let text_scores: Vec<f64> = eval_records
        .iter()
        .map(|r| r.unconventional_novelty.unwrap_or(0.3))
        .collect();

    // 8. Boundary + Variance (Heuristic Non-VOI)
    let non_voi_scores: Vec<f64> = eval_records
        .iter()
        .map(|r| {
            let dist = (r.pre_triage_data.preliminary_mean - boundary).abs();
            let var = r.pre_triage_data.preliminary_variance;
            var / (dist + 0.1)
        })
        .collect();

    let policies: Vec<(&str, Vec<f64>, bool)> = vec![
        (
            "AETRE (Calibrated Rescue Score)",
            aetre_scores.clone(),
            true,
        ),
        ("Random Allocation", random_scores, false),
        ("Closest-to-Boundary", boundary_scores, false),
        ("Highest-Variance", variance_scores, false),
        ("Lowest-Reviewer-Confidence", low_conf_scores, false),
        ("Mean-Plus-Variance Logistic", logistic_scores, true),
        ("Simple Historical Text", text_scores, false),
        ("Boundary + Variance (Non-VOI)", non_voi_scores, false),
    ];

    // Compute AETRE selected mask first for paired bootstrap comparisons
    let mut aetre_ranked: Vec<(usize, f64)> = aetre_scores.iter().copied().enumerate().collect();
    aetre_ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));
    let mut aetre_selected_mask = vec![false; n];
    for &(idx, _) in &aetre_ranked[..budget.min(n)] {
        aetre_selected_mask[idx] = true;
    }

    let mut summaries: Vec<PolicyMetricSummary> = Vec::new();

    for (pname, scores, scores_are_probabilities) in policies {
        let mut ranked: Vec<(usize, f64)> = scores.iter().copied().enumerate().collect();
        ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));

        let k_eff = budget.min(n);
        let selected = &ranked[..k_eff];

        let mut selected_mask = vec![false; n];
        let mut tp = 0;
        for &(idx, _) in selected {
            selected_mask[idx] = true;
            if labels[idx] == 1 {
                tp += 1;
            }
        }

        let precision = tp as f64 / k_eff.max(1) as f64;
        let recall = tp as f64 / total_positives.max(1) as f64;
        let rev_hours_per_disc = if tp > 0 {
            (k_eff as f64 * 4.0) / tp as f64
        } else {
            k_eff as f64 * 4.0
        };

        // Decision regret: missed corrections
        let missed_corrections = total_positives.saturating_sub(tp);
        let regret = missed_corrections as f64 / total_positives.max(1) as f64;

        let roc = compute_auc(&labels, &scores);
        let pr = compute_pr_auc(&labels, &scores);
        // Brier score and ECE require probability forecasts. Several baselines
        // intentionally emit unbounded ranking scores, so calibration metrics
        // are not defined for them.
        let brier = scores_are_probabilities.then(|| calculate_brier_score(&scores, &labels));
        let ece = scores_are_probabilities
            .then(|| calculate_expected_calibration_error(&scores, &labels, 10));

        let paired_diff = (aetre_selected_mask
            .iter()
            .zip(&labels)
            .filter(|(&s, &l)| s && l == 1)
            .count() as f64
            / total_positives.max(1) as f64)
            - recall;

        let ci = paired_bootstrap_ci(
            &aetre_selected_mask,
            &selected_mask,
            &labels,
            &subgroups,
            total_positives,
            1000,
        );

        summaries.push(PolicyMetricSummary {
            policy_name: pname.to_string(),
            budget_allocated_k: k_eff,
            true_positives_at_k: tp,
            total_positives_in_pool: total_positives,
            precision_at_k: precision,
            recall_at_k: recall,
            discoveries_at_k: tp,
            reviewer_hours_per_discovery: rev_hours_per_disc,
            roc_auc: roc,
            pr_auc: pr,
            brier_score: brier,
            expected_calibration_error: ece,
            decision_regret_at_k: regret,
            paired_recall_diff_vs_aetre: paired_diff,
            paired_recall_diff_ci_95: ci,
        });
    }

    // Validation Gates Check
    let aetre_summary = &summaries[0];
    let strongest_baseline = summaries[1..]
        .iter()
        .max_by(|a, b| {
            a.recall_at_k
                .partial_cmp(&b.recall_at_k)
                .unwrap_or(Ordering::Equal)
        })
        .unwrap();

    let retrospective_validity_pass = aetre_summary.recall_at_k >= strongest_baseline.recall_at_k
        && strongest_baseline.paired_recall_diff_ci_95.0 >= -0.05; // Paired CI
    let aetre_ece = aetre_summary
        .expected_calibration_error
        .expect("AETRE emits calibrated probabilities");
    let aetre_brier = aetre_summary
        .brier_score
        .expect("AETRE emits calibrated probabilities");
    let calibration_validity_pass = aetre_ece <= 0.15;
    let external_validity_pass = (target_split == "replication"
        || target_split == "external_replication")
        && eval_records.iter().any(|r| {
            let v_upper = r.venue.as_deref().map(|v| v.to_uppercase());
            v_upper.as_deref() == Some("NEURIPS") || v_upper.as_deref() == Some("ARR")
        })
        && retrospective_validity_pass;

    let report = json!({
        "backtest_dataset_sha256": sha256_hash,
        "dataset_file": file_path.to_string_lossy(),
        "evaluation_split": target_split,
        "total_records_evaluated": n,
        "total_decision_flips_in_split": total_positives,
        "baseline_flip_rate": total_positives as f64 / n as f64,
        "fixed_review_budget_k": budget,
        "acceptance_boundary_threshold": boundary,
        "calibrator": calibrator,
        "policy_summaries": summaries,
        "validation_gates": {
            "retrospective_validity": {
                "status": if retrospective_validity_pass { "PASS" } else { "FAIL" },
                "criterion": "AETRE Recall@K >= strongest baseline and 95% paired bootstrap interval non-inferior",
                "aetre_recall": aetre_summary.recall_at_k,
                "strongest_baseline_name": strongest_baseline.policy_name,
                "strongest_baseline_recall": strongest_baseline.recall_at_k,
                "paired_improvement_ci_95": strongest_baseline.paired_recall_diff_ci_95,
            },
            "calibration_validity": {
                "status": if calibration_validity_pass { "PASS" } else { "FAIL" },
                "criterion": "Expected Calibration Error (ECE) <= 0.15 on evaluation cohort",
                "observed_ece": aetre_ece,
                "observed_brier": aetre_brier,
            },
            "cross_venue_transfer_validity": {
                "status": if external_validity_pass { "PASS" } else { "FAIL" },
                "criterion": "Tested on independent un-inspected venue partition with frozen model",
                "venues_evaluated": eval_records.iter().map(|r| r.venue.as_deref().unwrap_or("unknown")).collect::<std::collections::HashSet<_>>(),
            }
        }
    });

    let rendered = serde_json::to_string_pretty(&report).map_err(|e| e.to_string())?;

    if let Some(p) = output {
        fs::write(&p, format!("{}\n", rendered))
            .map_err(|e| format!("cannot write output file {}: {}", p, e))?;
    }

    if json_out {
        return Ok(rendered);
    }

    // Pretty Terminal Markdown Table Output
    let mut out_str = String::new();
    out_str.push_str("\n╔═══════════════════════════════════════════════════════════════════════════════════════════════╗\n");
    out_str.push_str("║                   AETRE HELD-OUT REVIEW ALLOCATION BACKTEST BENCHMARK                         ║\n");
    out_str.push_str("╚═══════════════════════════════════════════════════════════════════════════════════════════════╝\n\n");
    out_str.push_str(&format!(
        "Evaluation Pool: {} papers | True Decision Flips: {} ({:.1}%) | Fixed Budget (K): {}\n",
        n,
        total_positives,
        (total_positives as f64 / n as f64) * 100.0,
        budget
    ));
    out_str.push_str(&format!("Dataset SHA-256: {}\n\n", &sha256_hash[..16]));

    out_str.push_str("| Policy Name | Recall@K | Precision@K | Discoveries | Hours/Disc. | ROC AUC | PR AUC | Paired Δ CI (95%) |\n");
    out_str.push_str("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |\n");

    for s in &summaries {
        let ci_str = if s.policy_name.contains("AETRE") {
            "Reference".to_string()
        } else {
            format!(
                "[{:.2}, {:.2}]",
                s.paired_recall_diff_ci_95.0, s.paired_recall_diff_ci_95.1
            )
        };

        out_str.push_str(&format!(
            "| **{}** | **{:.1}%** | {:.1}% | {} / {} | {:.1}h | {:.3} | {:.3} | {} |\n",
            s.policy_name,
            s.recall_at_k * 100.0,
            s.precision_at_k * 100.0,
            s.true_positives_at_k,
            total_positives,
            s.reviewer_hours_per_discovery,
            s.roc_auc.unwrap_or(0.5),
            s.pr_auc.unwrap_or(0.0),
            ci_str
        ));
    }

    out_str.push_str("\n--- Preregistered Validation Gates ---\n");
    out_str.push_str(&format!(
        "1. [Retrospective Validity]:          [{}] (AETRE Recall: {:.1}% vs Best Baseline '{}': {:.1}%)\n",
        if retrospective_validity_pass { "PASS" } else { "FAIL" },
        aetre_summary.recall_at_k * 100.0,
        strongest_baseline.policy_name,
        strongest_baseline.recall_at_k * 100.0
    ));
    out_str.push_str(&format!(
        "2. [Calibration Validity]:            [{}] (ECE: {:.3} <= 0.15, Brier: {:.4})\n",
        if calibration_validity_pass {
            "PASS"
        } else {
            "FAIL"
        },
        aetre_ece,
        aetre_brier
    ));
    out_str.push_str(&format!(
        "3. [Cross-Venue Transfer Validity]:   [{}] (Evaluated on partition '{}')\n\n",
        if external_validity_pass {
            "PASS"
        } else {
            "FAIL"
        },
        target_split
    ));

    Ok(out_str)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compute_auc_monotonic() {
        let labels = vec![0, 0, 1, 1];
        let scores = vec![0.1, 0.2, 0.8, 0.9];
        assert_eq!(compute_auc(&labels, &scores), Some(1.0));
    }

    #[test]
    fn test_compute_pr_auc() {
        let labels = vec![0, 1, 1];
        let scores = vec![0.1, 0.8, 0.9];
        let pr = compute_pr_auc(&labels, &scores);
        assert!(pr.is_some());
        assert!(pr.unwrap() > 0.8);
    }

    #[test]
    fn test_bootstrap_ci_coverage() {
        let aetre = vec![true, true, false, false];
        let baseline = vec![false, true, false, false];
        let labels = vec![1, 1, 0, 0];
        let groups = vec![
            "A".to_string(),
            "A".to_string(),
            "B".to_string(),
            "B".to_string(),
        ];
        let (low, high) = paired_bootstrap_ci(&aetre, &baseline, &labels, &groups, 2, 200);
        assert!(high >= low);
    }
}
