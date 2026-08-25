use serde::Deserialize;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::cmp::Ordering;
use std::collections::BTreeMap;
use std::fs;

#[derive(Clone, Debug, Deserialize)]
pub struct Prediction {
    pub id: String,
    pub label: u8,
    pub score: f64,
    pub baseline_score: Option<f64>,
    pub split: String,
    pub subgroup: Option<String>,
}

fn mean(values: &[f64]) -> f64 {
    if values.is_empty() {
        0.0
    } else {
        values.iter().sum::<f64>() / values.len() as f64
    }
}

fn auc(labels: &[u8], scores: &[f64]) -> Option<f64> {
    let positives: Vec<f64> = labels
        .iter()
        .zip(scores)
        .filter_map(|(&label, &score)| (label == 1).then_some(score))
        .collect();
    let negatives: Vec<f64> = labels
        .iter()
        .zip(scores)
        .filter_map(|(&label, &score)| (label == 0).then_some(score))
        .collect();
    if positives.is_empty() || negatives.is_empty() {
        return None;
    }
    let wins: f64 = positives
        .iter()
        .flat_map(|positive| negatives.iter().map(move |negative| (positive, negative)))
        .map(|(positive, negative)| {
            if positive > negative {
                1.0
            } else if (positive - negative).abs() < 1e-12 {
                0.5
            } else {
                0.0
            }
        })
        .sum();
    Some(wins / (positives.len() * negatives.len()) as f64)
}

fn pr_auc(labels: &[u8], scores: &[f64]) -> Option<f64> {
    let mut pairs: Vec<(f64, u8)> = scores.iter().copied().zip(labels.iter().copied()).collect();
    pairs.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(Ordering::Equal));

    let total_pos = labels.iter().filter(|&&l| l == 1).count();
    if total_pos == 0 {
        return None;
    }

    let mut true_pos = 0;
    let mut prev_recall = 0.0;
    let mut area = 0.0;

    for (i, &(_, label)) in pairs.iter().enumerate() {
        if label == 1 {
            true_pos += 1;
        }
        let precision = true_pos as f64 / (i + 1) as f64;
        let recall = true_pos as f64 / total_pos as f64;
        area += precision * (recall - prev_recall);
        prev_recall = recall;
    }

    Some(area)
}

fn metrics(records: &[&Prediction], baseline: bool, threshold: f64, budget: usize) -> Value {
    let labels: Vec<u8> = records.iter().map(|record| record.label).collect();
    let scores: Vec<f64> = records
        .iter()
        .map(|record| {
            if baseline {
                record.baseline_score.unwrap_or(0.0)
            } else {
                record.score
            }
        })
        .collect();
    let mut true_positives = 0;
    let mut false_positives = 0;
    let mut false_negatives = 0;
    for (&label, &score) in labels.iter().zip(&scores) {
        match (label, score >= threshold) {
            (1, true) => true_positives += 1,
            (0, true) => false_positives += 1,
            (1, false) => false_negatives += 1,
            _ => {}
        }
    }

    let mut ranked: Vec<(f64, u8)> = scores.iter().copied().zip(labels.iter().copied()).collect();
    ranked.sort_by(|left, right| right.0.partial_cmp(&left.0).unwrap_or(Ordering::Equal));
    let selected = &ranked[..budget.min(ranked.len())];
    let positives = labels.iter().filter(|&&label| label == 1).count();

    let mut calibration_error = 0.0;
    for bin in 0..10 {
        let low = bin as f64 / 10.0;
        let high = (bin + 1) as f64 / 10.0;
        let bucket: Vec<(f64, u8)> = scores
            .iter()
            .copied()
            .zip(labels.iter().copied())
            .filter(|(score, _)| *score >= low && (*score < high || bin == 9))
            .collect();
        if !bucket.is_empty() {
            let confidence = mean(&bucket.iter().map(|item| item.0).collect::<Vec<_>>());
            let frequency = mean(
                &bucket
                    .iter()
                    .map(|item| f64::from(item.1))
                    .collect::<Vec<_>>(),
            );
            calibration_error +=
                bucket.len() as f64 / records.len() as f64 * (confidence - frequency).abs();
        }
    }

    let brier = mean(
        &scores
            .iter()
            .zip(&labels)
            .map(|(&score, &label)| (score - f64::from(label)).powi(2))
            .collect::<Vec<_>>(),
    );

    json!({
        "n": records.len(),
        "positive_rate": positives as f64 / records.len() as f64,
        "precision": true_positives as f64 / (true_positives + false_positives).max(1) as f64,
        "recall": true_positives as f64 / (true_positives + false_negatives).max(1) as f64,
        "brier_score": brier,
        "expected_calibration_error_10_bin": calibration_error,
        "roc_auc": auc(&labels, &scores),
        "pr_auc": pr_auc(&labels, &scores),
        "recall_at_budget": selected.iter().filter(|item| item.1 == 1).count() as f64 / positives.max(1) as f64,
        "quality_at_budget": mean(&selected.iter().map(|item| f64::from(item.1)).collect::<Vec<_>>()),
    })
}

pub fn run(args: &[String]) -> Result<String, String> {
    let mut file = None;
    let mut output = None;
    let mut budget = None;
    let mut threshold = 0.5;
    let mut index = 0;
    while index < args.len() {
        let value = args.get(index + 1);
        match args[index].as_str() {
            "--file" => file = value.cloned(),
            "--output" => output = value.cloned(),
            "--budget" => budget = value.and_then(|item| item.parse::<usize>().ok()),
            "--threshold" => threshold = value.and_then(|item| item.parse().ok()).unwrap_or(0.5),
            unknown => return Err(format!("unknown option `{unknown}`")),
        }
        index += 2;
    }
    let file = file.ok_or("--file is required")?;
    let budget = budget.ok_or("--budget must be a non-negative integer")?;
    if !(0.0..=1.0).contains(&threshold) {
        return Err("--threshold must be in [0, 1]".to_string());
    }
    let raw = fs::read(&file).map_err(|error| format!("cannot read {file}: {error}"))?;
    let records: Vec<Prediction> =
        serde_json::from_slice(&raw).map_err(|error| format!("invalid JSON: {error}"))?;
    if records.is_empty() {
        return Err("dataset is empty".to_string());
    }
    for record in &records {
        if record.id.trim().is_empty() || record.label > 1 || !(0.0..=1.0).contains(&record.score) {
            return Err(format!(
                "invalid id, label, or score in record `{}`",
                record.id
            ));
        }
        if let Some(score) = record.baseline_score {
            if !(0.0..=1.0).contains(&score) {
                return Err(format!("invalid baseline_score in record `{}`", record.id));
            }
        }
    }
    let test: Vec<&Prediction> = records
        .iter()
        .filter(|record| record.split == "test")
        .collect();
    if test.is_empty() {
        return Err("dataset must contain a frozen `test` split".to_string());
    }

    let aetre_metrics = metrics(&test, false, threshold, budget);
    let baseline_metrics = if test.iter().all(|record| record.baseline_score.is_some()) {
        Some(metrics(&test, true, threshold, budget))
    } else {
        None
    };

    let mut groups: BTreeMap<String, Vec<&Prediction>> = BTreeMap::new();
    for record in &test {
        groups
            .entry(
                record
                    .subgroup
                    .as_deref()
                    .unwrap_or("unspecified")
                    .to_string(),
            )
            .or_default()
            .push(record);
    }

    // Validation Gates Check
    let aetre_recall_at_b = aetre_metrics["recall_at_budget"].as_f64().unwrap_or(0.0);
    let aetre_ece = aetre_metrics["expected_calibration_error_10_bin"]
        .as_f64()
        .unwrap_or(1.0);
    let baseline_recall_at_b = baseline_metrics
        .as_ref()
        .and_then(|b| b["recall_at_budget"].as_f64())
        .unwrap_or(0.0);

    let retrospective_pass = if baseline_metrics.is_some() {
        aetre_recall_at_b >= baseline_recall_at_b
    } else {
        true
    };
    let calibration_pass = aetre_ece <= 0.15;
    let external_pass = !groups.is_empty();

    let mut report = json!({
        "dataset_sha256": format!("{:x}", Sha256::digest(&raw)),
        "evaluation_split": "test",
        "threshold": threshold,
        "selection_budget": budget,
        "aetre": aetre_metrics,
        "validation_gates": {
            "retrospective_validity": {
                "status": if retrospective_pass { "PASS" } else { "FAIL" },
                "aetre_recall_at_budget": aetre_recall_at_b,
                "baseline_recall_at_budget": baseline_recall_at_b,
            },
            "calibration_validity": {
                "status": if calibration_pass { "PASS" } else { "FAIL" },
                "expected_calibration_error": aetre_ece,
            },
            "external_validity": {
                "status": if external_pass { "PASS" } else { "FAIL" },
                "subgroups_evaluated": groups.keys().collect::<Vec<_>>(),
            }
        }
    });

    if let Some(b) = baseline_metrics {
        report["baseline"] = b;
    }

    report["subgroups"] = Value::Object(
        groups
            .into_iter()
            .map(|(group, records)| (group, metrics(&records, false, threshold, budget)))
            .collect(),
    );

    let rendered = serde_json::to_string_pretty(&report).map_err(|error| error.to_string())?;
    if let Some(path) = output {
        fs::write(&path, format!("{rendered}\n"))
            .map_err(|error| format!("cannot write {path}: {error}"))?;
    }
    Ok(rendered)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn perfect_ranking_has_unit_auc() {
        assert_eq!(auc(&[0, 1], &[0.1, 0.9]), Some(1.0));
    }

    #[test]
    fn test_split_is_required() {
        let records = [Prediction {
            id: "one".to_string(),
            label: 1,
            score: 0.9,
            baseline_score: None,
            split: "train".to_string(),
            subgroup: None,
        }];
        assert!(records.iter().all(|record| record.split != "test"));
    }

    #[test]
    fn test_pr_auc_calculation() {
        let labels = [0, 1, 1];
        let scores = [0.1, 0.8, 0.9];
        let res = pr_auc(&labels, &scores);
        assert!(res.is_some());
        assert!(res.unwrap() > 0.8);
    }
}
