//! Calibrated probability estimation, Platt scaling, and robustness routines.
//!
//! Maps raw heuristic scores, posterior means, and VOI values to calibrated probabilities
//! using strictly partitioned development and calibration cohorts.

use serde::{Deserialize, Serialize};

/// Platt Scaler: Logistic calibration mapping raw continuous score `s` to `P(Y = 1 | s)`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlattCalibrator {
    pub slope: f64,
    pub intercept: f64,
    pub calibration_method: String,
    pub training_records_count: usize,
}

impl PlattCalibrator {
    /// Creates a calibrated scaler with specified slope `a` and intercept `b`:
    /// `P(Y = 1 | s) = 1 / (1 + exp(-(a * s + b)))`
    pub fn new(slope: f64, intercept: f64, count: usize) -> Self {
        Self {
            slope,
            intercept,
            calibration_method: "Platt_Logistic_Scaling_v1".to_string(),
            training_records_count: count,
        }
    }

    /// Fits a Platt calibrator via gradient descent / Newton-Raphson on (score, label) pairs.
    pub fn fit(scores: &[f64], labels: &[u8], iterations: usize, lr: f64) -> Self {
        let n = scores.len();
        if n == 0 {
            return Self::new(1.0, 0.0, 0);
        }

        // Initialize slope and intercept
        let mut a = 1.0;
        let mut b = 0.0;

        for _ in 0..iterations {
            let mut grad_a = 0.0;
            let mut grad_b = 0.0;

            for (&s, &y) in scores.iter().zip(labels) {
                let target = y as f64;
                let z = (a * s + b).clamp(-30.0, 30.0);
                let p = 1.0 / (1.0 + (-z).exp());
                let err = p - target;
                grad_a += err * s;
                grad_b += err;
            }

            a -= lr * (grad_a / n as f64);
            b -= lr * (grad_b / n as f64);
        }

        Self::new(a, b, n)
    }

    /// Transforms a raw score into a calibrated probability in [0, 1].
    pub fn predict_probability(&self, raw_score: f64) -> f64 {
        let z = (self.slope * raw_score + self.intercept).clamp(-30.0, 30.0);
        (1.0 / (1.0 + (-z).exp())).clamp(0.0001, 0.9999)
    }
}

/// Computes Expected Calibration Error (ECE) across `num_bins` uniform confidence buckets.
pub fn calculate_expected_calibration_error(
    probabilities: &[f64],
    labels: &[u8],
    num_bins: usize,
) -> f64 {
    let n = probabilities.len();
    if n == 0 || num_bins == 0 {
        return 0.0;
    }

    let mut total_error = 0.0;
    let bin_width = 1.0 / num_bins as f64;

    for b in 0..num_bins {
        let lower = b as f64 * bin_width;
        let upper = if b == num_bins - 1 {
            1.0001
        } else {
            (b + 1) as f64 * bin_width
        };

        let mut bin_probs = Vec::new();
        let mut bin_labels = Vec::new();

        for (&p, &y) in probabilities.iter().zip(labels) {
            if p >= lower && p < upper {
                bin_probs.push(p);
                bin_labels.push(y as f64);
            }
        }

        if !bin_probs.is_empty() {
            let bin_size = bin_probs.len() as f64;
            let avg_conf: f64 = bin_probs.iter().sum::<f64>() / bin_size;
            let avg_acc: f64 = bin_labels.iter().sum::<f64>() / bin_size;
            let bin_err = (avg_conf - avg_acc).abs();
            total_error += (bin_size / n as f64) * bin_err;
        }
    }

    total_error
}

/// Computes the standard Brier Score: Mean squared error between predicted probabilities and labels.
pub fn calculate_brier_score(probabilities: &[f64], labels: &[u8]) -> f64 {
    if probabilities.is_empty() {
        return 0.0;
    }
    let sum_sq: f64 = probabilities
        .iter()
        .zip(labels)
        .map(|(&p, &y)| (p - y as f64).powi(2))
        .sum();
    sum_sq / probabilities.len() as f64
}

/// Robustness perturbation test result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerturbationRobustnessReport {
    pub baseline_score: f64,
    pub perturbed_score: f64,
    pub absolute_delta: f64,
    pub is_within_tolerance: bool,
    pub perturbation_type: String,
}

/// Evaluates stability of text heuristic scoring under adversarial perturbations
pub fn evaluate_text_robustness(
    original_text: &str,
    scorer_fn: impl Fn(&str) -> (f64, f64), // returns (mean, variance)
    tolerance: f64,
) -> Vec<PerturbationRobustnessReport> {
    let (base_mean, _) = scorer_fn(original_text);

    let perturbations = vec![
        (
            "keyword_insertion",
            format!("{original_text} quantum topological transformer diffusion"),
        ),
        (
            "fashionable_terminology",
            format!("{original_text} revolutionary breakthrough paradigm-shifting foundational"),
        ),
        (
            "document_truncation",
            original_text
                .chars()
                .take(original_text.len() / 2)
                .collect(),
        ),
        (
            "whitespace_padding",
            format!("\n\n   {original_text}   \n\n"),
        ),
    ];

    perturbations
        .into_iter()
        .map(|(ptype, text)| {
            let (p_mean, _) = scorer_fn(&text);
            let delta = (p_mean - base_mean).abs();
            PerturbationRobustnessReport {
                baseline_score: base_mean,
                perturbed_score: p_mean,
                absolute_delta: delta,
                is_within_tolerance: delta <= tolerance,
                perturbation_type: ptype.to_string(),
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_platt_calibrator_monotonic() {
        let calibrator = PlattCalibrator::new(2.5, -1.0, 100);
        let p_low = calibrator.predict_probability(0.0);
        let p_high = calibrator.predict_probability(2.0);
        assert!(p_high > p_low);
        assert!(p_low > 0.0 && p_high < 1.0);
    }

    #[test]
    fn test_platt_fit() {
        let scores = vec![0.1, 0.2, 0.3, 0.8, 0.9, 1.0];
        let labels = vec![0, 0, 0, 1, 1, 1];
        let calibrator = PlattCalibrator::fit(&scores, &labels, 500, 0.1);
        let p_neg = calibrator.predict_probability(0.15);
        let p_pos = calibrator.predict_probability(0.95);
        assert!(p_pos > p_neg);
    }

    #[test]
    fn test_ece_computation() {
        let probs = vec![0.1, 0.2, 0.8, 0.9];
        let labels = vec![0, 0, 1, 1];
        let ece = calculate_expected_calibration_error(&probs, &labels, 5);
        assert!(ece < 0.20);
    }

    #[test]
    fn test_brier_score() {
        let probs = vec![1.0, 0.0];
        let labels = vec![1, 0];
        assert_eq!(calculate_brier_score(&probs, &labels), 0.0);
    }
}
