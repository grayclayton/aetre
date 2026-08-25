use serde::{Deserialize, Serialize};

/// Proposition 1 Throughput-Recall Bound state and evaluation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThroughputRecallBound {
    pub total_candidates: usize,
    pub selection_capacity: usize,
    pub high_value_rate: f64,
    pub expected_high_value_count: f64,
    pub theoretical_max_recall: f64,
    pub is_capacity_constrained: bool,
}

/// Evaluates Proposition 1: R_N <= min(1, K_N / H_N)
///
/// For any screening or ranking rule (including perfect information),
/// if candidates N grow faster than selection capacity K, recall must collapse.
pub fn calculate_proposition_1_bound(
    total_candidates: usize,
    selection_capacity: usize,
    high_value_rate: f64,
) -> ThroughputRecallBound {
    let expected_high_value = total_candidates as f64 * high_value_rate;
    let theoretical_max_recall = if expected_high_value > 0.0 {
        (selection_capacity as f64 / expected_high_value).min(1.0)
    } else {
        1.0
    };
    let is_capacity_constrained = (selection_capacity as f64) < expected_high_value;

    ThroughputRecallBound {
        total_candidates,
        selection_capacity,
        high_value_rate,
        expected_high_value_count: expected_high_value,
        theoretical_max_recall,
        is_capacity_constrained,
    }
}

/// Simulates recall curve across an arrival expansion scale [1x, 2x, 5x, 10x, 20x, 50x]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecallScalingPoint {
    pub arrival_multiplier: f64,
    pub arrivals: usize,
    pub selection_capacity: usize,
    pub max_theoretical_recall: f64,
}

pub fn generate_recall_scaling_curve(
    baseline_arrivals: usize,
    selection_capacity: usize,
    high_value_rate: f64,
    multipliers: &[f64],
) -> Vec<RecallScalingPoint> {
    multipliers
        .iter()
        .map(|&m| {
            let arrivals = (baseline_arrivals as f64 * m).round() as usize;
            let bound =
                calculate_proposition_1_bound(arrivals, selection_capacity, high_value_rate);
            RecallScalingPoint {
                arrival_multiplier: m,
                arrivals,
                selection_capacity,
                max_theoretical_recall: bound.theoretical_max_recall,
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_proposition_1_bound() {
        // Baseline 1000 arrivals, 200 capacity, 6.7% high-value (~67 high-value candidates)
        // 200 / 67 > 1.0 -> theoretical max recall = 1.0
        let b1 = calculate_proposition_1_bound(1000, 200, 0.067);
        assert_eq!(b1.theoretical_max_recall, 1.0);

        // 5000 arrivals, 200 capacity, 6.7% high-value (~335 high-value candidates)
        // 200 / 335 ~ 0.597 max recall
        let b2 = calculate_proposition_1_bound(5000, 200, 0.067);
        assert!((b2.theoretical_max_recall - 0.597).abs() < 0.01);

        // 20000 arrivals, 200 capacity, 6.7% high-value (~1340 high-value candidates)
        // 200 / 1340 ~ 0.149 max recall
        let b3 = calculate_proposition_1_bound(20000, 200, 0.067);
        assert!((b3.theoretical_max_recall - 0.149).abs() < 0.01);
    }

    #[test]
    fn test_scaling_curve_monotonic_decay() {
        let multipliers = vec![1.0, 2.0, 5.0, 10.0, 20.0, 50.0];
        let curve = generate_recall_scaling_curve(1000, 200, 0.067, &multipliers);
        assert_eq!(curve.len(), 6);
        for w in curve.windows(2) {
            assert!(w[0].max_theoretical_recall >= w[1].max_theoretical_recall);
        }
    }
}
