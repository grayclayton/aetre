use serde::{Deserialize, Serialize};

/// Output of a randomized exploration audit on deprioritized candidates
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuditResult {
    pub deprioritized_pool_size: usize,
    pub audited_sample_size: usize,
    pub audited_high_value_found: usize,
    pub estimated_hidden_high_value: f64,
    pub estimated_hidden_high_value_std_err: f64,
    pub confidence_interval_95: (f64, f64),
}

/// Calculates the unbiased Horvitz-Thompson / simple random sampling estimator (H_hat_D)
/// for hidden high-value candidates in the deprioritized/rejected pool.
pub fn calculate_exploration_audit(
    deprioritized_pool_size: usize,
    audited_sample_size: usize,
    audited_high_value_found: usize,
) -> AuditResult {
    if audited_sample_size == 0 || deprioritized_pool_size == 0 {
        return AuditResult {
            deprioritized_pool_size,
            audited_sample_size: 0,
            audited_high_value_found: 0,
            estimated_hidden_high_value: 0.0,
            estimated_hidden_high_value_std_err: 0.0,
            confidence_interval_95: (0.0, 0.0),
        };
    }

    let n = audited_sample_size as f64;
    let n_total = deprioritized_pool_size as f64;
    let p_hat = audited_high_value_found as f64 / n;
    let estimated_hidden_high_value = n_total * p_hat;

    // Finite population correction variance: (N^2) * (p * (1-p) / (n - 1)) * (1 - n/N)
    let variance = if n > 1.0 && n_total > n {
        let sample_var = p_hat * (1.0 - p_hat) / (n - 1.0);
        let fpc = (n_total - n) / n_total;
        (n_total * n_total) * sample_var * fpc
    } else {
        0.0
    };

    let std_err = variance.max(0.0).sqrt();
    let margin = 1.96 * std_err;
    let ci_lower = (estimated_hidden_high_value - margin).max(0.0);
    let ci_upper = (estimated_hidden_high_value + margin).min(n_total);

    AuditResult {
        deprioritized_pool_size,
        audited_sample_size,
        audited_high_value_found,
        estimated_hidden_high_value,
        estimated_hidden_high_value_std_err: std_err,
        confidence_interval_95: (ci_lower, ci_upper),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_audit_estimator() {
        // Pool of 4800 rejected, sample 25, found 1 high-value
        // Expected hidden: 4800 * (1/25) = 192
        let res = calculate_exploration_audit(4800, 25, 1);
        assert_eq!(res.estimated_hidden_high_value, 192.0);
        assert!(res.confidence_interval_95.0 < 192.0);
        assert!(res.confidence_interval_95.1 > 192.0);
        assert!(res.confidence_interval_95.0 >= 0.0);
        assert!(res.confidence_interval_95.1 <= 4800.0);
    }

    #[test]
    fn test_audit_empty_or_zero_sample() {
        let res1 = calculate_exploration_audit(1000, 0, 0);
        assert_eq!(res1.estimated_hidden_high_value, 0.0);

        let res2 = calculate_exploration_audit(0, 50, 0);
        assert_eq!(res2.estimated_hidden_high_value, 0.0);
    }

    #[test]
    fn test_audit_all_found() {
        let res = calculate_exploration_audit(100, 100, 10);
        assert_eq!(res.estimated_hidden_high_value, 10.0);
        // When sample == total, FPC makes variance = 0
        assert_eq!(res.estimated_hidden_high_value_std_err, 0.0);
    }
}
