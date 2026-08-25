//! Dynamic Sequential Stopping Boundary Optimizer
//!
//! Evaluates incoming sequential reviewer observations and calculates the optimal Bayesian
//! stopping rule: whether to Accept, Reject, or Solicit an Additional Reviewer based on
//! posterior decision confidence and boundary-crossing VOI.

use crate::types::{SequentialDecision, SequentialReviewStep, SequentialStoppingResult};
use crate::voi::{calculate_boundary_voi, normal_cdf};

/// Evaluates a sequential review trajectory and determines whether to stop or request more reviews.
///
/// # Arguments
/// * `prior_mean` - Initial baseline prior mean (e.g. 5.0).
/// * `prior_variance` - Initial epistemic variance (e.g. 1.0).
/// * `threshold` - Acceptance cutline (e.g. 6.0).
/// * `reviews` - Ordered sequence of completed reviewer evaluations.
/// * `next_review_noise_sd` - Expected noise standard deviation of a future review (default 0.8).
/// * `next_review_cost` - Marginal cost of requesting an additional review (default 1.0).
/// * `confidence_threshold` - Required posterior confidence to finalize early (default 0.90).
pub fn evaluate_sequential_stopping(
    prior_mean: f64,
    prior_variance: f64,
    threshold: f64,
    reviews: &[SequentialReviewStep],
    next_review_noise_sd: Option<f64>,
    next_review_cost: Option<f64>,
    confidence_threshold: Option<f64>,
) -> SequentialStoppingResult {
    let conf_bound = confidence_threshold.unwrap_or(0.90).clamp(0.51, 0.99);
    let next_noise = next_review_noise_sd.unwrap_or(0.80).max(0.1);
    let next_cost = next_review_cost.unwrap_or(1.0).max(0.01);

    // Initial state
    let mut current_mean = prior_mean;
    let mut current_var = prior_variance.max(1e-6);
    let mut total_cost = 0.0;

    // Apply sequential Bayesian conjugate updates
    for rev in reviews {
        let noise_var = rev.noise_sd.max(0.1).powi(2);
        let post_precision = (1.0 / current_var) + (1.0 / noise_var);
        let post_var = 1.0 / post_precision;
        let post_mean = post_var * ((current_mean / current_var) + (rev.score / noise_var));

        current_mean = post_mean;
        current_var = post_var;
        total_cost += rev.cost.max(0.0);
    }

    let current_sd = current_var.sqrt();
    let z_score = (current_mean - threshold) / current_sd;
    let prob_exceed_threshold = normal_cdf(z_score);
    let boundary_dist = (current_mean - threshold).abs();

    // Calculate prospective VOI of one additional review
    let prospective_voi =
        calculate_boundary_voi(current_mean, current_var, threshold, next_noise, next_cost);

    // Decision Logic
    let (decision, rationale) = if prob_exceed_threshold >= conf_bound {
        (
            SequentialDecision::Accept,
            format!(
                "High confidence acceptance: P(q >= {:.2}) = {:.1}% >= {:.1}% target. Variance is well-resolved (sigma={:.2}).",
                threshold, prob_exceed_threshold * 100.0, conf_bound * 100.0, current_sd
            ),
        )
    } else if (1.0 - prob_exceed_threshold) >= conf_bound {
        (
            SequentialDecision::Reject,
            format!(
                "High confidence rejection: P(q < {:.2}) = {:.1}% >= {:.1}% target. Variance is well-resolved (sigma={:.2}).",
                threshold, (1.0 - prob_exceed_threshold) * 100.0, conf_bound * 100.0, current_sd
            ),
        )
    } else if prospective_voi < 0.02 {
        // Value of Information is too low to justify additional cost
        if current_mean >= threshold {
            (
                SequentialDecision::Accept,
                format!(
                    "Low residual VOI ({:.3} < 0.02); marginal review not cost-effective. Settling on Accept by posterior mean ({:.2} >= {:.2}).",
                    prospective_voi, current_mean, threshold
                ),
            )
        } else {
            (
                SequentialDecision::Reject,
                format!(
                    "Low residual VOI ({:.3} < 0.02); marginal review not cost-effective. Settling on Reject by posterior mean ({:.2} < {:.2}).",
                    prospective_voi, current_mean, threshold
                ),
            )
        }
    } else {
        (
            SequentialDecision::SolicitMoreReviews {
                recommended_next_evaluations: 1,
                expected_cost: next_cost,
            },
            format!(
                "Boundary uncertainty: candidate is near threshold ({:.2} vs {:.2}) with high VOI ({:.3}). Requesting 1 additional review.",
                current_mean, threshold, prospective_voi
            ),
        )
    };

    SequentialStoppingResult {
        current_step: reviews.len(),
        posterior_mean: current_mean,
        posterior_variance: current_var,
        decision,
        decision_confidence: prob_exceed_threshold.max(1.0 - prob_exceed_threshold),
        current_voi: prospective_voi,
        boundary_distance: boundary_dist,
        total_accumulated_cost: total_cost,
        stopping_rationale: rationale,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sequential_stopping_clear_accept() {
        let reviews = vec![
            SequentialReviewStep {
                step: 1,
                reviewer_id: "r1".to_string(),
                score: 8.5,
                noise_sd: 0.5,
                cost: 1.0,
            },
            SequentialReviewStep {
                step: 2,
                reviewer_id: "r2".to_string(),
                score: 8.0,
                noise_sd: 0.5,
                cost: 1.0,
            },
        ];

        let result = evaluate_sequential_stopping(5.0, 1.0, 6.0, &reviews, None, None, None);
        assert_eq!(result.decision, SequentialDecision::Accept);
        assert!(result.decision_confidence > 0.95);
    }

    #[test]
    fn test_sequential_stopping_clear_reject() {
        let reviews = vec![
            SequentialReviewStep {
                step: 1,
                reviewer_id: "r1".to_string(),
                score: 3.5,
                noise_sd: 0.5,
                cost: 1.0,
            },
            SequentialReviewStep {
                step: 2,
                reviewer_id: "r2".to_string(),
                score: 4.0,
                noise_sd: 0.5,
                cost: 1.0,
            },
        ];

        let result = evaluate_sequential_stopping(5.0, 1.0, 6.0, &reviews, None, None, None);
        assert_eq!(result.decision, SequentialDecision::Reject);
        assert!(result.decision_confidence > 0.95);
    }

    #[test]
    fn test_sequential_stopping_solicit_more_reviews() {
        let reviews = vec![
            SequentialReviewStep {
                step: 1,
                reviewer_id: "r1".to_string(),
                score: 6.1,
                noise_sd: 1.2,
                cost: 1.0,
            },
            SequentialReviewStep {
                step: 2,
                reviewer_id: "r2".to_string(),
                score: 5.9,
                noise_sd: 1.2,
                cost: 1.0,
            },
        ];

        let result = evaluate_sequential_stopping(6.0, 1.0, 6.0, &reviews, None, None, Some(0.90));
        assert!(matches!(
            result.decision,
            SequentialDecision::SolicitMoreReviews { .. }
        ));
    }
}
