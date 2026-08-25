//! Multi-Attribute Bayesian Value-of-Information (VOI) Epistemic Triage
//!
//! Evaluates proposals across orthogonal evaluation dimensions (e.g. Novelty, Empirical Rigor, Feasibility, Impact).
//! Computes composite utility, dimension variance shares, and marginal VOI to identify which specific
//! dimension yields the highest decision-correction value per review unit.

use crate::types::{DimensionVoiContribution, MultiAttributeDimension, MultiAttributeVoiResult};
use crate::voi::{normal_cdf, normal_pdf};

/// Computes multi-attribute Bayesian VOI and optimal dimension-specific review recommendations.
///
/// # Arguments
/// * `dimensions` - Vector of evaluation dimensions with prior means, variances, weights, and noise.
/// * `composite_threshold` - Overall acceptance threshold on composite score (default 6.0 if 0).
/// * `review_cost_per_dim` - Marginal review cost per dimension.
pub fn evaluate_multi_attribute_voi(
    dimensions: &[MultiAttributeDimension],
    composite_threshold: f64,
    review_cost_per_dim: f64,
) -> MultiAttributeVoiResult {
    if dimensions.is_empty() {
        return MultiAttributeVoiResult {
            composite_prior_mean: 0.0,
            composite_prior_variance: 0.0,
            composite_threshold,
            composite_voi: 0.0,
            dimension_contributions: Vec::new(),
            recommended_review_dimensions: Vec::new(),
            suggested_routing: "FastDrop".to_string(),
        };
    }

    let raw_weight_sum: f64 = dimensions.iter().map(|d| d.weight.max(0.0)).sum();
    let weight_norm = if raw_weight_sum > 0.0 {
        raw_weight_sum
    } else {
        dimensions.len() as f64
    };

    let mut composite_mean = 0.0;
    let mut composite_var = 0.0;

    for dim in dimensions {
        let w = if raw_weight_sum > 0.0 {
            dim.weight.max(0.0) / weight_norm
        } else {
            1.0 / dimensions.len() as f64
        };
        composite_mean += w * dim.prior_mean;
        composite_var += w * w * dim.prior_variance.max(1e-6);
    }

    let threshold = if composite_threshold > 0.0 {
        composite_threshold
    } else {
        6.0
    };
    let cost = review_cost_per_dim.max(0.01);

    let mut contributions = Vec::new();
    let mut ranked_dims: Vec<(String, f64)> = Vec::new();

    for dim in dimensions {
        let w = if raw_weight_sum > 0.0 {
            dim.weight.max(0.0) / weight_norm
        } else {
            1.0 / dimensions.len() as f64
        };

        let dim_var = dim.prior_variance.max(1e-6);
        let noise_var = (dim.review_noise_sd.max(0.1)).powi(2);
        let new_dim_var = 1.0 / ((1.0 / dim_var) + (1.0 / noise_var));
        let var_reduction_dim = (dim_var - new_dim_var).max(0.0);

        // Effective variance shift on the composite score from reviewing this dimension
        let sigma_mu_composite = w * var_reduction_dim.sqrt();

        let marginal_voi = if sigma_mu_composite > 1e-6 {
            let z = (composite_mean - threshold).abs() / sigma_mu_composite;
            let phi = normal_pdf(z);
            let big_phi = normal_cdf(-z);
            ((sigma_mu_composite * phi - (composite_mean - threshold).abs() * big_phi) / cost)
                .max(0.0)
        } else {
            0.0
        };

        let var_share = if composite_var > 0.0 {
            (w * w * dim_var) / composite_var
        } else {
            0.0
        };

        contributions.push(DimensionVoiContribution {
            dimension: dim.name.clone(),
            weight: w,
            marginal_voi,
            variance_share: var_share,
        });

        ranked_dims.push((dim.name.clone(), marginal_voi));
    }

    ranked_dims.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    let recommended_review_dimensions: Vec<String> = ranked_dims
        .into_iter()
        .filter(|(_, voi)| *voi > 0.005)
        .take(2)
        .map(|(name, _)| name)
        .collect();

    // Composite total VOI across all dimensions
    let total_composite_voi: f64 = contributions.iter().map(|c| c.marginal_voi).sum();

    let routing = if composite_mean >= threshold + 1.2 && composite_var < 0.25 {
        "AutoPass (Direct Phase 2)".to_string()
    } else if composite_mean < threshold - 1.5 && composite_var < 0.25 {
        "FastDrop (Low Quality, Low Uncertainty)".to_string()
    } else if total_composite_voi > 0.05 {
        format!(
            "Targeted Deep Review on: {}",
            if recommended_review_dimensions.is_empty() {
                "All Dimensions".to_string()
            } else {
                recommended_review_dimensions.join(", ")
            }
        )
    } else {
        "Standard Queue".to_string()
    };

    MultiAttributeVoiResult {
        composite_prior_mean: composite_mean,
        composite_prior_variance: composite_var,
        composite_threshold: threshold,
        composite_voi: total_composite_voi,
        dimension_contributions: contributions,
        recommended_review_dimensions,
        suggested_routing: routing,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_multi_attribute_voi_calculation() {
        let dimensions = vec![
            MultiAttributeDimension {
                name: "Novelty".to_string(),
                prior_mean: 6.5,
                prior_variance: 1.2,
                weight: 0.4,
                threshold: Some(6.0),
                review_noise_sd: 0.7,
            },
            MultiAttributeDimension {
                name: "Empirical Rigor".to_string(),
                prior_mean: 5.2,
                prior_variance: 0.8,
                weight: 0.4,
                threshold: Some(6.0),
                review_noise_sd: 0.6,
            },
            MultiAttributeDimension {
                name: "Broader Impact".to_string(),
                prior_mean: 5.8,
                prior_variance: 0.3,
                weight: 0.2,
                threshold: Some(6.0),
                review_noise_sd: 0.9,
            },
        ];

        let result = evaluate_multi_attribute_voi(&dimensions, 6.0, 1.0);

        // Composite mean: 0.4*6.5 + 0.4*5.2 + 0.2*5.8 = 2.6 + 2.08 + 1.16 = 5.84
        assert!((result.composite_prior_mean - 5.84).abs() < 1e-4);
        assert!(result.composite_voi > 0.0);
        assert_eq!(result.dimension_contributions.len(), 3);
        // High variance & high weight dimensions should have highest marginal VOI
        assert!(!result.recommended_review_dimensions.is_empty());
    }

    #[test]
    fn test_clear_autopass_routing() {
        let dimensions = vec![
            MultiAttributeDimension {
                name: "Novelty".to_string(),
                prior_mean: 8.5,
                prior_variance: 0.05,
                weight: 0.5,
                threshold: Some(6.0),
                review_noise_sd: 0.5,
            },
            MultiAttributeDimension {
                name: "Rigor".to_string(),
                prior_mean: 8.0,
                prior_variance: 0.05,
                weight: 0.5,
                threshold: Some(6.0),
                review_noise_sd: 0.5,
            },
        ];

        let result = evaluate_multi_attribute_voi(&dimensions, 6.0, 1.0);
        assert!(result.suggested_routing.contains("AutoPass"));
    }

    #[test]
    fn test_clear_fastdrop_routing() {
        let dimensions = vec![
            MultiAttributeDimension {
                name: "Novelty".to_string(),
                prior_mean: 3.0,
                prior_variance: 0.05,
                weight: 0.5,
                threshold: Some(6.0),
                review_noise_sd: 0.5,
            },
            MultiAttributeDimension {
                name: "Rigor".to_string(),
                prior_mean: 3.5,
                prior_variance: 0.05,
                weight: 0.5,
                threshold: Some(6.0),
                review_noise_sd: 0.5,
            },
        ];

        let result = evaluate_multi_attribute_voi(&dimensions, 6.0, 1.0);
        assert!(result.suggested_routing.contains("FastDrop"));
    }
}
