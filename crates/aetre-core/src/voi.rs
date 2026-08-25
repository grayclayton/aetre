use crate::types::{
    AgentEvaluation, AuthorPreflightReport, CorrelatedUpdateResult, HeavyTailVoiResult,
};
use std::f64::consts::PI;

/// Fast vectorized normal-CDF approximation with max error ~ 1e-4
pub fn normal_cdf(x: f64) -> f64 {
    0.5 * (1.0 + ((2.0 / PI).sqrt() * (x + 0.044715 * x.powi(3))).tanh())
}

/// Standard normal probability density function (PDF)
pub fn normal_pdf(x: f64) -> f64 {
    (-0.5 * x * x).exp() / (2.0 * PI).sqrt()
}

/// Normal-normal conjugate posterior update for latent quality
/// Given prior ~ N(prior_mean, prior_variance) and signal ~ N(quality, signal_noise^2)
pub fn posterior_update(
    prior_mean: f64,
    prior_variance: f64,
    signal: f64,
    signal_noise: f64,
) -> (f64, f64) {
    let signal_variance = signal_noise * signal_noise;
    let posterior_variance = 1.0 / (1.0 / prior_variance + 1.0 / signal_variance);
    let posterior_mean =
        posterior_variance * (prior_mean / prior_variance + signal / signal_variance);
    (posterior_mean, posterior_variance)
}

/// Bayesian conjugate posterior update for multiple correlated evaluator agents
///
/// Under an equi-correlated noise structure where inter-agent correlation is rho in [0, 1):
/// The effective number of independent evaluators is M_eff = M / (1 + (M - 1) * rho).
/// As rho -> 1, M_eff -> 1 regardless of how many LLM agents evaluate the candidate.
pub fn correlated_posterior_update(
    prior_mean: f64,
    prior_variance: f64,
    evaluations: &[AgentEvaluation],
    inter_agent_correlation: f64,
) -> CorrelatedUpdateResult {
    let m = evaluations.len();
    if m == 0 {
        return CorrelatedUpdateResult {
            posterior_mean: prior_mean,
            posterior_variance: prior_variance,
            effective_evaluator_count: 0.0,
            correlation_discount: 0.0,
        };
    }

    let rho = inter_agent_correlation.clamp(0.0, 0.999);
    let m_f64 = m as f64;
    let m_eff = m_f64 / (1.0 + (m_f64 - 1.0) * rho);
    let correlation_discount = 1.0 - (m_eff / m_f64);

    // Compute average noise variance across agents
    let mean_noise_variance = evaluations
        .iter()
        .map(|e| e.noise_sd * e.noise_sd)
        .sum::<f64>()
        / m_f64;

    // Aggregate precision accounting for effective evaluator sample size
    let agent_precision = (m_eff / mean_noise_variance.max(1e-6)).max(1e-12);
    let prior_precision = (1.0 / prior_variance.max(1e-6)).max(1e-12);

    let posterior_variance = 1.0 / (prior_precision + agent_precision);

    let weighted_signal_sum: f64 = evaluations.iter().map(|e| e.score).sum::<f64>() / m_f64;
    let posterior_mean =
        posterior_variance * (prior_precision * prior_mean + agent_precision * weighted_signal_sum);

    CorrelatedUpdateResult {
        posterior_mean,
        posterior_variance,
        effective_evaluator_count: m_eff,
        correlation_discount,
    }
}

/// Computes the Value of Information (VOI) for crossing a top-K selection boundary
///
/// Conditional on current information, the posterior mean after one additional
/// signal has variance v - v_new. The index is the expected decision boundary
/// crossing gain divided by signal cost.
pub fn calculate_boundary_voi(
    posterior_mean: f64,
    posterior_variance: f64,
    selection_boundary: f64,
    signal_noise: f64,
    review_cost: f64,
) -> f64 {
    let signal_variance = signal_noise * signal_noise;
    let new_variance = 1.0 / (1.0 / posterior_variance + 1.0 / signal_variance);
    let mean_shift_sd = (posterior_variance - new_variance).max(1e-12).sqrt();

    let gap = (posterior_mean - selection_boundary).abs();
    let z = gap / mean_shift_sd;
    let density = normal_pdf(z);
    let crossing_value = mean_shift_sd * density - gap * normal_cdf(-z);

    if review_cost > 0.0 {
        crossing_value / review_cost
    } else {
        crossing_value
    }
}

/// Computes Value of Information under a Heavy-Tailed / Pareto Payoff distribution
///
/// In innovation breakthroughs (venture capital, scientific discoveries, biotech),
/// payoffs follow a power law P(V > x) ~ x^(-alpha) for x >= threshold.
/// This VOI explicitly weights the positive black swan tail payoff.
pub fn calculate_heavy_tailed_voi(
    posterior_mean: f64,
    posterior_variance: f64,
    selection_boundary: f64,
    tail_index_alpha: f64, // alpha > 1.0 (typical Pareto index is 1.1 - 2.5)
    signal_noise: f64,
    review_cost: f64,
) -> HeavyTailVoiResult {
    let alpha = tail_index_alpha.max(1.05); // ensure alpha > 1 for finite expected mean
    let signal_variance = signal_noise * signal_noise;
    let new_variance = 1.0 / (1.0 / posterior_variance + 1.0 / signal_variance);
    let mean_shift_sd = (posterior_variance - new_variance).max(1e-12).sqrt();

    let gap = selection_boundary - posterior_mean;
    let z = gap / mean_shift_sd;
    let tail_probability = normal_cdf(-z);

    // Expected excess value above selection boundary under Generalized Pareto tail
    let threshold_scale = selection_boundary.abs().max(1.0);
    let expected_excess_payoff = threshold_scale / (alpha - 1.0);

    let gross_voi = mean_shift_sd
        * (normal_pdf(z) + tail_probability * (expected_excess_payoff / threshold_scale));
    let voi_index = if review_cost > 0.0 {
        gross_voi / review_cost
    } else {
        gross_voi
    };

    HeavyTailVoiResult {
        voi_index,
        tail_probability,
        expected_excess_payoff,
        tail_index: alpha,
    }
}

/// Evaluates a proposal draft from the author/researcher perspective:
/// Calculates crowd novelty percentile, reviewer disagreement split risk,
/// predicted triage stream, and generates prescriptive action items.
pub fn evaluate_author_preflight(
    title: &str,
    prior_mean: f64,
    epistemic_variance: f64,
    novelty_score: f64,
    selection_boundary: f64,
) -> AuthorPreflightReport {
    let voi = calculate_boundary_voi(prior_mean, epistemic_variance, selection_boundary, 0.8, 0.5);

    // Crowd novelty percentile approximation against baseline standard normal distribution
    // where median novelty is 0.35, standard deviation 0.20
    let novelty_z = (novelty_score - 0.35) / 0.20;
    let crowd_percentile = (normal_cdf(novelty_z) * 100.0).clamp(1.0, 99.9);

    let (stream, split_risk) = if prior_mean >= selection_boundary && epistemic_variance < 0.45 {
        ("FAST-PASS: DIRECT PHASE 2".to_string(), "LOW".to_string())
    } else if epistemic_variance >= 0.50 {
        (
            "HIGH VOI: DEEP REVIEW QUEUE".to_string(),
            "CRITICAL_SPLIT_RISK".to_string(),
        )
    } else if prior_mean < 0.40 {
        (
            "FAST-REJECT / SPAM FILTER".to_string(),
            "LOW_SIGNAL_REJECTION".to_string(),
        )
    } else {
        ("STANDARD_EVALUATION".to_string(), "MODERATE".to_string())
    };

    let mut action_plan = Vec::new();
    let mut variance_target = epistemic_variance;

    if epistemic_variance >= 0.50 {
        action_plan.push("High reviewer disagreement risk detected: Radical novelty without defensive empirical proofs triggers conservative consensus veto.".to_string());
        action_plan.push("Action: Add preliminary benchmark tables, negative control tests, or formal error bounds to compress variance below 0.35.".to_string());
        variance_target = 0.35;
    }

    if prior_mean < selection_boundary && novelty_score > 0.60 {
        action_plan.push(format!("Novelty is exceptionally high ({:.2}), but expected quality ({:.2}) sits below payline ({:.2}).", novelty_score, prior_mean, selection_boundary));
        action_plan.push("Action: Highlight clear translational milestones and reproducibility protocols to boost prior quality score.".to_string());
    }

    if novelty_score < 0.25 {
        action_plan.push("Crowded herd warning: Methodology semantic distance is low, indicating high overlap with incremental prior art.".to_string());
        action_plan.push("Action: Emphasize unique differentiation, structural mechanism departures, or non-linear scaling advantages.".to_string());
    }

    if action_plan.is_empty() {
        action_plan.push("Optimal epistemic configuration: Clear positive signal with low variance. Ready for submission.".to_string());
    }

    let hash_input = format!(
        "{}:{:.4}:{:.4}:{:.4}",
        title, prior_mean, epistemic_variance, novelty_score
    );
    use sha2::{Digest, Sha256};
    let evaluation_fingerprint = format!("aetre-eval-v1-{:x}", Sha256::digest(hash_input));

    let rank_str = format!(
        "Top_{:.1}%_Novelty",
        100.0 - (crowd_percentile * 10.0).round() / 10.0
    );
    let color = if prior_mean >= selection_boundary && epistemic_variance <= 0.35 {
        "2ea44f"
    } else if epistemic_variance >= 0.50 {
        "0969da"
    } else {
        "6e7781"
    };

    let markdown_badge = format!(
        "![AETRE Pre-Flight Evaluation](https://img.shields.io/badge/AETRE_Evaluated-{}-{}.svg)",
        rank_str, color
    );

    AuthorPreflightReport {
        title: title.to_string(),
        prior_mean,
        epistemic_variance,
        novelty_score,
        crowd_novelty_percentile: (crowd_percentile * 10.0).round() / 10.0,
        reviewer_disagreement_risk: split_risk,
        predicted_triage_stream: stream,
        voi_index: (voi * 1000.0).round() / 1000.0,
        prescriptive_action_plan: action_plan,
        variance_reduction_target: variance_target,
        evaluation_fingerprint,
        markdown_badge,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_normal_cdf() {
        assert!((normal_cdf(0.0) - 0.5).abs() < 1e-4);
        assert!((normal_cdf(1.96) - 0.975).abs() < 1e-3);
        assert!((normal_cdf(-1.96) - 0.025).abs() < 1e-3);
    }

    #[test]
    fn test_posterior_update() {
        let (mean, var) = posterior_update(0.0, 1.0, 2.0, 1.0);
        assert!((mean - 1.0).abs() < 1e-6);
        assert!((var - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_correlated_posterior_update_independent() {
        let evals = vec![
            AgentEvaluation {
                agent_id: "agent_1".into(),
                score: 2.0,
                noise_sd: 1.0,
            },
            AgentEvaluation {
                agent_id: "agent_2".into(),
                score: 2.0,
                noise_sd: 1.0,
            },
        ];
        let res = correlated_posterior_update(0.0, 1.0, &evals, 0.0);
        assert!((res.effective_evaluator_count - 2.0).abs() < 1e-6);
        assert!((res.correlation_discount - 0.0).abs() < 1e-6);
        assert!((res.posterior_variance - (1.0 / 3.0)).abs() < 1e-6);
    }

    #[test]
    fn test_correlated_posterior_update_fully_correlated() {
        let evals = vec![
            AgentEvaluation {
                agent_id: "agent_1".into(),
                score: 2.0,
                noise_sd: 1.0,
            },
            AgentEvaluation {
                agent_id: "agent_2".into(),
                score: 2.0,
                noise_sd: 1.0,
            },
            AgentEvaluation {
                agent_id: "agent_3".into(),
                score: 2.0,
                noise_sd: 1.0,
            },
            AgentEvaluation {
                agent_id: "agent_4".into(),
                score: 2.0,
                noise_sd: 1.0,
            },
        ];
        // High correlation: 4 agents collapse to effective ~1 agent
        let res = correlated_posterior_update(0.0, 1.0, &evals, 0.999);
        assert!((res.effective_evaluator_count - 1.0).abs() < 0.05);
        assert!(res.correlation_discount > 0.70);
    }

    #[test]
    fn test_heavy_tailed_voi() {
        let res = calculate_heavy_tailed_voi(1.2, 0.8, 1.5, 1.5, 0.5, 1.0);
        assert!(res.voi_index > 0.0);
        assert!(res.tail_probability > 0.0 && res.tail_probability < 1.0);
        assert_eq!(res.tail_index, 1.5);
    }

    #[test]
    fn test_boundary_voi_peak_at_boundary() {
        let boundary = 1.0;
        let voi_at_boundary = calculate_boundary_voi(1.0, 0.5, boundary, 0.8, 0.5);
        let voi_far_away = calculate_boundary_voi(4.0, 0.5, boundary, 0.8, 0.5);
        assert!(voi_at_boundary > voi_far_away);
        assert!(voi_at_boundary >= 0.0);
        assert!(voi_far_away >= 0.0);
    }

    #[test]
    fn test_author_preflight() {
        // High novelty, high variance candidate
        let report = evaluate_author_preflight("Novel Quantum Battery", 0.90, 0.85, 0.80, 1.20);
        assert!(report.crowd_novelty_percentile > 90.0);
        assert_eq!(report.reviewer_disagreement_risk, "CRITICAL_SPLIT_RISK");
        assert_eq!(
            report.predicted_triage_stream,
            "HIGH VOI: DEEP REVIEW QUEUE"
        );
        assert!(!report.prescriptive_action_plan.is_empty());
        assert_eq!(report.variance_reduction_target, 0.35);
    }
}
