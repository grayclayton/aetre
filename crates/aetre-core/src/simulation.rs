use rand::seq::SliceRandom;
use rand::Rng;
use rand_distr::{Distribution, Normal};

use crate::audit::calculate_exploration_audit;
use crate::types::{Candidate, CandidateStatus, Parameters, RegimeStats, SelectionSummary};
use crate::voi::{calculate_boundary_voi, posterior_update};

/// Generates a synthetic cohort of innovation proposals
pub fn generate_candidates<R: Rng>(
    rng: &mut R,
    arrivals: usize,
    unconventional_share: f64,
) -> Vec<Candidate> {
    let normal = Normal::new(0.0, 1.0).unwrap();
    (0..arrivals)
        .map(|id| {
            let latent_quality = normal.sample(rng);
            let is_unconventional = rng.gen_bool(unconventional_share);
            Candidate {
                id,
                latent_quality,
                is_unconventional,
                posterior_mean: 0.0,
                posterior_variance: 1.0,
                voi_index: 0.0,
                status: CandidateStatus::Pending,
            }
        })
        .collect()
}

/// Helper to get indices of top-K elements in a slice of f64
pub fn top_indices(values: &[f64], k: usize) -> Vec<usize> {
    if k == 0 || values.is_empty() {
        return Vec::new();
    }
    let k = k.min(values.len());
    let mut indexed: Vec<(usize, f64)> = values.iter().copied().enumerate().collect();
    indexed.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    indexed.into_iter().take(k).map(|(idx, _)| idx).collect()
}

/// Summarizes the results of a selection run
pub fn summarize_selection(
    candidates: &[Candidate],
    accepted_indices: &[usize],
    human_reviews: usize,
    high_value_threshold: f64,
    estimated_hidden: Option<f64>,
) -> SelectionSummary {
    let arrivals = candidates.len();
    let accepted_count = accepted_indices.len();

    let mut total_quality = 0.0;
    let mut negative_quality_count = 0;
    let mut total_unconventional_high_value = 0;
    let mut captured_unconventional_high_value = 0;

    let accepted_set: std::collections::HashSet<usize> = accepted_indices.iter().copied().collect();

    for (idx, c) in candidates.iter().enumerate() {
        let is_high_value = c.latent_quality >= high_value_threshold;
        let is_unconventional_hv = is_high_value && c.is_unconventional;
        if is_unconventional_hv {
            total_unconventional_high_value += 1;
        }

        if accepted_set.contains(&idx) {
            total_quality += c.latent_quality;
            if c.latent_quality < 0.0 {
                negative_quality_count += 1;
            }
            if is_unconventional_hv {
                captured_unconventional_high_value += 1;
            }
        }
    }

    let mean_quality = if accepted_count > 0 {
        total_quality / accepted_count as f64
    } else {
        0.0
    };

    let fdr = if accepted_count > 0 {
        negative_quality_count as f64 / accepted_count as f64
    } else {
        0.0
    };

    let unconventional_recall = if total_unconventional_high_value > 0 {
        captured_unconventional_high_value as f64 / total_unconventional_high_value as f64
    } else {
        0.0
    };

    SelectionSummary {
        arrivals,
        accepted: accepted_count,
        human_reviews,
        quality_throughput: total_quality,
        mean_accepted_quality: mean_quality,
        false_discovery_rate: fdr,
        unconventional_high_value_recall: unconventional_recall,
        final_signal_noise: 0.0,
        evaluation_cost: 0.0,
        estimated_hidden_unconventional: estimated_hidden,
    }
}

/// Regime 1: Baseline, single-pass unmanaged screening
pub fn run_unmanaged_screening<R: Rng>(
    rng: &mut R,
    candidates: Vec<Candidate>,
    params: &Parameters,
) -> SelectionSummary {
    let arrivals = candidates.len();
    let attention_dilution = (arrivals as f64 / params.evaluation_budget.max(1.0)).sqrt();
    let sigma = (params.unmanaged_baseline_noise * attention_dilution).max(1e-9);
    let normal_noise = Normal::new(0.0, sigma).unwrap_or_else(|_| Normal::new(0.0, 1e-9).unwrap());

    let mut scores = Vec::with_capacity(arrivals);
    for c in &candidates {
        let noise = normal_noise.sample(rng);
        let novelty_bias = if c.is_unconventional {
            params.novelty_penalty
        } else {
            0.0
        };
        scores.push(c.latent_quality + noise + novelty_bias);
    }

    let accepted = top_indices(&scores, params.acceptance_capacity);
    let mut summary = summarize_selection(
        &candidates,
        &accepted,
        arrivals,
        params.high_value_threshold,
        None,
    );
    summary.final_signal_noise = sigma;
    summary.evaluation_cost = params.evaluation_budget;
    summary
}

/// Regimes 3 & 4: Adaptive Bayesian VOI Screening (with optional exploration audit)
pub fn run_voi_screening<R: Rng>(
    rng: &mut R,
    mut candidates: Vec<Candidate>,
    params: &Parameters,
    audit_budget_share: f64,
) -> SelectionSummary {
    let arrivals = candidates.len();
    let initial_cost = params.initial_screen_cost * arrivals as f64;
    let mut remaining_budget = (params.evaluation_budget - initial_cost).max(0.0);

    // 1. Initial Cheap Screen across all arrivals
    let init_sigma = params.initial_screen_noise.max(1e-9);
    let initial_noise =
        Normal::new(0.0, init_sigma).unwrap_or_else(|_| Normal::new(0.0, 1e-9).unwrap());
    for c in &mut candidates {
        let noise = initial_noise.sample(rng);
        let novelty_bias = if c.is_unconventional {
            params.novelty_penalty
        } else {
            0.0
        };
        let initial_signal = c.latent_quality + noise + novelty_bias;
        let (p_mean, p_var) = posterior_update(0.0, 1.0, initial_signal, init_sigma);
        c.posterior_mean = p_mean;
        c.posterior_variance = p_var;
    }

    let audit_budget = (audit_budget_share * params.evaluation_budget).min(remaining_budget);
    let audit_capacity = if params.deep_review_cost > 0.0 {
        (audit_budget / params.deep_review_cost).floor() as usize
    } else {
        0
    };

    let fast_budget = params.fast_review_budget_share * remaining_budget;
    let fast_capacity = if params.fast_review_cost > 0.0 {
        (fast_budget / params.fast_review_cost).floor() as usize
    } else {
        0
    };

    // Find current top-K boundary
    let mut current_means: Vec<f64> = candidates.iter().map(|c| c.posterior_mean).collect();
    current_means.sort_unstable_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
    let boundary_idx = (params.acceptance_capacity.min(arrivals)).saturating_sub(1);
    let boundary = current_means.get(boundary_idx).copied().unwrap_or(0.0);

    // Calculate VOI for all candidates
    for c in &mut candidates {
        c.voi_index = calculate_boundary_voi(
            c.posterior_mean,
            c.posterior_variance,
            boundary,
            params.fast_review_noise.max(1e-9),
            params.fast_review_cost,
        );
    }

    // 2. Fast Review stage assigned to top VOI candidates
    let voi_scores: Vec<f64> = candidates.iter().map(|c| c.voi_index).collect();
    let fast_reviewed_indices = top_indices(&voi_scores, fast_capacity);

    let fast_sigma = params.fast_review_noise.max(1e-9);
    let fast_noise =
        Normal::new(0.0, fast_sigma).unwrap_or_else(|_| Normal::new(0.0, 1e-9).unwrap());
    for &idx in &fast_reviewed_indices {
        let c = &mut candidates[idx];
        let signal = c.latent_quality + fast_noise.sample(rng);
        let (p_mean, p_var) =
            posterior_update(c.posterior_mean, c.posterior_variance, signal, fast_sigma);
        c.posterior_mean = p_mean;
        c.posterior_variance = p_var;
    }

    remaining_budget =
        (remaining_budget - params.fast_review_cost * fast_reviewed_indices.len() as f64).max(0.0);

    // 3. Optional Randomized Exploration Audit Pool
    let mut estimated_hidden = None;
    let fast_set: std::collections::HashSet<usize> =
        fast_reviewed_indices.iter().copied().collect();
    let mut audit_pool: Vec<usize> = (0..arrivals).filter(|i| !fast_set.contains(i)).collect();

    let actual_audit_capacity = audit_capacity.min(audit_pool.len());
    let mut audited_indices = Vec::new();

    if actual_audit_capacity > 0 {
        audit_pool.shuffle(rng);
        audited_indices = audit_pool.into_iter().take(actual_audit_capacity).collect();

        let deep_sigma = params.deep_review_noise.max(1e-9);
        let deep_noise =
            Normal::new(0.0, deep_sigma).unwrap_or_else(|_| Normal::new(0.0, 1e-9).unwrap());
        let mut audited_high_value_unconventional = 0;

        for &idx in &audited_indices {
            let c = &mut candidates[idx];
            let signal = c.latent_quality + deep_noise.sample(rng);
            let (p_mean, p_var) =
                posterior_update(c.posterior_mean, c.posterior_variance, signal, deep_sigma);
            c.posterior_mean = p_mean;
            c.posterior_variance = p_var;
            c.status = CandidateStatus::RandomlyAudited;

            if c.latent_quality >= params.high_value_threshold && c.is_unconventional {
                audited_high_value_unconventional += 1;
            }
        }

        let deprioritized_pool_size = arrivals.saturating_sub(fast_reviewed_indices.len());
        let audit_res = calculate_exploration_audit(
            deprioritized_pool_size,
            actual_audit_capacity,
            audited_high_value_unconventional,
        );
        estimated_hidden = Some(audit_res.estimated_hidden_high_value);
        remaining_budget =
            (remaining_budget - params.deep_review_cost * actual_audit_capacity as f64).max(0.0);
    }

    // 4. Final Selection based on updated posterior means
    let final_means: Vec<f64> = candidates.iter().map(|c| c.posterior_mean).collect();
    let accepted = top_indices(&final_means, params.acceptance_capacity);

    let total_human_reviews = fast_reviewed_indices.len() + audited_indices.len();
    let mut summary = summarize_selection(
        &candidates,
        &accepted,
        total_human_reviews,
        params.high_value_threshold,
        estimated_hidden,
    );
    summary.evaluation_cost = params.evaluation_budget - remaining_budget;
    summary
}

/// Runs a full multi-replicate Monte Carlo benchmark across all 4 regimes
pub fn run_benchmark_replications<R: Rng>(
    rng: &mut R,
    replications: usize,
    params: &Parameters,
) -> Vec<RegimeStats> {
    let mut summaries_baseline = Vec::with_capacity(replications);
    let mut summaries_flood = Vec::with_capacity(replications);
    let mut summaries_voi = Vec::with_capacity(replications);
    let mut summaries_audit = Vec::with_capacity(replications);

    let flood_arrivals =
        (params.baseline_arrivals as f64 * params.ai_arrival_multiplier).round() as usize;

    for _ in 0..replications {
        // Use a shared latent cohort for the three flood regimes. This paired design
        // isolates routing effects from differences in randomly generated candidates.
        let flood_cohort = generate_candidates(rng, flood_arrivals, params.unconventional_share);

        // The baseline is the first N candidates from the same generated population.
        let c1 = flood_cohort[..params.baseline_arrivals.min(flood_cohort.len())].to_vec();
        summaries_baseline.push(run_unmanaged_screening(rng, c1, params));

        // Regime 2: Flood 5000 / 200 Unmanaged
        summaries_flood.push(run_unmanaged_screening(rng, flood_cohort.clone(), params));

        // Regime 3: Flood 5000 / 200 VOI
        summaries_voi.push(run_voi_screening(rng, flood_cohort.clone(), params, 0.0));

        // Regime 4: Flood 5000 / 200 VOI + 5% Audit
        summaries_audit.push(run_voi_screening(
            rng,
            flood_cohort,
            params,
            params.randomized_audit_budget_share,
        ));
    }

    vec![
        aggregate_stats(
            "Baseline, one pass",
            params.baseline_arrivals,
            params.acceptance_capacity,
            &summaries_baseline,
        ),
        aggregate_stats(
            "Fivefold arrivals, one pass",
            flood_arrivals,
            params.acceptance_capacity,
            &summaries_flood,
        ),
        aggregate_stats(
            "Fivefold arrivals, VOI screening",
            flood_arrivals,
            params.acceptance_capacity,
            &summaries_voi,
        ),
        aggregate_stats(
            "Fivefold arrivals, VOI + 5% audit",
            flood_arrivals,
            params.acceptance_capacity,
            &summaries_audit,
        ),
    ]
}

fn aggregate_stats(
    name: &str,
    arrivals: usize,
    acceptances: usize,
    data: &[SelectionSummary],
) -> RegimeStats {
    if data.is_empty() {
        return RegimeStats {
            regime_name: name.to_string(),
            arrivals,
            acceptances,
            quality_throughput_mean: 0.0,
            quality_throughput_run_interval: (0.0, 0.0),
            mean_accepted_quality_mean: 0.0,
            mean_accepted_quality_run_interval: (0.0, 0.0),
            unconventional_recall_mean: 0.0,
            unconventional_recall_run_interval: (0.0, 0.0),
            false_discovery_rate_mean: 0.0,
            false_discovery_rate_run_interval: (0.0, 0.0),
            human_reviews_mean: 0.0,
            estimated_hidden_unconventional_mean: None,
        };
    }

    let n = data.len() as f64;
    let qt_vals: Vec<f64> = data.iter().map(|s| s.quality_throughput).collect();
    let mq_vals: Vec<f64> = data.iter().map(|s| s.mean_accepted_quality).collect();
    let ur_vals: Vec<f64> = data
        .iter()
        .map(|s| s.unconventional_high_value_recall)
        .collect();
    let fdr_vals: Vec<f64> = data.iter().map(|s| s.false_discovery_rate).collect();
    let hr_vals: Vec<f64> = data.iter().map(|s| s.human_reviews as f64).collect();

    let qt_mean = qt_vals.iter().sum::<f64>() / n;
    let mq_mean = mq_vals.iter().sum::<f64>() / n;
    let ur_mean = ur_vals.iter().sum::<f64>() / n;
    let fdr_mean = fdr_vals.iter().sum::<f64>() / n;
    let hr_mean = hr_vals.iter().sum::<f64>() / n;

    let qt_run_interval = compute_percentiles(&qt_vals, 0.025, 0.975);
    let mq_run_interval = compute_percentiles(&mq_vals, 0.025, 0.975);
    let ur_run_interval = compute_percentiles(&ur_vals, 0.025, 0.975);
    let fdr_run_interval = compute_percentiles(&fdr_vals, 0.025, 0.975);

    let hidden_vals: Vec<f64> = data
        .iter()
        .filter_map(|s| s.estimated_hidden_unconventional)
        .collect();
    let hidden_mean = if hidden_vals.is_empty() {
        None
    } else {
        Some(hidden_vals.iter().sum::<f64>() / hidden_vals.len() as f64)
    };

    RegimeStats {
        regime_name: name.to_string(),
        arrivals,
        acceptances,
        quality_throughput_mean: qt_mean,
        quality_throughput_run_interval: qt_run_interval,
        mean_accepted_quality_mean: mq_mean,
        mean_accepted_quality_run_interval: mq_run_interval,
        unconventional_recall_mean: ur_mean,
        unconventional_recall_run_interval: ur_run_interval,
        false_discovery_rate_mean: fdr_mean,
        false_discovery_rate_run_interval: fdr_run_interval,
        human_reviews_mean: hr_mean,
        estimated_hidden_unconventional_mean: hidden_mean,
    }
}

fn compute_percentiles(vals: &[f64], p_low: f64, p_high: f64) -> (f64, f64) {
    let mut sorted = vals.to_vec();
    sorted.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let n = sorted.len();
    let idx_low = ((n as f64 * p_low).floor() as usize).min(n - 1);
    let idx_high = ((n as f64 * p_high).floor() as usize).min(n - 1);
    (sorted[idx_low], sorted[idx_high])
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::StdRng;
    use rand::SeedableRng;

    #[test]
    fn test_zero_noise_does_not_panic() {
        let mut rng = StdRng::seed_from_u64(42);
        let params = Parameters {
            unmanaged_baseline_noise: 0.0,
            initial_screen_noise: 0.0,
            fast_review_noise: 0.0,
            deep_review_noise: 0.0,
            ..Default::default()
        };

        let candidates = generate_candidates(&mut rng, 100, 0.1);
        let s1 = run_unmanaged_screening(&mut rng, candidates.clone(), &params);
        assert_eq!(s1.accepted, params.acceptance_capacity.min(100));

        let s2 = run_voi_screening(&mut rng, candidates, &params, 0.05);
        assert_eq!(s2.accepted, params.acceptance_capacity.min(100));
    }

    #[test]
    fn test_extreme_budget_depletion_does_not_panic() {
        let mut rng = StdRng::seed_from_u64(42);
        let params = Parameters {
            evaluation_budget: 1.0,
            initial_screen_cost: 10.0, // Initial cost will greatly exceed budget
            ..Default::default()
        };

        let candidates = generate_candidates(&mut rng, 200, 0.1);
        let s = run_voi_screening(&mut rng, candidates, &params, 0.05);
        assert_eq!(s.accepted, params.acceptance_capacity.min(200));
    }

    #[test]
    fn test_top_indices_ordering() {
        let values = vec![1.2, 5.5, 3.1, -0.4, 9.8];
        let top3 = top_indices(&values, 3);
        assert_eq!(top3, vec![4, 1, 2]); // indices of 9.8, 5.5, 3.1
    }

    #[test]
    fn test_aggregate_stats_empty_data_does_not_panic() {
        let stats = aggregate_stats("empty_regime", 100, 10, &[]);
        assert_eq!(stats.arrivals, 100);
        assert_eq!(stats.quality_throughput_mean, 0.0);
    }
}
