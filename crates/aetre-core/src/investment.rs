use crate::types::{VentureBenchmarkComparison, VentureCohortSummary, VentureDealCandidate};
use crate::voi::calculate_heavy_tailed_voi;

fn pseudo_rand(seed: u64) -> f64 {
    let mut x = seed.wrapping_mul(0xdaba_0b69_7093_06c5);
    x ^= x >> 32;
    x = x.wrapping_mul(0x4bba_c779_fa70_4735);
    x ^= x >> 28;
    ((x >> 11) as f64) / ((1u64 << 53) as f64)
}

/// Generates a deterministic synthetic cohort of venture deals with power-law Pareto outcomes
pub fn generate_synthetic_venture_dealflow(
    n_deals: usize,
    tail_alpha: f64,
    wrapper_pct: f64,
    selection_boundary: f64,
) -> Vec<VentureDealCandidate> {
    let mut deals = Vec::with_capacity(n_deals);
    let sectors = [
        "AI & Autonomous Systems",
        "DeepTech & Quantum",
        "Biotech & Synthetic Bio",
        "ClimateTech & Fusion",
        "Fintech & Crypto Infrastructure",
        "B2B Enterprise SaaS",
    ];

    for i in 0..n_deals {
        let p1 = pseudo_rand((i as u64) * 3 + 1);
        let p2 = pseudo_rand((i as u64) * 3 + 2);
        let p3 = pseudo_rand((i as u64) * 3 + 3);

        let sector = sectors[i % sectors.len()].to_string();
        let is_wrapper = p1 < wrapper_pct;

        let (prelim_score, epistemic_var, true_multiplier, is_unicorn) = if is_wrapper {
            // Commodity wrapper: polished buzzwords give high superficial score, but low variance and 0x return
            let score = 5.8 + p2 * 1.2;
            (score, 0.12, 0.0, false)
        } else if p2 < 0.08 {
            // High-novelty black swan unicorn candidate (8% of non-wrappers)
            // Polarizing / unconventional: preliminary score is mixed (4.8 - 6.0), but variance is HIGH
            let u = p3.clamp(0.01, 0.98);
            let pareto_mult = 15.0 / (1.0 - u).powf(1.0 / tail_alpha.max(1.05));
            let multiplier = pareto_mult.clamp(15.0, 150.0);

            let score = 5.0 + p1 * 1.0;
            let var = 0.90 + p3 * 0.50;
            (score, var, multiplier, true)
        } else if p2 < 0.28 {
            // Solid base hit (20% of non-wrappers)
            let multiplier = 2.0 + p3 * 4.0; // 2x - 6x
            let score = 5.4 + p1 * 1.2;
            let var = 0.30 + p3 * 0.20;
            (score, var, multiplier, false)
        } else {
            // Standard failure / write-off
            let multiplier = p3 * 0.6; // 0x - 0.6x
            let score = 3.2 + p1 * 2.2;
            let var = 0.20 + p3 * 0.20;
            (score, var, multiplier, false)
        };

        // Calculate heavy-tailed VOI
        let voi_res = calculate_heavy_tailed_voi(
            prelim_score,
            epistemic_var,
            selection_boundary,
            tail_alpha,
            0.80,
            1.0,
        );

        deals.push(VentureDealCandidate {
            deal_id: format!("deal_{:04}", i + 1),
            company_name: format!(
                "Startup_{:04}_{}",
                i + 1,
                sector.split_whitespace().next().unwrap_or("Tech")
            ),
            sector,
            preliminary_score: prelim_score,
            epistemic_variance: epistemic_var,
            true_payoff_multiplier: true_multiplier,
            is_commodity_wrapper: is_wrapper,
            is_unicorn_outlier: is_unicorn,
            heavy_tailed_voi: voi_res.voi_index,
        });
    }

    deals
}

/// Runs a comparative benchmark between Status-Quo Screener and AETRE Heavy-Tailed VOI
pub fn evaluate_venture_benchmark(
    deals: &[VentureDealCandidate],
    diligence_budget: usize,
    tail_alpha: f64,
    hours_per_diligence: f64,
) -> VentureBenchmarkComparison {
    let k = diligence_budget.min(deals.len()).max(1);
    let total_unicorns = deals.iter().filter(|d| d.is_unicorn_outlier).count();

    // 1. Status Quo Strategy: Rank purely by preliminary score descending
    let mut sq_deals = deals.to_vec();
    sq_deals.sort_by(|a, b| {
        b.preliminary_score
            .partial_cmp(&a.preliminary_score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let sq_selected = &sq_deals[..k];

    let sq_moic = sq_selected
        .iter()
        .map(|d| d.true_payoff_multiplier)
        .sum::<f64>()
        / (k as f64);
    let sq_irr = if sq_moic > 0.0 {
        (sq_moic.powf(1.0 / 5.0) - 1.0).max(-1.0)
    } else {
        -1.0
    };
    let sq_unicorns = sq_selected.iter().filter(|d| d.is_unicorn_outlier).count();
    let sq_wrappers_invested = sq_selected
        .iter()
        .filter(|d| d.is_commodity_wrapper)
        .count();
    let sq_wrappers_avoided =
        deals.iter().filter(|d| d.is_commodity_wrapper).count() - sq_wrappers_invested;
    let sq_hours_per_uni = if sq_unicorns > 0 {
        ((k as f64) * hours_per_diligence) / (sq_unicorns as f64)
    } else {
        (k as f64) * hours_per_diligence
    };

    let status_quo_summary = VentureCohortSummary {
        strategy_name: "Status Quo (Score-Only Screener)".to_string(),
        deals_evaluated: deals.len(),
        diligence_deals_selected: k,
        portfolio_moic: sq_moic,
        portfolio_irr_approx: sq_irr,
        unicorns_captured: sq_unicorns,
        total_unicorns,
        outlier_recall: if total_unicorns > 0 {
            sq_unicorns as f64 / total_unicorns as f64
        } else {
            0.0
        },
        diligence_hours_per_unicorn: sq_hours_per_uni,
        wrappers_avoided_count: sq_wrappers_avoided,
        wrappers_invested_count: sq_wrappers_invested,
    };

    // 2. AETRE Strategy: Rank by Heavy-Tailed Bayesian VOI descending
    let mut aetre_deals = deals.to_vec();
    aetre_deals.sort_by(|a, b| {
        b.heavy_tailed_voi
            .partial_cmp(&a.heavy_tailed_voi)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let aetre_selected = &aetre_deals[..k];

    let aetre_moic = aetre_selected
        .iter()
        .map(|d| d.true_payoff_multiplier)
        .sum::<f64>()
        / (k as f64);
    let aetre_irr = if aetre_moic > 0.0 {
        (aetre_moic.powf(1.0 / 5.0) - 1.0).max(-1.0)
    } else {
        -1.0
    };
    let aetre_unicorns = aetre_selected
        .iter()
        .filter(|d| d.is_unicorn_outlier)
        .count();
    let aetre_wrappers_invested = aetre_selected
        .iter()
        .filter(|d| d.is_commodity_wrapper)
        .count();
    let aetre_wrappers_avoided =
        deals.iter().filter(|d| d.is_commodity_wrapper).count() - aetre_wrappers_invested;
    let aetre_hours_per_uni = if aetre_unicorns > 0 {
        ((k as f64) * hours_per_diligence) / (aetre_unicorns as f64)
    } else {
        (k as f64) * hours_per_diligence
    };

    let aetre_summary = VentureCohortSummary {
        strategy_name: "AETRE (Heavy-Tailed VOI Triage)".to_string(),
        deals_evaluated: deals.len(),
        diligence_deals_selected: k,
        portfolio_moic: aetre_moic,
        portfolio_irr_approx: aetre_irr,
        unicorns_captured: aetre_unicorns,
        total_unicorns,
        outlier_recall: if total_unicorns > 0 {
            aetre_unicorns as f64 / total_unicorns as f64
        } else {
            0.0
        },
        diligence_hours_per_unicorn: aetre_hours_per_uni,
        wrappers_avoided_count: aetre_wrappers_avoided,
        wrappers_invested_count: aetre_wrappers_invested,
    };

    let moic_gain = if sq_moic > 0.0 {
        aetre_moic / sq_moic
    } else {
        1.0
    };
    let irr_uplift = (aetre_irr - sq_irr) * 100.0;
    let hours_saved_pct = if sq_hours_per_uni > 0.0 {
        (1.0 - (aetre_hours_per_uni / sq_hours_per_uni)) * 100.0
    } else {
        0.0
    };
    let recall_improvement = if sq_unicorns > 0 {
        aetre_unicorns as f64 / sq_unicorns as f64
    } else {
        1.0
    };

    VentureBenchmarkComparison {
        total_dealflow_universe: deals.len(),
        diligence_budget: k,
        tail_index_alpha: tail_alpha,
        status_quo: status_quo_summary,
        aetre_heavy_tailed: aetre_summary,
        moic_multiplier_gain: moic_gain,
        irr_percentage_point_uplift: irr_uplift,
        partner_hours_saved_pct: hours_saved_pct,
        outlier_capture_improvement_ratio: recall_improvement,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_synthetic_dealflow_generation() {
        let deals = generate_synthetic_venture_dealflow(500, 1.25, 0.25, 6.0);
        assert_eq!(deals.len(), 500);
        let unicorns = deals.iter().filter(|d| d.is_unicorn_outlier).count();
        assert!(unicorns > 5, "Should generate black swan unicorns");
        let wrappers = deals.iter().filter(|d| d.is_commodity_wrapper).count();
        assert!(wrappers > 50, "Should generate commodity wrappers");
    }

    #[test]
    fn test_aetre_outperforms_status_quo_in_heavy_tailed_venture() {
        let deals = generate_synthetic_venture_dealflow(1000, 1.25, 0.30, 6.0);
        let bench = evaluate_venture_benchmark(&deals, 50, 1.25, 20.0);

        assert!(
            bench.aetre_heavy_tailed.unicorns_captured >= bench.status_quo.unicorns_captured,
            "AETRE VOI should capture at least as many or more unicorns as status quo"
        );
        assert!(
            bench.aetre_heavy_tailed.portfolio_moic >= bench.status_quo.portfolio_moic,
            "AETRE VOI should deliver higher or equal portfolio MOIC"
        );
        assert!(
            bench.aetre_heavy_tailed.wrappers_invested_count
                <= bench.status_quo.wrappers_invested_count,
            "AETRE should filter more commodity wrappers than status quo"
        );
    }
}
