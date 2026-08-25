use serde::{Deserialize, Serialize};

/// Submitter entry equilibrium analysis under generation costs and staking fees
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SubmitterEquilibrium {
    pub generation_cost: f64,
    pub submission_fee: f64,
    pub private_acceptance_value: f64,
    pub total_potential_applicants: usize,
    pub threshold_acceptance_prob: f64,
    pub estimated_entry_volume: f64,
    pub low_quality_spam_deterred_pct: f64,
}

/// Point on a staking fee parameter sweep curve
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StakingCurvePoint {
    pub submission_fee: f64,
    pub threshold_acceptance_prob: f64,
    pub estimated_entry_volume: f64,
    pub low_quality_spam_deterred_pct: f64,
}

/// Evaluates applicant entry condition: P(a | s, lambda) * V - (c_gen + c_sub) >= 0
pub fn evaluate_submitter_equilibrium(
    c_gen: f64,
    c_sub: f64,
    private_acceptance_value: f64,
    total_potential_applicants: usize,
    acceptance_capacity: usize,
) -> SubmitterEquilibrium {
    let total_cost = c_gen + c_sub;
    let threshold_acceptance_prob = if private_acceptance_value > 0.0 {
        (total_cost / private_acceptance_value).clamp(0.0, 1.0)
    } else {
        1.0
    };

    // If costs are 0, everyone enters (N)
    // As threshold_acceptance_prob rises, low-signal applicants self-censor
    let entry_rate = if threshold_acceptance_prob <= 0.0 {
        1.0
    } else {
        // Assume signal distribution ~ standard normal, entry drops when required prob is high
        let baseline_acceptance_rate =
            acceptance_capacity as f64 / total_potential_applicants.max(1) as f64;
        let relative_barrier = threshold_acceptance_prob / baseline_acceptance_rate.max(1e-4);
        (1.0 / (1.0 + relative_barrier)).clamp(0.05, 1.0)
    };

    let estimated_entry_volume = total_potential_applicants as f64 * entry_rate;
    let baseline_entry = total_potential_applicants as f64;
    let low_quality_spam_deterred_pct = if baseline_entry > 0.0 {
        ((baseline_entry - estimated_entry_volume) / baseline_entry * 100.0).max(0.0)
    } else {
        0.0
    };

    SubmitterEquilibrium {
        generation_cost: c_gen,
        submission_fee: c_sub,
        private_acceptance_value,
        total_potential_applicants,
        threshold_acceptance_prob,
        estimated_entry_volume,
        low_quality_spam_deterred_pct,
    }
}

/// Generates a sensitivity curve of entry volume and spam deterrence across fee levels
pub fn generate_staking_curve(
    c_gen: f64,
    private_acceptance_value: f64,
    total_potential_applicants: usize,
    acceptance_capacity: usize,
    max_fee: f64,
    steps: usize,
) -> Vec<StakingCurvePoint> {
    let step_size = if steps > 1 {
        max_fee / (steps - 1) as f64
    } else {
        max_fee
    };
    (0..steps)
        .map(|i| {
            let fee = i as f64 * step_size;
            let eq = evaluate_submitter_equilibrium(
                c_gen,
                fee,
                private_acceptance_value,
                total_potential_applicants,
                acceptance_capacity,
            );
            StakingCurvePoint {
                submission_fee: fee,
                threshold_acceptance_prob: eq.threshold_acceptance_prob,
                estimated_entry_volume: eq.estimated_entry_volume,
                low_quality_spam_deterred_pct: eq.low_quality_spam_deterred_pct,
            }
        })
        .collect()
}

use crate::types::QuadraticStakingResult;

/// Evaluates super-linear / quadratic anti-sybil staking fee escalation
///
/// Under quadratic escalation: MarginalStake(m) = S_0 * m^gamma
/// This makes mass AI-generation submissions economically prohibitive for spam rings,
/// while keeping the barrier low for single-proposal human/authentic innovators.
pub fn evaluate_quadratic_staking(
    base_fee: f64,
    escalation_exponent: f64,
    submission_count: usize,
    c_gen: f64,
    private_acceptance_value: f64,
) -> QuadraticStakingResult {
    let m = submission_count.max(1);
    let gamma = escalation_exponent.max(1.0);
    let s0 = base_fee.max(0.0);

    let mut total_stake_required = 0.0;
    for k in 1..=m {
        total_stake_required += s0 * (k as f64).powf(gamma);
    }

    let marginal_stake_for_next = s0 * ((m + 1) as f64).powf(gamma);
    let marginal_total_cost = c_gen + marginal_stake_for_next;

    let required_prob = if private_acceptance_value > 0.0 {
        (marginal_total_cost / private_acceptance_value).clamp(0.0, 1.0)
    } else {
        1.0
    };

    let spam_deterrence_pct = (required_prob * 100.0).clamp(0.0, 100.0);

    QuadraticStakingResult {
        base_fee: s0,
        escalation_exponent: gamma,
        submission_count: m,
        total_stake_required,
        marginal_stake_for_next,
        spam_deterrence_pct,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_staking_deters_entry() {
        let no_fee = evaluate_submitter_equilibrium(0.01, 0.0, 100.0, 5000, 200);
        let with_fee = evaluate_submitter_equilibrium(0.01, 5.0, 100.0, 5000, 200);
        assert!(with_fee.estimated_entry_volume < no_fee.estimated_entry_volume);
        assert!(with_fee.low_quality_spam_deterred_pct > no_fee.low_quality_spam_deterred_pct);
    }

    #[test]
    fn test_staking_monotonicity() {
        let curve = generate_staking_curve(0.05, 100.0, 5000, 200, 20.0, 10);
        assert_eq!(curve.len(), 10);
        for w in curve.windows(2) {
            assert!(w[0].estimated_entry_volume >= w[1].estimated_entry_volume);
            assert!(w[0].low_quality_spam_deterred_pct <= w[1].low_quality_spam_deterred_pct);
        }
    }

    #[test]
    fn test_quadratic_staking_escalation() {
        let res_1 = evaluate_quadratic_staking(5.0, 2.0, 1, 0.01, 100.0);
        let res_5 = evaluate_quadratic_staking(5.0, 2.0, 5, 0.01, 100.0);
        let res_10 = evaluate_quadratic_staking(5.0, 2.0, 10, 0.01, 100.0);

        assert_eq!(res_1.total_stake_required, 5.0);
        assert!(res_5.total_stake_required > res_1.total_stake_required * 5.0); // Super-linear
        assert!(res_10.marginal_stake_for_next > res_5.marginal_stake_for_next);
        assert!(res_10.spam_deterrence_pct >= res_1.spam_deterrence_pct);
    }
}
