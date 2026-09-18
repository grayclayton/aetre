//! Pillar I: Decoupled Economic Governor (Rust Engine).
//!
//! Implements the Bayesian Bellman supervisory controller that regulates
//! verification effort, evaluates multi-step Value of Information (VOI),
//! and enforces stopping and admission rules under asymmetric loss stakes.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Action {
    Accept,
    Reject,
    Probe,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Decision {
    pub action: String,
    pub stage: usize,
    pub belief: f64,
    pub expected_utility: f64,
    pub probe_idx: Option<usize>,
    pub voi: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ReviewBoundaryDecision {
    pub action: String,
    pub u_auto: f64,
    pub u_review: f64,
    pub u_abstain: f64,
    pub belief: f64,
    pub shadow_price_lambda: f64,
    pub dominant_utility: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Governor {
    pub reward: f64,
    pub loss: f64,
    pub prior: f64,
    pub max_stages: usize,
    pub p_star: f64,
    pub defect_leakage: f64,
    pub conforming_pass_rate: f64,
    pub cost_schedule: Vec<f64>,
    dp_cache: HashMap<(usize, usize), f64>,
    policy_cache: HashMap<(usize, usize), Action>,
}

impl Governor {
    pub fn new(
        reward: f64,
        loss: f64,
        prior: f64,
        max_stages: usize,
        cost_schedule: Option<Vec<f64>>,
        defect_leakage: Option<f64>,
        conforming_pass_rate: Option<f64>,
    ) -> Result<Self, String> {
        if reward <= 0.0 || loss <= 0.0 {
            return Err("Reward and loss must be strictly positive".to_string());
        }
        if !(0.0 < prior && prior < 1.0) {
            return Err("Prior belief must be in (0, 1)".to_string());
        }
        let q = defect_leakage.unwrap_or(0.6125);
        if !(0.0..=1.0).contains(&q) {
            return Err("Defect leakage q must be in [0, 1]".to_string());
        }
        let p_conf = conforming_pass_rate.unwrap_or(1.0);

        let p_star = loss / (reward + loss);
        let schedule = match cost_schedule {
            Some(cs) => {
                if cs.len() != max_stages {
                    return Err(format!(
                        "Cost schedule length ({}) must equal max_stages ({})",
                        cs.len(),
                        max_stages
                    ));
                }
                cs
            }
            None => vec![1e-6; max_stages],
        };

        let mut gov = Governor {
            reward,
            loss,
            prior,
            max_stages,
            p_star,
            defect_leakage: q,
            conforming_pass_rate: p_conf,
            cost_schedule: schedule,
            dp_cache: HashMap::new(),
            policy_cache: HashMap::new(),
        };

        gov.solve_dp();
        Ok(gov)
    }

    #[inline]
    pub fn terminal_utility(&self, belief: f64) -> f64 {
        let accept_utility = belief * self.reward - (1.0 - belief) * self.loss;
        accept_utility.max(0.0)
    }

    pub fn posterior_belief(&self, prior_b: f64, outcome_pass: bool, _probe_idx: usize) -> f64 {
        let p_pass_given_conf = self.conforming_pass_rate;
        let p_pass_given_def = self.defect_leakage;

        let (num, den) = if outcome_pass {
            let num = prior_b * p_pass_given_conf;
            let den = num + (1.0 - prior_b) * p_pass_given_def;
            (num, den)
        } else {
            let num = prior_b * (1.0 - p_pass_given_conf);
            let den = num + (1.0 - prior_b) * (1.0 - p_pass_given_def);
            (num, den)
        };

        if den <= 0.0 {
            if !outcome_pass {
                0.0
            } else {
                1.0
            }
        } else {
            (num / den).clamp(0.0, 1.0)
        }
    }

    pub fn belief_at_pass_count(&self, pass_count: usize) -> f64 {
        let q_eff = self.defect_leakage.powi(pass_count as i32);
        let p_conf = self.conforming_pass_rate.powi(pass_count as i32);
        let num = self.prior * p_conf;
        let den = num + (1.0 - self.prior) * q_eff;
        if den <= 0.0 {
            0.0
        } else {
            num / den
        }
    }

    fn solve_dp(&mut self) {
        self.dp_cache.clear();
        self.policy_cache.clear();

        // Base cases at max_stages (horizon H)
        for k in 0..=self.max_stages {
            let b_k = self.belief_at_pass_count(k);
            let u_term = self.terminal_utility(b_k);
            self.dp_cache.insert((self.max_stages, k), u_term);
            let action = if b_k >= self.p_star {
                Action::Accept
            } else {
                Action::Reject
            };
            self.policy_cache.insert((self.max_stages, k), action);
        }

        // Backward induction from t = H-1 down to 0
        for t in (0..self.max_stages).rev() {
            let cost = self.cost_schedule[t];
            for k in 0..=t {
                let b = self.belief_at_pass_count(k);
                let u_stop = self.terminal_utility(b);

                let p_pass = b * self.conforming_pass_rate + (1.0 - b) * self.defect_leakage;
                let p_fail = 1.0 - p_pass;

                let v_pass = self.dp_cache[&(t + 1, k + 1)];
                let v_fail = 0.0;

                let v_continue = -cost + (p_pass * v_pass + p_fail * v_fail);

                if v_continue > u_stop + 1e-12 {
                    self.dp_cache.insert((t, k), v_continue);
                    self.policy_cache.insert((t, k), Action::Probe);
                } else {
                    self.dp_cache.insert((t, k), u_stop);
                    let action = if b >= self.p_star {
                        Action::Accept
                    } else {
                        Action::Reject
                    };
                    self.policy_cache.insert((t, k), action);
                }
            }
        }
    }

    pub fn evaluate_state(&self, stage: usize, consecutive_passes: usize) -> Decision {
        let stage_clamped = stage.min(self.max_stages);
        let passes_clamped = consecutive_passes.min(stage_clamped);

        let b = self.belief_at_pass_count(passes_clamped);
        let u_stop = self.terminal_utility(b);
        let expected_v = *self
            .dp_cache
            .get(&(stage_clamped, passes_clamped))
            .unwrap_or(&u_stop);
        let action_name = *self
            .policy_cache
            .get(&(stage_clamped, passes_clamped))
            .unwrap_or(&Action::Reject);

        let mut voi = 0.0;
        if stage_clamped < self.max_stages {
            let p_pass = b * self.conforming_pass_rate + (1.0 - b) * self.defect_leakage;
            let _p_fail = 1.0 - p_pass;
            let v_pass = *self
                .dp_cache
                .get(&(stage_clamped + 1, passes_clamped + 1))
                .unwrap_or(&0.0);
            let e_next = p_pass * v_pass;
            voi = (e_next - u_stop).max(0.0);
        }

        match action_name {
            Action::Probe => Decision {
                action: "CONTINUE".to_string(),
                stage: stage_clamped,
                belief: b,
                expected_utility: expected_v,
                probe_idx: Some(stage_clamped),
                voi,
            },
            Action::Accept => Decision {
                action: "HALT_AND_COMMIT".to_string(),
                stage: stage_clamped,
                belief: b,
                expected_utility: expected_v,
                probe_idx: None,
                voi: 0.0,
            },
            Action::Reject => Decision {
                action: "HALT_AND_REJECT".to_string(),
                stage: stage_clamped,
                belief: b,
                expected_utility: 0.0,
                probe_idx: None,
                voi: 0.0,
            },
        }
    }

    pub fn is_myopic_paralyzed(&self, stage: usize, consecutive_passes: usize) -> bool {
        let b = self.belief_at_pass_count(consecutive_passes);
        let u_stop = self.terminal_utility(b);
        if stage >= self.max_stages {
            return false;
        }

        let cost = self.cost_schedule[stage];
        let p_pass = b * self.conforming_pass_rate + (1.0 - b) * self.defect_leakage;
        let b_next = self.posterior_belief(b, true, stage);
        let u_next_term = self.terminal_utility(b_next);
        let myopic_continue_v = -cost + p_pass * u_next_term;

        let myopic_wants_stop = myopic_continue_v <= u_stop;
        let dp_wants_continue =
            self.policy_cache.get(&(stage, consecutive_passes)) == Some(&Action::Probe);
        myopic_wants_stop && dp_wants_continue
    }

    pub fn dp_table(&self) -> HashMap<(usize, usize), f64> {
        self.dp_cache.clone()
    }

    /// Evaluates the tripartite review boundary from Section 4.1 (Gray 2026d).
    ///
    /// Compares:
    /// 1. E[U(auto, theta) | z_t] = max(0, b * R - (1 - b) * L)
    /// 2. E[U(review, theta) | z_t] = (b * R * acc) - review_cost - lambda_K
    /// 3. E[U(abstain, theta) | z_t] = 0.0
    pub fn evaluate_review_boundary(
        &self,
        belief: f64,
        review_cost: f64,
        shadow_price_lambda: f64,
        review_accuracy: f64,
    ) -> Result<ReviewBoundaryDecision, String> {
        if !(0.0..=1.0).contains(&belief) {
            return Err("Belief must be in [0, 1]".to_string());
        }
        if review_cost < 0.0 {
            return Err("Review cost must be non-negative".to_string());
        }
        if shadow_price_lambda < 0.0 {
            return Err("Shadow price lambda must be non-negative".to_string());
        }
        if !(0.0..=1.0).contains(&review_accuracy) {
            return Err("Review accuracy must be in [0, 1]".to_string());
        }

        let u_auto = belief * self.reward - (1.0 - belief) * self.loss;
        let u_review = (belief * self.reward * review_accuracy) - review_cost - shadow_price_lambda;
        let u_abstain = 0.0;

        let (action, dom) = if u_auto >= u_review.max(u_abstain) {
            ("AUTO".to_string(), u_auto)
        } else if u_review > u_abstain {
            ("REVIEW".to_string(), u_review)
        } else {
            ("ABSTAIN".to_string(), u_abstain)
        };

        Ok(ReviewBoundaryDecision {
            action,
            u_auto,
            u_review,
            u_abstain,
            belief,
            shadow_price_lambda,
            dominant_utility: dom,
        })
    }
}
