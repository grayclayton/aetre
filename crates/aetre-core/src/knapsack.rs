//! Pillar V: Capacity-Aware Knapsack Admission (Rust Engine).
//!
//! Regulates downstream review and staging queues according to the
//! macroeconomic and absorption constraints established in Gray (2026a, b):
//! - Enforces optimal selection depth m* = min(n, floor(K/k)) under uniform workloads
//! - Implements density ranking rho_i = E[U_i] / k_i and queuing-theoretic c-mu scheduling
//! - Tracks fractional review capacity, total admitted effort, and dual shadow price Delta_K W
//! - Prevents candidate pull request flooding under bounded review bandwidth.

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CandidateSubmission {
    pub candidate_id: String,
    pub task_id: String,
    pub posterior_belief: f64,
    pub reward: f64,
    pub loss: f64,
    pub review_cost: Option<f64>,
}

impl CandidateSubmission {
    pub fn new(
        candidate_id: impl Into<String>,
        task_id: impl Into<String>,
        posterior_belief: f64,
        reward: Option<f64>,
        loss: Option<f64>,
        review_cost: Option<f64>,
    ) -> Self {
        CandidateSubmission {
            candidate_id: candidate_id.into(),
            task_id: task_id.into(),
            posterior_belief,
            reward: reward.unwrap_or(0.02),
            loss: loss.unwrap_or(0.10),
            review_cost: review_cost.filter(|&c| c > 0.0),
        }
    }

    #[inline]
    pub fn expected_utility(&self) -> f64 {
        self.posterior_belief * self.reward - (1.0 - self.posterior_belief) * self.loss
    }

    #[inline]
    pub fn p_star(&self) -> f64 {
        self.loss / (self.reward + self.loss)
    }

    #[inline]
    pub fn is_intrinsically_acceptable(&self) -> bool {
        self.posterior_belief >= self.p_star()
    }

    #[inline]
    pub fn effective_cost(&self, default_cost: f64) -> f64 {
        self.review_cost.unwrap_or(default_cost)
    }

    #[inline]
    pub fn density(&self) -> f64 {
        self.expected_utility() / self.effective_cost(1.0)
    }

    #[inline]
    pub fn density_with_cost(&self, default_cost: f64) -> f64 {
        self.expected_utility() / self.effective_cost(default_cost)
    }

    #[inline]
    pub fn c_mu_index(&self) -> f64 {
        self.density()
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct KnapsackAdmissionReport {
    pub capacity_k: f64,
    pub review_cost_k: f64,
    pub optimal_selection_depth: usize,
    pub total_candidates: usize,
    pub total_admitted: usize,
    pub total_rejected: usize,
    pub total_welfare: f64,
    pub shadow_price_lambda: f64,
    pub total_admitted_cost: f64,
    pub remaining_capacity: f64,
    pub admitted: Vec<CandidateSubmission>,
    pub rejected: Vec<CandidateSubmission>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KnapsackController {
    pub capacity_k: f64,
    pub review_cost_k: f64,
}

impl KnapsackController {
    pub fn new(default_capacity: f64, review_cost_k: f64) -> Result<Self, String> {
        if default_capacity <= 0.0 || review_cost_k <= 0.0 {
            return Err("Capacity and review cost must be positive".to_string());
        }
        Ok(KnapsackController {
            capacity_k: default_capacity,
            review_cost_k,
        })
    }

    pub fn admit_batch(
        &self,
        candidates: &[CandidateSubmission],
        capacity_k: Option<f64>,
        review_cost_k: Option<f64>,
    ) -> Result<KnapsackAdmissionReport, String> {
        let k_cap = capacity_k.unwrap_or(self.capacity_k);
        let k_cost = review_cost_k.unwrap_or(self.review_cost_k);
        if k_cap <= 0.0 || k_cost <= 0.0 {
            return Err("Capacity and review cost must be positive".to_string());
        }

        let n = candidates.len();

        // Check if workload is strictly homogeneous with respect to k_cost
        // Candidates with review_cost == None inherit k_cost
        let is_homogeneous = candidates
            .iter()
            .all(|c| c.review_cost.is_none() || (c.review_cost.unwrap() - k_cost).abs() < 1e-9);

        // Rank candidates descending by value density rho_i = E[U_i] / k_i (the c-mu rule)
        // breaking ties by expected utility
        let mut ranked: Vec<CandidateSubmission> = candidates.to_vec();
        ranked.sort_by(|a, b| {
            let dens_b = b.density_with_cost(k_cost);
            let dens_a = a.density_with_cost(k_cost);
            dens_b
                .partial_cmp(&dens_a)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| {
                    b.expected_utility()
                        .partial_cmp(&a.expected_utility())
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
        });

        let mut admitted: Vec<CandidateSubmission> = Vec::new();
        let mut rejected: Vec<CandidateSubmission> = Vec::new();
        let mut total_admitted_cost = 0.0;

        for cand in ranked.iter() {
            let cost = cand.effective_cost(k_cost);
            if cand.is_intrinsically_acceptable() && (total_admitted_cost + cost <= k_cap + 1e-9) {
                total_admitted_cost += cost;
                admitted.push(cand.clone());
            } else {
                rejected.push(cand.clone());
            }
        }

        let total_welfare: f64 = admitted.iter().map(|c| c.expected_utility()).sum();
        let remaining_capacity = (k_cap - total_admitted_cost).max(0.0);

        // Calculate shadow price Delta_K W
        let shadow_price =
            if is_homogeneous && k_cap.fract().abs() < 1e-9 && k_cost.fract().abs() < 1e-9 {
                let cap_int = k_cap.round() as usize;
                let cost_int = k_cost.round() as usize;
                let m_star = n.min(cap_int / cost_int);
                let next_depth = n.min((cap_int + 1) / cost_int);
                if next_depth == m_star || admitted.len() < m_star {
                    0.0
                } else if m_star < n && ranked[m_star].is_intrinsically_acceptable() {
                    ranked[m_star].expected_utility()
                } else {
                    0.0
                }
            } else {
                let first_unadmitted_acceptable = ranked.iter().find(|c| {
                    c.is_intrinsically_acceptable()
                        && !admitted.iter().any(|a| a.candidate_id == c.candidate_id)
                });
                match first_unadmitted_acceptable {
                    Some(cand) => cand.density_with_cost(k_cost),
                    None => 0.0,
                }
            };

        let optimal_selection_depth =
            if is_homogeneous && k_cap.fract().abs() < 1e-9 && k_cost.fract().abs() < 1e-9 {
                let cap_int = k_cap.round() as usize;
                let cost_int = k_cost.round() as usize;
                n.min(cap_int / cost_int)
            } else {
                admitted.len()
            };

        Ok(KnapsackAdmissionReport {
            capacity_k: k_cap,
            review_cost_k: k_cost,
            optimal_selection_depth,
            total_candidates: n,
            total_admitted: admitted.len(),
            total_rejected: rejected.len(),
            total_welfare,
            shadow_price_lambda: shadow_price,
            total_admitted_cost,
            remaining_capacity,
            admitted,
            rejected,
        })
    }
}
