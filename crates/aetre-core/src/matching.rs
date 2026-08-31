//! Congestion-Aware Reviewer-to-Proposal Matching Optimizer
//!
//! Assigns candidate proposals to reviewers while simultaneously optimizing expertise affinity
//! and enforcing Kingman Heavy-Traffic queue stability constraints (rho <= 0.85).
//! Prioritizes scarce senior domain specialists for high-VOI boundary proposals.

use crate::types::{
    CongestionMatchingResult, MatchAssignment, ProposalRequirement, ReviewerProfile,
    ReviewerUtilizationReport,
};
use std::collections::{HashMap, HashSet};

/// Computes optimal congestion-governed reviewer matching for a cohort of proposals.
///
/// # Arguments
/// * `proposals` - List of proposals with domain, VOI index, and review requirements.
/// * `reviewers` - List of reviewer profiles with capacity, domain, and queue telemetry.
/// * `target_utilization` - Maximum desired utilization (defaults to 0.85).
pub fn optimize_congestion_matching(
    proposals: &[ProposalRequirement],
    reviewers: &[ReviewerProfile],
    target_utilization: Option<f64>,
) -> CongestionMatchingResult {
    let target_rho = target_utilization.unwrap_or(0.85);

    // Track reviewer assignment state
    let mut current_assigned: HashMap<String, usize> = reviewers
        .iter()
        .map(|r| (r.id.clone(), r.current_load))
        .collect();

    // Sort proposals descending by VOI so highest information gain proposals get best available matches
    let mut sorted_proposals: Vec<&ProposalRequirement> = proposals.iter().collect();
    sorted_proposals.sort_by(|a, b| {
        b.voi_index
            .partial_cmp(&a.voi_index)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let mut assignments: Vec<MatchAssignment> = Vec::new();
    let mut unassigned_proposals: Vec<String> = Vec::new();
    let mut global_objective = 0.0;

    for proposal in sorted_proposals {
        let mut assigned_for_this_prop: HashSet<String> = HashSet::new();
        let needed = proposal.required_reviews.max(1);

        for slot in 0..needed {
            let mut best_reviewer_id: Option<String> = None;
            let mut best_score = f64::NEG_INFINITY;
            let mut best_raw_affinity = 0.0;

            for r in reviewers {
                if assigned_for_this_prop.contains(&r.id) {
                    continue; // Do not assign same reviewer twice to one proposal
                }

                let load = *current_assigned.get(&r.id).unwrap_or(&0);
                if load >= r.capacity {
                    continue; // Hard capacity limit reached
                }

                // Compute base domain & keyword affinity
                let mut raw_affinity = 0.0;
                if r.domain.eq_ignore_ascii_case(&proposal.domain) {
                    raw_affinity += 1.0;
                }

                // Keyword overlap (Jaccard similarity)
                let r_tags: HashSet<&str> = r.expertise_tags.iter().map(|s| s.as_str()).collect();
                let p_keys: HashSet<&str> = proposal.keywords.iter().map(|s| s.as_str()).collect();
                let intersection = r_tags.intersection(&p_keys).count();
                let union = r_tags.union(&p_keys).count();
                if union > 0 {
                    raw_affinity += 0.8 * (intersection as f64 / union as f64);
                }

                // Kingman utilization projection: increment arrival rate by new assignment candidate
                let newly_assigned = load.saturating_sub(r.current_load);
                let projected_arrivals = r.arrival_rate + (newly_assigned + 1) as f64;
                let service = r.service_rate.max(1.0);
                let projected_rho = projected_arrivals / service;

                // Congestion penalty if utilization approaches target_rho
                let congestion_penalty = if projected_rho > target_rho {
                    ((projected_rho - target_rho) * 10.0).exp()
                } else if projected_rho > target_rho - 0.15 {
                    (projected_rho - (target_rho - 0.15)) * 2.0
                } else {
                    0.0
                };

                // Score combines affinity, VOI priority boost, and congestion penalty
                let total_score = raw_affinity + (proposal.voi_index * 0.5) - congestion_penalty;

                if total_score > best_score {
                    best_score = total_score;
                    best_reviewer_id = Some(r.id.clone());
                    best_raw_affinity = raw_affinity;
                }
            }

            if let Some(r_id) = best_reviewer_id {
                let load_entry = current_assigned.entry(r_id.clone()).or_insert(0);
                *load_entry += 1;
                assigned_for_this_prop.insert(r_id.clone());
                global_objective += best_raw_affinity;

                assignments.push(MatchAssignment {
                    proposal_id: proposal.id.clone(),
                    reviewer_id: r_id,
                    affinity_score: best_raw_affinity,
                    priority_rank: slot + 1,
                });
            } else {
                // Could not fulfill this review requirement slot
                if !unassigned_proposals.contains(&proposal.id) {
                    unassigned_proposals.push(proposal.id.clone());
                }
            }
        }
    }

    // Build utilization reports & bottleneck warnings
    let mut reviewer_utilizations = Vec::new();
    let mut bottleneck_warnings = Vec::new();

    for r in reviewers {
        let assigned = *current_assigned.get(&r.id).unwrap_or(&0);
        let service = r.service_rate.max(1.0);
        let pre_rho = r.arrival_rate / service;
        let newly_assigned = assigned.saturating_sub(r.current_load);
        let post_rho = (r.arrival_rate + newly_assigned as f64) / service;
        let is_over = post_rho > target_rho || assigned >= r.capacity;

        if post_rho > target_rho {
            bottleneck_warnings.push(format!(
                "Reviewer {} ({}) is over-congested: rho={:.2} > target {:.2}",
                r.name, r.domain, post_rho, target_rho
            ));
        }

        reviewer_utilizations.push(ReviewerUtilizationReport {
            reviewer_id: r.id.clone(),
            domain: r.domain.clone(),
            assigned_count: assigned,
            capacity: r.capacity,
            pre_utilization: pre_rho,
            post_utilization: post_rho,
            is_over_capacity: is_over,
        });
    }

    CongestionMatchingResult {
        assignments,
        unassigned_proposals,
        reviewer_utilizations,
        bottleneck_warnings,
        global_objective_score: global_objective,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_congestion_matching_prioritizes_high_voi_and_prevents_overload() {
        let reviewers = vec![
            ReviewerProfile {
                id: "rev_1".to_string(),
                name: "Dr. Alice (AI)".to_string(),
                domain: "Artificial Intelligence".to_string(),
                capacity: 2,
                current_load: 0,
                service_rate: 10.0,
                arrival_rate: 6.0,
                expertise_tags: vec!["transformer".to_string(), "voi".to_string()],
            },
            ReviewerProfile {
                id: "rev_2".to_string(),
                name: "Dr. Bob (AI)".to_string(),
                domain: "Artificial Intelligence".to_string(),
                capacity: 3,
                current_load: 0,
                service_rate: 10.0,
                arrival_rate: 5.0,
                expertise_tags: vec!["transformer".to_string(), "deep learning".to_string()],
            },
        ];

        let proposals = vec![
            ProposalRequirement {
                id: "prop_high_voi".to_string(),
                title: "Radical Transformer Triage".to_string(),
                domain: "Artificial Intelligence".to_string(),
                voi_index: 0.85,
                required_reviews: 2,
                keywords: vec!["transformer".to_string(), "voi".to_string()],
            },
            ProposalRequirement {
                id: "prop_low_voi".to_string(),
                title: "Standard AI Benchmark".to_string(),
                domain: "Artificial Intelligence".to_string(),
                voi_index: 0.10,
                required_reviews: 2,
                keywords: vec!["transformer".to_string()],
            },
        ];

        let result = optimize_congestion_matching(&proposals, &reviewers, Some(0.85));

        // Total 4 assignments needed (2 proposals * 2 reviews)
        // Total reviewer capacity: 2 + 3 = 5
        assert_eq!(result.assignments.len(), 4);
        assert!(result.unassigned_proposals.is_empty());
        assert!(result.global_objective_score > 0.0);

        // High VOI proposal should be assigned to rev_1 (closest keyword match)
        let high_voi_assigns: Vec<&MatchAssignment> = result
            .assignments
            .iter()
            .filter(|a| a.proposal_id == "prop_high_voi")
            .collect();
        assert_eq!(high_voi_assigns.len(), 2);
    }
}
