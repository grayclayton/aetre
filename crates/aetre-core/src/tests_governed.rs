#[cfg(test)]
mod tests {
    use crate::governor::Governor;
    use crate::knapsack::{CandidateSubmission, KnapsackController};

    #[test]
    fn test_governor_initialization() {
        let gov = Governor::new(0.02, 0.10, 0.50, 4, None, None, None).unwrap();
        assert_eq!(gov.reward, 0.02);
        assert_eq!(gov.loss, 0.10);
        assert_eq!(gov.prior, 0.50);
        assert_eq!(gov.max_stages, 4);
        assert!((gov.p_star - 0.10 / 0.12).abs() < 1e-12);
        assert_eq!(gov.cost_schedule.len(), 4);
    }

    #[test]
    fn test_governor_validation_errors() {
        assert!(Governor::new(-0.02, 0.10, 0.50, 4, None, None, None).is_err());
        assert!(Governor::new(0.02, 0.10, 0.0, 4, None, None, None).is_err());
        assert!(Governor::new(0.02, 0.10, 1.0, 4, None, None, None).is_err());
        assert!(Governor::new(0.02, 0.10, 0.50, 4, Some(vec![0.001]), None, None).is_err());
    }

    #[test]
    fn test_terminal_utility() {
        let gov = Governor::new(0.02, 0.10, 0.50, 4, None, None, None).unwrap();
        assert_eq!(gov.terminal_utility(0.0), 0.0);
        assert_eq!(gov.terminal_utility(gov.p_star), 0.0);
        assert!((gov.terminal_utility(1.0) - 0.02).abs() < 1e-12);
        assert!((gov.terminal_utility(0.90) - (0.90 * 0.02 - 0.10 * 0.10)).abs() < 1e-12);
    }

    #[test]
    fn test_governor_evaluation_actions() {
        // High loss stakes -> should continue verification at prior 0.50
        let costs = vec![0.0001, 0.0005, 0.0005, 0.0020];
        let gov = Governor::new(0.02, 0.10, 0.50, 4, Some(costs), Some(0.5875), Some(1.0)).unwrap();
        let dec0 = gov.evaluate_state(0, 0);
        assert_eq!(dec0.action, "CONTINUE");
        assert!(dec0.voi > 0.0);

        // Max stages with belief exceeding p* -> HALT_AND_COMMIT
        let dec_final_commit = gov.evaluate_state(4, 4);
        assert_eq!(dec_final_commit.action, "HALT_AND_COMMIT");
        assert!(dec_final_commit.belief > gov.p_star);

        // Zero passes at end -> HALT_AND_REJECT
        let dec_final_reject = gov.evaluate_state(4, 0);
        assert_eq!(dec_final_reject.action, "HALT_AND_REJECT");
    }

    #[test]
    fn test_evaluate_review_boundary_tripartite() {
        let gov = Governor::new(0.02, 0.10, 0.50, 4, None, None, None).unwrap();

        // 1. High belief (b = 0.98 > p* = 0.833):
        // Expected loss from defect: (1 - 0.98) * 0.10 = 0.002 < review_cost (0.005).
        // u_auto = 0.98 * 0.02 - 0.02 * 0.10 = 0.0196 - 0.002 = 0.0176
        // u_review (review_cost=0.005, lambda=0.0) = 0.98 * 0.02 - 0.005 = 0.0146
        // u_auto > u_review -> AUTO!
        let auto_dec = gov.evaluate_review_boundary(0.98, 0.005, 0.0, 1.0).unwrap();
        assert_eq!(auto_dec.action, "AUTO");
        assert!((auto_dec.u_auto - 0.0176).abs() < 1e-6);

        // 2. Intermediate belief (b = 0.70 < p* = 0.833):
        // u_auto = 0.0 (negative, clamped to 0 in terminal_utility)
        // If queue is slack (lambda = 0.0) and review_cost = 0.002:
        // u_review = 0.70 * 0.02 - 0.002 = 0.012 > 0 -> REVIEW!
        let review_dec = gov.evaluate_review_boundary(0.70, 0.002, 0.0, 1.0).unwrap();
        assert_eq!(review_dec.action, "REVIEW");
        assert!((review_dec.u_review - 0.012).abs() < 1e-6);

        // 3. Same intermediate belief, but queue is congested (lambda = 0.015 > 0.012):
        // u_review = 0.70 * 0.02 - 0.002 - 0.015 = -0.005 < 0
        // Review is no longer viable -> ABSTAIN!
        let abstain_dec = gov
            .evaluate_review_boundary(0.70, 0.002, 0.015, 1.0)
            .unwrap();
        assert_eq!(abstain_dec.action, "ABSTAIN");

        // 4. Low belief (b = 0.10):
        // u_auto = 0, u_review = 0.10 * 0.02 - 0.005 = -0.003 < 0 -> ABSTAIN!
        let low_dec = gov.evaluate_review_boundary(0.10, 0.005, 0.0, 1.0).unwrap();
        assert_eq!(low_dec.action, "ABSTAIN");
    }

    #[test]
    fn test_knapsack_controller_admission() {
        let controller = KnapsackController::new(2.0, 1.0).unwrap();
        let candidates = vec![
            CandidateSubmission::new("c1", "t1", 0.95, Some(0.02), Some(0.10), None),
            CandidateSubmission::new("c2", "t2", 0.90, Some(0.02), Some(0.10), None),
            CandidateSubmission::new("c3", "t3", 0.88, Some(0.02), Some(0.10), None),
            CandidateSubmission::new("c4", "t4", 0.20, Some(0.02), Some(0.10), None),
        ];

        let report = controller.admit_batch(&candidates, None, None).unwrap();
        assert_eq!(report.capacity_k, 2.0);
        assert_eq!(report.optimal_selection_depth, 2);
        assert_eq!(report.total_candidates, 4);
        assert_eq!(report.total_admitted, 2);
        assert_eq!(report.total_rejected, 2);
        assert_eq!(report.admitted[0].candidate_id, "c1");
        assert_eq!(report.admitted[1].candidate_id, "c2");

        // Capacity is binding and candidate c3 is acceptable -> shadow price = expected utility of c3
        assert!((report.shadow_price_lambda - candidates[2].expected_utility()).abs() < 1e-12);
    }

    #[test]
    fn test_knapsack_slack_capacity() {
        let controller = KnapsackController::new(10.0, 1.0).unwrap();
        let candidates = vec![
            CandidateSubmission::new("c1", "t1", 0.95, Some(0.02), Some(0.10), None),
            CandidateSubmission::new("c2", "t2", 0.90, Some(0.02), Some(0.10), None),
        ];
        let report = controller.admit_batch(&candidates, None, None).unwrap();
        assert_eq!(report.total_admitted, 2);
        assert_eq!(report.shadow_price_lambda, 0.0);
    }

    #[test]
    fn test_knapsack_override_validation_and_unit_shadow_price() {
        let controller = KnapsackController::new(2.0, 2.0).unwrap();
        assert!(controller.admit_batch(&[], Some(0.0), None).is_err());
        assert!(controller.admit_batch(&[], None, Some(0.0)).is_err());

        let candidates = vec![
            CandidateSubmission::new("c1", "t1", 0.99, None, None, None),
            CandidateSubmission::new("c2", "t2", 0.95, None, None, None),
        ];
        let report = controller.admit_batch(&candidates, None, None).unwrap();
        assert_eq!(report.shadow_price_lambda, 0.0);
    }

    #[test]
    fn test_knapsack_heterogeneous_c_mu_admission() {
        // Reviewer budget: 4.0 hours
        let controller = KnapsackController::new(4.0, 1.0).unwrap();

        // Candidate A: 10-minute quick fix (0.167 hr), high posterior
        // Candidate B: 3.0-hour deep refactor, higher raw utility than A, but much lower density
        // Candidate C: 1.0-hour feature, high density
        // Candidate D: 2.0-hour refactor, moderate density
        let c_a = CandidateSubmission::new(
            "cand_quick_fix",
            "task_a",
            0.95,
            Some(0.02),
            Some(0.10),
            Some(0.167),
        );
        let c_b = CandidateSubmission::new(
            "cand_deep_refactor",
            "task_b",
            0.99,
            Some(0.05),
            Some(0.10),
            Some(3.0),
        );
        let c_c = CandidateSubmission::new(
            "cand_feature",
            "task_c",
            0.92,
            Some(0.03),
            Some(0.10),
            Some(1.0),
        );
        let c_d = CandidateSubmission::new(
            "cand_doc_fix",
            "task_d",
            0.88,
            Some(0.02),
            Some(0.10),
            Some(0.5),
        );

        let candidates = vec![c_b.clone(), c_a.clone(), c_d.clone(), c_c.clone()];
        let report = controller.admit_batch(&candidates, None, None).unwrap();

        // Quick fix (density ~0.088) must be admitted before deep refactor (density ~0.016)
        assert!(report
            .admitted
            .iter()
            .any(|c| c.candidate_id == "cand_quick_fix"));
        assert!(report
            .admitted
            .iter()
            .any(|c| c.candidate_id == "cand_feature"));
        assert!(report
            .admitted
            .iter()
            .any(|c| c.candidate_id == "cand_doc_fix"));

        // Admitted total cost must be within budget: 0.167 + 1.0 + 0.5 = 1.667 <= 4.0
        assert!(report.total_admitted_cost <= 4.0);
        assert!((report.total_admitted_cost - (0.167 + 1.0 + 0.5 + 3.0)).abs() > 1e-6);

        // Remaining capacity should be accurately tracked
        assert_eq!(report.remaining_capacity, 4.0 - report.total_admitted_cost);
    }

    #[test]
    fn test_native_performance_throughput() {
        use std::time::Instant;

        // 1. Native Governor Throughput (100,000 evaluations)
        let gov = Governor::new(0.02, 0.10, 0.50, 6, None, None, None).unwrap();
        let start = Instant::now();
        let iterations = 100_000;
        let mut total_voi = 0.0;
        for i in 0..iterations {
            let stage = i % 7;
            let passes = (i / 7) % (stage + 1);
            let dec = gov.evaluate_state(stage, passes);
            total_voi += dec.voi;
        }
        let elapsed = start.elapsed();
        println!(
            "\n[NATIVE RUST] Governor 100,000 evaluations in {:?} ({:.0} ops/sec, checksum={:.4})",
            elapsed,
            iterations as f64 / elapsed.as_secs_f64(),
            total_voi
        );

        // 2. Native Knapsack Throughput (10,000 batches of 50 candidates = 500,000 candidates)
        let controller = KnapsackController::new(20.0, 2.0).unwrap();
        let cands: Vec<CandidateSubmission> = (0..50)
            .map(|i| {
                CandidateSubmission::new(
                    format!("c{}", i),
                    "t1",
                    0.50 + (i as f64) * 0.009,
                    Some(0.02),
                    Some(0.10),
                    None,
                )
            })
            .collect();
        let start = Instant::now();
        let batches = 10_000;
        let mut total_welfare = 0.0;
        for _ in 0..batches {
            let rep = controller.admit_batch(&cands, None, None).unwrap();
            total_welfare += rep.total_welfare;
        }
        let elapsed = start.elapsed();
        println!(
            "[NATIVE RUST] Knapsack 500,000 candidates in {:?} ({:.0} cands/sec, checksum={:.4})",
            elapsed,
            (batches * 50) as f64 / elapsed.as_secs_f64(),
            total_welfare
        );
    }
}
