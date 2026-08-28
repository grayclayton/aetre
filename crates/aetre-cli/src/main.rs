use aetre_core::{
    calculate_boundary_voi, calculate_exploration_audit, calculate_governor_action,
    calculate_proposition_1_bound, evaluate_multi_attribute_voi, evaluate_sequential_stopping,
    evaluate_stage_queue, evaluate_submitter_equilibrium, evaluate_venture_benchmark,
    generate_recall_scaling_curve, generate_staking_curve, generate_synthetic_venture_dealflow,
    optimize_congestion_matching, run_benchmark_replications, MultiAttributeDimension, Parameters,
    ProposalRequirement, ReviewerProfile, SequentialReviewStep,
};
use rand::thread_rng;
use serde::{Deserialize, Serialize};
use std::env;
use std::fs;
use std::io::{self, Write};
use std::path::PathBuf;

mod backtest;
mod shadow_pilot;
mod validation;

fn main() {
    let args: Vec<String> = env::args().collect();
    let command = args.get(1).map(|s| s.as_str()).unwrap_or("help");

    match command {
        "benchmark" => run_benchmark_cmd(&args[2..]),
        "bound" => run_bound_cmd(&args[2..]),
        "queue" => run_queue_cmd(&args[2..]),
        "audit" => run_audit_cmd(&args[2..]),
        "staking" => run_staking_cmd(&args[2..]),
        "multi-attribute" | "multi-dim" => run_multi_attribute_cmd(&args[2..]),
        "match" | "matching" => run_match_cmd(&args[2..]),
        "sequential" => run_sequential_cmd(&args[2..]),
        "vc-benchmark" | "investment" => run_vc_benchmark_cmd(&args[2..]),
        "test" => run_test_dataset_cmd(&args[2..]),
        "backtest" => match backtest::run_heldout_backtest(&args[2..]) {
            Ok(report) => println!("{report}"),
            Err(error) => {
                eprintln!("Backtest error: {error}");
                std::process::exit(2);
            }
        },
        "shadow-pilot" | "pilot" => match shadow_pilot::run_shadow_pilot(&args[2..]) {
            Ok(report) => println!("{report}"),
            Err(error) => {
                eprintln!("Shadow Pilot error: {error}");
                std::process::exit(2);
            }
        },
        "validate-predictions" => match validation::run(&args[2..]) {
            Ok(report) => println!("{report}"),
            Err(error) => {
                eprintln!("Validation error: {error}");
                std::process::exit(2);
            }
        },
        "login" => run_login_cmd(&args[2..]),
        "license" | "status" => run_license_cmd(&args[2..]),
        "help" | "--help" | "-h" => print_help(),
        other => {
            eprintln!("Unknown command: '{}'. Run 'aetre help' for usage.", other);
            std::process::exit(1);
        }
    }
}

fn print_help() {
    println!(
        r#"
╔═══════════════════════════════════════════════════════════════════════════╗
║   AETRE: Adaptive Epistemic Triage & Recall Engine (CLI v0.1.0)           ║
║   Operationalizing "The Innovation-Absorption Gap" (Gray, 2026)           ║
╚═══════════════════════════════════════════════════════════════════════════╝

USAGE:
    aetre <COMMAND> [OPTIONS]

COMMANDS:
    benchmark    Run the 4-regime Monte Carlo simulation (replicates Table 1)
                 Options: --replications <N> (default: 500) [--json | --csv]
    
    bound        Evaluate Proposition 1 Throughput-Recall Bound (R_N <= min(1, K/H))
                 Options: --arrivals <N> --capacity <K> --high-rate <P> [--json | --csv]
    
    queue        Evaluate Kingman Heavy-Traffic queueing and Governor actions
                 Options: --arrival-rate <L> --service-rate <M> --cv-a <CA> --cv-s <CS> [--json | --csv]
    
    audit        Compute randomized exploration audit estimator (H_hat_D)
                 Options: --pool <N> --sample <n> --found <k> [--json | --csv]
    
    staking      Analyze submitter congestion and anti-spam staking elasticity
                 Options: --c-gen <G> --c-sub <S> --val <V> --applicants <N> --capacity <K> [--json | --csv]
    
    multi-attribute
                 Run multi-attribute Bayesian VOI epistemic triage
                 Options: [--file <dims.json>] [--threshold <T>] [--cost <C>] [--json]

    match        Run congestion-aware reviewer assignment optimizer
                 Options: [--file <match_data.json>] [--target-utilization <0.85>] [--json]

    sequential   Evaluate dynamic Bayesian sequential review stopping rule
                 Options: [--scores <6.2,5.9>] [--prior-mean <M>] [--threshold <T>] [--confidence <0.90>] [--json]

    vc-benchmark Run venture capital & equity dealflow triage benchmark
                 Options: [--deals <N>] [--budget <K>] [--alpha <A>] [--wrapper-pct <W>] [--json]

    test         Evaluate triage routing on domain benchmark datasets or JSON files
                 Options: --dataset <openreview|nih|uspto|paperswithcode|proposals>
                          --file <path/to/custom_data.json> [--boundary <B>] [--json | --csv]

    backtest     Run held-out review allocation backtest comparing 8 triage policies
                 Options: --file <openreview_heldout.json> --budget <K> [--boundary <B>] [--split <test|all>] [--json]

    shadow-pilot Run Level 4 prospective shadow-mode pilot, prediction freeze, or randomized 3-arm trial
                 Options: --mode <freeze|randomize|reconcile|simulate> --budget <K> [--audit-rate <0.05>] [--output <path>] [--json]

    validate-predictions
                 Evaluate frozen test-split predictions against labels and a baseline
                 Options: --file <predictions.json> --budget <K> [--threshold <P>] [--output <path>]
    
    help         Show this help message
"#
    );
}

fn run_multi_attribute_cmd(args: &[String]) {
    let mut file_path: Option<String> = None;
    let mut threshold = 6.0;
    let mut cost_per_dim = 1.0;
    let mut json_out = false;

    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--file" | "-f" => {
                if let Some(val) = args.get(i + 1) {
                    file_path = Some(val.clone());
                    i += 1;
                }
            }
            "--threshold" | "-t" => {
                if let Some(val) = args.get(i + 1) {
                    threshold = val.parse().unwrap_or(6.0);
                    i += 1;
                }
            }
            "--cost" | "-c" => {
                if let Some(val) = args.get(i + 1) {
                    cost_per_dim = val.parse().unwrap_or(1.0);
                    i += 1;
                }
            }
            "--json" => json_out = true,
            _ => {}
        }
        i += 1;
    }

    let dimensions: Vec<MultiAttributeDimension> = if let Some(ref path) = file_path {
        match fs::read_to_string(path) {
            Ok(content) => match serde_json::from_str::<Vec<MultiAttributeDimension>>(&content) {
                Ok(dims) => dims,
                Err(_) => {
                    #[derive(Deserialize)]
                    struct Wrapper {
                        dimensions: Vec<MultiAttributeDimension>,
                    }
                    serde_json::from_str::<Wrapper>(&content)
                        .map(|w| w.dimensions)
                        .unwrap_or_else(|e| {
                            eprintln!("Error parsing JSON from {}: {}", path, e);
                            std::process::exit(1);
                        })
                }
            },
            Err(e) => {
                eprintln!("Error reading file {}: {}", path, e);
                std::process::exit(1);
            }
        }
    } else {
        vec![
            MultiAttributeDimension {
                name: "Novelty & Originality".to_string(),
                prior_mean: 6.5,
                prior_variance: 1.2,
                weight: 0.35,
                threshold: Some(6.0),
                review_noise_sd: 0.75,
            },
            MultiAttributeDimension {
                name: "Empirical & Theoretical Rigor".to_string(),
                prior_mean: 5.4,
                prior_variance: 0.8,
                weight: 0.35,
                threshold: Some(6.0),
                review_noise_sd: 0.60,
            },
            MultiAttributeDimension {
                name: "Feasibility & Methodology".to_string(),
                prior_mean: 5.8,
                prior_variance: 0.4,
                weight: 0.15,
                threshold: Some(6.0),
                review_noise_sd: 0.80,
            },
            MultiAttributeDimension {
                name: "Broader Impact".to_string(),
                prior_mean: 6.0,
                prior_variance: 0.3,
                weight: 0.15,
                threshold: Some(6.0),
                review_noise_sd: 0.90,
            },
        ]
    };

    let result = evaluate_multi_attribute_voi(&dimensions, threshold, cost_per_dim);

    if json_out {
        println!("{}", serde_json::to_string_pretty(&result).unwrap());
    } else {
        println!("\n╔═══════════════════════════════════════════════════════════════════════════╗");
        println!("║   AETRE: Multi-Attribute Bayesian Epistemic Triage                        ║");
        println!("╚═══════════════════════════════════════════════════════════════════════════╝\n");
        println!(
            "Composite Prior Mean:     {:.3}",
            result.composite_prior_mean
        );
        println!(
            "Composite Epistemic Var:  {:.3} (SD = {:.3})",
            result.composite_prior_variance,
            result.composite_prior_variance.sqrt()
        );
        println!(
            "Acceptance Threshold:     {:.2}",
            result.composite_threshold
        );
        println!("Total Composite VOI:      {:.4}", result.composite_voi);
        println!("Suggested Stream Routing: {}", result.suggested_routing);
        println!("\nDIMENSION BREAKDOWN:");
        println!("┌────────────────────────────────────┬────────┬─────────────┬──────────────┐");
        println!("│ Dimension                          │ Weight │ Marg. VOI   │ Var Share    │");
        println!("├────────────────────────────────────┼────────┼─────────────┼──────────────┤");
        for c in &result.dimension_contributions {
            println!(
                "│ {:<34} │ {:>6.2} │ {:>11.4} │ {:>11.1}% │",
                c.dimension,
                c.weight,
                c.marginal_voi,
                c.variance_share * 100.0
            );
        }
        println!("└────────────────────────────────────┴────────┴─────────────┴──────────────┘");
        if !result.recommended_review_dimensions.is_empty() {
            println!(
                "\nTargeted Review Advice: Focus scarce review bandwidth on [{}]\n",
                result.recommended_review_dimensions.join(", ")
            );
        }
    }
}

fn run_match_cmd(args: &[String]) {
    let mut file_path: Option<String> = None;
    let mut target_rho = 0.85;
    let mut json_out = false;

    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--file" | "-f" => {
                if let Some(val) = args.get(i + 1) {
                    file_path = Some(val.clone());
                    i += 1;
                }
            }
            "--target-utilization" | "--target-rho" | "-u" => {
                if let Some(val) = args.get(i + 1) {
                    target_rho = val.parse().unwrap_or(0.85);
                    i += 1;
                }
            }
            "--json" => json_out = true,
            _ => {}
        }
        i += 1;
    }

    #[derive(Deserialize)]
    struct MatchInput {
        proposals: Vec<ProposalRequirement>,
        reviewers: Vec<ReviewerProfile>,
    }

    let (proposals, reviewers) = if let Some(ref path) = file_path {
        match fs::read_to_string(path) {
            Ok(content) => match serde_json::from_str::<MatchInput>(&content) {
                Ok(input) => (input.proposals, input.reviewers),
                Err(e) => {
                    eprintln!("Error parsing JSON from {}: {}", path, e);
                    std::process::exit(1);
                }
            },
            Err(e) => {
                eprintln!("Error reading file {}: {}", path, e);
                std::process::exit(1);
            }
        }
    } else {
        let sample_props = vec![
            ProposalRequirement {
                id: "prop_1".to_string(),
                title: "Quantum Variational Eigensolvers".to_string(),
                domain: "Quantum Computing".to_string(),
                voi_index: 0.85,
                required_reviews: 2,
                keywords: vec!["quantum".to_string(), "vqe".to_string()],
            },
            ProposalRequirement {
                id: "prop_2".to_string(),
                title: "Transformer Latency Optimization".to_string(),
                domain: "Artificial Intelligence".to_string(),
                voi_index: 0.45,
                required_reviews: 2,
                keywords: vec!["transformer".to_string(), "inference".to_string()],
            },
            ProposalRequirement {
                id: "prop_3".to_string(),
                title: "CRISPR Epigenetic Silencing".to_string(),
                domain: "Biotechnology".to_string(),
                voi_index: 0.70,
                required_reviews: 2,
                keywords: vec!["crispr".to_string(), "mrna".to_string()],
            },
        ];
        let sample_revs = vec![
            ReviewerProfile {
                id: "rev_1".to_string(),
                name: "Dr. Alice (Quantum)".to_string(),
                domain: "Quantum Computing".to_string(),
                capacity: 2,
                current_load: 0,
                service_rate: 10.0,
                arrival_rate: 6.0,
                expertise_tags: vec!["quantum".to_string(), "vqe".to_string()],
            },
            ReviewerProfile {
                id: "rev_2".to_string(),
                name: "Dr. Bob (AI)".to_string(),
                domain: "Artificial Intelligence".to_string(),
                capacity: 3,
                current_load: 0,
                service_rate: 10.0,
                arrival_rate: 5.0,
                expertise_tags: vec!["transformer".to_string(), "attention".to_string()],
            },
            ReviewerProfile {
                id: "rev_3".to_string(),
                name: "Dr. Carol (Bio)".to_string(),
                domain: "Biotechnology".to_string(),
                capacity: 2,
                current_load: 0,
                service_rate: 8.0,
                arrival_rate: 4.0,
                expertise_tags: vec!["crispr".to_string(), "genetics".to_string()],
            },
        ];
        (sample_props, sample_revs)
    };

    let result = optimize_congestion_matching(&proposals, &reviewers, Some(target_rho));

    if json_out {
        println!("{}", serde_json::to_string_pretty(&result).unwrap());
    } else {
        println!("\n╔═══════════════════════════════════════════════════════════════════════════╗");
        println!(
            "║   AETRE: Congestion-Aware Reviewer Matching (Kingman rho <= {:.2})         ║",
            target_rho
        );
        println!("╚═══════════════════════════════════════════════════════════════════════════╝\n");
        println!("Proposals Evaluated:       {}", proposals.len());
        println!("Reviewers in Pool:         {}", reviewers.len());
        println!("Assignments Fulfilled:     {}", result.assignments.len());
        println!(
            "Global Affinity Score:     {:.2}",
            result.global_objective_score
        );
        if !result.unassigned_proposals.is_empty() {
            println!(
                "⚠️ Unassigned Slots:        {}",
                result.unassigned_proposals.join(", ")
            );
        }
        println!("\nASSIGNMENTS:");
        println!("┌────────────────────────┬────────────────────────┬──────────┬────────┐");
        println!("│ Proposal ID            │ Reviewer ID            │ Affinity │ Slot # │");
        println!("├────────────────────────┼────────────────────────┼──────────┼────────┤");
        for a in &result.assignments {
            println!(
                "│ {:<22} │ {:<22} │ {:>8.2} │ {:>6} │",
                a.proposal_id, a.reviewer_id, a.affinity_score, a.priority_rank
            );
        }
        println!("└────────────────────────┴────────────────────────┴──────────┴────────┘");

        println!("\nREVIEWER UTILIZATION POST-MATCHING:");
        for u in &result.reviewer_utilizations {
            let status = if u.is_over_capacity {
                "⚠️ OVERLOADED"
            } else {
                "✅ STABLE"
            };
            println!(
                "- {} ({}) : Assigned {}/{} | Pre-rho: {:.2} -> Post-rho: {:.2} [{}]",
                u.reviewer_id,
                u.domain,
                u.assigned_count,
                u.capacity,
                u.pre_utilization,
                u.post_utilization,
                status
            );
        }
        for w in &result.bottleneck_warnings {
            println!("  [WARN] {}", w);
        }
        println!();
    }
}

fn run_sequential_cmd(args: &[String]) {
    let mut prior_mean = 5.0;
    let mut prior_var = 1.0;
    let mut threshold = 6.0;
    let mut confidence = 0.90;
    let mut raw_scores: Vec<f64> = Vec::new();
    let mut json_out = false;

    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--prior-mean" | "-m" => {
                if let Some(val) = args.get(i + 1) {
                    prior_mean = val.parse().unwrap_or(5.0);
                    i += 1;
                }
            }
            "--prior-var" | "-v" => {
                if let Some(val) = args.get(i + 1) {
                    prior_var = val.parse().unwrap_or(1.0);
                    i += 1;
                }
            }
            "--threshold" | "-t" => {
                if let Some(val) = args.get(i + 1) {
                    threshold = val.parse().unwrap_or(6.0);
                    i += 1;
                }
            }
            "--confidence" | "-c" => {
                if let Some(val) = args.get(i + 1) {
                    confidence = val.parse().unwrap_or(0.90);
                    i += 1;
                }
            }
            "--scores" | "-s" => {
                if let Some(val) = args.get(i + 1) {
                    raw_scores = val
                        .split(',')
                        .filter_map(|s| s.trim().parse::<f64>().ok())
                        .collect();
                    i += 1;
                }
            }
            "--json" => json_out = true,
            _ => {}
        }
        i += 1;
    }

    if raw_scores.is_empty() {
        raw_scores = vec![6.2, 5.9];
    }

    let reviews: Vec<SequentialReviewStep> = raw_scores
        .iter()
        .enumerate()
        .map(|(idx, &score)| SequentialReviewStep {
            step: idx + 1,
            reviewer_id: format!("reviewer_{}", idx + 1),
            score,
            noise_sd: 0.8,
            cost: 1.0,
        })
        .collect();

    let result = evaluate_sequential_stopping(
        prior_mean,
        prior_var,
        threshold,
        &reviews,
        Some(0.80),
        Some(1.0),
        Some(confidence),
    );

    if json_out {
        println!("{}", serde_json::to_string_pretty(&result).unwrap());
    } else {
        println!("\n╔═══════════════════════════════════════════════════════════════════════════╗");
        println!("║   AETRE: Dynamic Sequential Bayesian Stopping Rule                        ║");
        println!("╚═══════════════════════════════════════════════════════════════════════════╝\n");
        println!("Reviews Evaluated:        {}", result.current_step);
        println!("Scores Ingested:          {:?}", raw_scores);
        println!("Posterior Mean:           {:.3}", result.posterior_mean);
        println!(
            "Posterior Variance:       {:.3} (SD = {:.3})",
            result.posterior_variance,
            result.posterior_variance.sqrt()
        );
        println!("Decision Cutoff:          {:.2}", threshold);
        println!(
            "Decision Confidence:      {:.1}%",
            result.decision_confidence * 100.0
        );
        println!("Prospective VOI for Next: {:.4}", result.current_voi);
        println!("\nRECOMMENDED ACTION:       {:?}", result.decision);
        println!("RATIONALE:                {}\n", result.stopping_rationale);
    }
}

fn run_vc_benchmark_cmd(args: &[String]) {
    let mut n_deals = 1000;
    let mut diligence_budget = 50;
    let mut tail_alpha = 1.25;
    let mut wrapper_pct = 0.30;
    let mut selection_boundary = 6.0;
    let mut hours_per_deal = 20.0;
    let mut json_out = false;

    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--deals" | "-n" => {
                if let Some(val) = args.get(i + 1) {
                    n_deals = val.parse().unwrap_or(1000);
                    i += 1;
                }
            }
            "--budget" | "-k" => {
                if let Some(val) = args.get(i + 1) {
                    diligence_budget = val.parse().unwrap_or(50);
                    i += 1;
                }
            }
            "--alpha" | "-a" => {
                if let Some(val) = args.get(i + 1) {
                    tail_alpha = val.parse().unwrap_or(1.25);
                    i += 1;
                }
            }
            "--wrapper-pct" | "-w" => {
                if let Some(val) = args.get(i + 1) {
                    wrapper_pct = val.parse().unwrap_or(0.30);
                    i += 1;
                }
            }
            "--boundary" | "-b" => {
                if let Some(val) = args.get(i + 1) {
                    selection_boundary = val.parse().unwrap_or(6.0);
                    i += 1;
                }
            }
            "--hours" | "-h" => {
                if let Some(val) = args.get(i + 1) {
                    hours_per_deal = val.parse().unwrap_or(20.0);
                    i += 1;
                }
            }
            "--json" => {
                json_out = true;
            }
            _ => {}
        }
        i += 1;
    }

    let deals =
        generate_synthetic_venture_dealflow(n_deals, tail_alpha, wrapper_pct, selection_boundary);
    let comparison =
        evaluate_venture_benchmark(&deals, diligence_budget, tail_alpha, hours_per_deal);

    if json_out {
        println!("{}", serde_json::to_string_pretty(&comparison).unwrap());
    } else {
        println!("\n╔═══════════════════════════════════════════════════════════════════════════╗");
        println!("║   AETRE: Venture Capital & Equity Dealflow Triage Benchmark               ║");
        println!("╚═══════════════════════════════════════════════════════════════════════════╝\n");
        println!(
            "Dealflow Universe Evaluated:     {} deals",
            comparison.total_dealflow_universe
        );
        println!(
            "Partner Deep Diligence Budget K: {} deals (top {:.1}%)",
            comparison.diligence_budget,
            (comparison.diligence_budget as f64 / comparison.total_dealflow_universe as f64)
                * 100.0
        );
        println!(
            "Power-Law Pareto Tail Index α:   {:.2} (Heavy-Tailed Asymmetry)",
            comparison.tail_index_alpha
        );
        println!(
            "Total True Outliers in Cohort:   {} unicorns / fund returners",
            comparison.aetre_heavy_tailed.total_unicorns
        );

        println!("\nSTRATEGY COMPARISON:");
        println!(
            "┌────────────────────────────────────────┬──────────────────┬──────────────────┐"
        );
        println!(
            "│ Metric                                 │ Status Quo (Raw) │ AETRE (Tail VOI) │"
        );
        println!(
            "├────────────────────────────────────────┼──────────────────┼──────────────────┤"
        );
        println!(
            "│ Portfolio MOIC (Return Multiple)       │ {:>14.2}x │ {:>14.2}x │",
            comparison.status_quo.portfolio_moic, comparison.aetre_heavy_tailed.portfolio_moic
        );
        println!(
            "│ Estimated 5-Year Fund IRR              │ {:>15.1}% │ {:>15.1}% │",
            comparison.status_quo.portfolio_irr_approx * 100.0,
            comparison.aetre_heavy_tailed.portfolio_irr_approx * 100.0
        );
        println!(
            "│ Outliers / Fund Returners Captured     │ {:>16} │ {:>16} │",
            format!(
                "{}/{}",
                comparison.status_quo.unicorns_captured, comparison.status_quo.total_unicorns
            ),
            format!(
                "{}/{}",
                comparison.aetre_heavy_tailed.unicorns_captured,
                comparison.aetre_heavy_tailed.total_unicorns
            )
        );
        println!(
            "│ Outlier Discovery Recall Rate          │ {:>15.1}% │ {:>15.1}% │",
            comparison.status_quo.outlier_recall * 100.0,
            comparison.aetre_heavy_tailed.outlier_recall * 100.0
        );
        println!(
            "│ Partner Diligence Hours per Outlier    │ {:>13.1}h │ {:>13.1}h │",
            comparison.status_quo.diligence_hours_per_unicorn,
            comparison.aetre_heavy_tailed.diligence_hours_per_unicorn
        );
        println!(
            "│ AI Wrappers Inadvertently Audited     │ {:>16} │ {:>16} │",
            comparison.status_quo.wrappers_invested_count,
            comparison.aetre_heavy_tailed.wrappers_invested_count
        );
        println!(
            "│ AI Wrappers Successfully Filtered      │ {:>16} │ {:>16} │",
            comparison.status_quo.wrappers_avoided_count,
            comparison.aetre_heavy_tailed.wrappers_avoided_count
        );
        println!(
            "└────────────────────────────────────────┴──────────────────┴──────────────────┘"
        );

        println!("\nINSTITUTIONAL PERFORMANCE EDGE:");
        println!(
            "• Fund Multiple Gain:              {:.2}x higher MOIC vs. Status Quo",
            comparison.moic_multiplier_gain
        );
        println!(
            "• IRR Percentage Point Uplift:     +{:.1}% pt higher annualized IRR",
            comparison.irr_percentage_point_uplift
        );
        println!(
            "• Partner Diligence Hours Saved:   {:.1}% reduction in wasted partner research time",
            comparison.partner_hours_saved_pct
        );
        println!("• Outlier Capture Ratio:           {:.2}x more fund-returning investments identified\n", comparison.outlier_capture_improvement_ratio);
    }
}

// =========================================================================
// DATASET SCHEMAS & DYNAMIC EVALUATION
// =========================================================================

#[derive(Debug, Deserialize, Serialize)]
struct OpenReviewRecord {
    id: String,
    domain: String,
    title: String,
    authors: Vec<String>,
    review_scores: Vec<f64>,
    reviewer_confidence: Option<Vec<f64>>,
    mean_score: f64,
    score_variance: f64,
    is_boundary_case: bool,
    historical_decision: String,
    unconventional_novelty: f64,
    target_test_focus: String,
    abstract_text: Option<String>,
}

#[derive(Debug, Deserialize, Serialize)]
struct NihGrantRecord {
    id: String,
    domain: String,
    title: String,
    principal_investigator: String,
    requested_budget_usd: f64,
    initial_priority_percentile: f64,
    unconventional_risk_score: f64,
    epistemic_variance: f64,
    historical_funding_outcome: String,
    target_test_focus: String,
    #[serde(rename = "abstract")]
    abstract_text: Option<String>,
}

#[derive(Debug, Deserialize, Serialize)]
struct UsptoPatentRecord {
    id: String,
    domain: String,
    title: String,
    assignee: String,
    cpc_class: String,
    claims_count: usize,
    historical_pendency_months: f64,
    examiner_utilization_rho: f64,
    office_action_count: usize,
    target_test_focus: String,
    #[serde(rename = "abstract")]
    abstract_text: Option<String>,
}

#[derive(Debug, Deserialize, Serialize)]
struct GenericProposalRecord {
    id: String,
    domain: Option<String>,
    title: String,
    authors: Option<Vec<String>>,
    latent_quality_estimate: Option<f64>,
    epistemic_variance: Option<f64>,
    unconventional_novelty_score: Option<f64>,
    recommended_triage_stream: Option<String>,
    #[serde(rename = "abstract")]
    abstract_text: Option<String>,
}

#[derive(Debug, Deserialize, Serialize)]
struct PapersWithCodeRecord {
    id: String,
    domain: String,
    title: String,
    repository_url: String,
    code_language: String,
    has_runnable_benchmark: bool,
    claimed_throughput_speedup: String,
    sandboxed_execution_status: String,
    target_test_focus: String,
    #[serde(rename = "abstract")]
    abstract_text: Option<String>,
}

#[derive(Debug, Serialize)]
struct EvaluatedTriageItem {
    id: String,
    domain: String,
    title: String,
    metric_summary: String,
    voi_index: f64,
    triage_routing: String,
    rationale: String,
}

fn resolve_dataset_path(dataset_name: &str, custom_file: Option<&str>) -> Option<PathBuf> {
    if let Some(file_path) = custom_file {
        let p = PathBuf::from(file_path);
        if p.exists() {
            return Some(p);
        }
    }

    let candidates = match dataset_name.to_lowercase().as_str() {
        "openreview" | "peerread" => vec![
            "examples/datasets/openreview_peer_review.json",
            "../../examples/datasets/openreview_peer_review.json",
        ],
        "nih" | "grants" => vec![
            "examples/datasets/nih_grant_proposals.json",
            "../../examples/datasets/nih_grant_proposals.json",
        ],
        "uspto" | "patents" => vec![
            "examples/datasets/uspto_patent_applications.json",
            "../../examples/datasets/uspto_patent_applications.json",
        ],
        "paperswithcode" | "pwc" => vec![
            "examples/datasets/papers_with_code.json",
            "../../examples/datasets/papers_with_code.json",
        ],
        "proposals" => vec!["examples/proposals.json", "../../examples/proposals.json"],
        _ => vec![],
    };

    for path_str in candidates {
        let p = PathBuf::from(path_str);
        if p.exists() {
            return Some(p);
        }
    }

    None
}

fn run_test_dataset_cmd(args: &[String]) {
    let mut dataset_name = "openreview".to_string();
    let mut custom_file: Option<String> = None;
    let mut boundary = 6.5; // default boundary for score scale
    let mut json_out = false;
    let mut csv_out = false;

    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--dataset" | "-d" => {
                if let Some(val) = args.get(i + 1) {
                    dataset_name = val.clone();
                    i += 1;
                }
            }
            "--file" | "-f" => {
                if let Some(val) = args.get(i + 1) {
                    custom_file = Some(val.clone());
                    i += 1;
                }
            }
            "--boundary" | "-b" => {
                if let Some(val) = args.get(i + 1) {
                    boundary = val.parse().unwrap_or(6.5);
                    i += 1;
                }
            }
            "--json" => json_out = true,
            "--csv" => csv_out = true,
            _ => {}
        }
        i += 1;
    }

    let file_path = resolve_dataset_path(&dataset_name, custom_file.as_deref());
    let file_path = match file_path {
        Some(p) => p,
        None => {
            eprintln!(
                "Error: Could not locate dataset file for '{}' (or specified --file).",
                dataset_name
            );
            eprintln!("Available datasets: openreview, nih, uspto, paperswithcode, proposals");
            std::process::exit(1);
        }
    };

    let content = match fs::read_to_string(&file_path) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("Error reading dataset file {:?}: {}", file_path, e);
            std::process::exit(1);
        }
    };

    let mut evaluated_items: Vec<EvaluatedTriageItem> = Vec::new();

    // Cascading schema parser: auto-detects OpenReview, NIH, USPTO, PapersWithCode, or Generic Proposals
    if let Ok(records) = serde_json::from_str::<Vec<OpenReviewRecord>>(&content) {
        for r in records {
            let (mean, var) = if !r.review_scores.is_empty() {
                let m = r.review_scores.iter().sum::<f64>() / r.review_scores.len() as f64;
                let v = r
                    .review_scores
                    .iter()
                    .map(|&s| (s - m).powi(2))
                    .sum::<f64>()
                    / (r.review_scores.len() as f64 - 1.0).max(1.0);
                (m, v)
            } else {
                (r.mean_score, r.score_variance)
            };

            let voi = calculate_boundary_voi(mean, var.max(0.01), boundary, 0.80, 0.50);
            let (routing, rationale) = if voi > 0.15 && var > 1.5 {
                (
                    "🔍 DEEP VOI REVIEW QUEUE",
                    "High reviewer disagreement near boundary. Deep review has high probability of flipping outcome.",
                )
            } else if mean >= boundary + 1.0 && var < 1.0 {
                (
                    "🌟 FAST-PASS SPOTLIGHT",
                    "Consensus top-tier candidate with low epistemic variance. Direct promotion saves capacity.",
                )
            } else if mean < boundary - 1.0 && var < 1.0 {
                (
                    "🚫 FAST-REJECT FILTER",
                    "Consensus below boundary with low uncertainty. Reject to conserve human attention.",
                )
            } else {
                (
                    "🎲 EXPLORATION AUDIT POOL",
                    "Unconventional candidate sampled into 5% Horvitz-Thompson exploration audit.",
                )
            };

            evaluated_items.push(EvaluatedTriageItem {
                id: r.id,
                domain: r.domain,
                title: r.title,
                metric_summary: format!("Mean {:.2} (Var {:.2})", mean, var),
                voi_index: voi,
                triage_routing: routing.to_string(),
                rationale: rationale.to_string(),
            });
        }
    } else if let Ok(records) = serde_json::from_str::<Vec<NihGrantRecord>>(&content) {
        for r in records {
            let mean_equiv = 30.0 - r.initial_priority_percentile; // Higher is better
            let voi = calculate_boundary_voi(mean_equiv, r.epistemic_variance, 14.0, 0.80, 0.50);
            let (routing, rationale) = if r.initial_priority_percentile <= 10.0 {
                (
                    "🌟 FAST-PASS FUNDED",
                    "Top decile priority score with established methodology.",
                )
            } else if r.initial_priority_percentile <= 16.0 || voi > 0.20 {
                (
                    "🔍 DEEP VOI REVIEW QUEUE",
                    "High-variance boundary case. Area-expert panel review flips decision.",
                )
            } else {
                (
                    "🎲 EXPLORATION AUDIT POOL",
                    "High-novelty proposal near payline cutoff recovered by 5% randomized audit (H_hat_D).",
                )
            };

            evaluated_items.push(EvaluatedTriageItem {
                id: r.id,
                domain: r.domain,
                title: r.title,
                metric_summary: format!(
                    "Priority {:.1}% (${:.2}M)",
                    r.initial_priority_percentile,
                    r.requested_budget_usd / 1_000_000.0
                ),
                voi_index: voi,
                triage_routing: routing.to_string(),
                rationale: rationale.to_string(),
            });
        }
    } else if let Ok(records) = serde_json::from_str::<Vec<UsptoPatentRecord>>(&content) {
        for r in records {
            let q = evaluate_stage_queue(r.examiner_utilization_rho * 100.0, 100.0, 1.0, 1.0);
            let gov = calculate_governor_action(r.examiner_utilization_rho * 100.0, 100.0, 0.85);
            let (routing, rationale) = if gov.recommend_automated_triage {
                (
                    "⚠️ CAPACITY GOVERNOR THROTTLE",
                    "Examiner utilization exceeds rho = 0.85. Automated prior art triage activated.",
                )
            } else {
                (
                    "🪙 ANTI-SPAM STAKING FILTER",
                    "Moderate backlog. Submission deposit filters low-novelty claim floods.",
                )
            };

            evaluated_items.push(EvaluatedTriageItem {
                id: r.id,
                domain: r.domain,
                title: r.title,
                metric_summary: format!(
                    "Pendency {:.1} mo (rho = {:.2})",
                    r.historical_pendency_months, q.utilization
                ),
                voi_index: r.examiner_utilization_rho,
                triage_routing: routing.to_string(),
                rationale: rationale.to_string(),
            });
        }
    } else if let Ok(records) = serde_json::from_str::<Vec<PapersWithCodeRecord>>(&content) {
        for r in records {
            let reported_status = r.sandboxed_execution_status.trim().to_ascii_lowercase();
            let is_reported_verified = matches!(
                reported_status.as_str(),
                "verified" | "passed" | "pass" | "success" | "synthetic_pass"
            );
            let (routing, voi, rationale) = if is_reported_verified {
                (
                    "REPORTED ARTIFACT STATUS: PASS",
                    0.015,
                    "The input record reports a passing or verified artifact status; AETRE did not execute or independently verify the code.",
                )
            } else {
                (
                    "REPORTED ARTIFACT STATUS: NOT VERIFIED",
                    0.005,
                    "The input record does not report a verified artifact status; AETRE did not execute or independently assess the code.",
                )
            };

            evaluated_items.push(EvaluatedTriageItem {
                id: r.id,
                domain: r.domain,
                title: r.title,
                metric_summary: format!(
                    "{}: {}",
                    r.claimed_throughput_speedup, r.sandboxed_execution_status
                ),
                voi_index: voi,
                triage_routing: routing.to_string(),
                rationale: rationale.to_string(),
            });
        }
    } else if let Ok(records) = serde_json::from_str::<Vec<GenericProposalRecord>>(&content) {
        for r in records {
            let mean = r.latent_quality_estimate.unwrap_or(0.5);
            let var = r.epistemic_variance.unwrap_or(0.5);
            let voi = calculate_boundary_voi(mean, var, 1.2, 0.80, 0.50);
            let (routing, rationale) = if voi > 0.15 {
                (
                    "🔍 DEEP VOI REVIEW QUEUE",
                    "High epistemic variance near selection boundary.",
                )
            } else if mean > 1.2 {
                (
                    "🌟 FAST-PASS SPOTLIGHT",
                    "High expected quality with low variance.",
                )
            } else {
                (
                    "🚫 FAST-REJECT FILTER",
                    "Low expected quality with minimal epistemic uncertainty.",
                )
            };

            evaluated_items.push(EvaluatedTriageItem {
                id: r.id,
                domain: r.domain.unwrap_or_else(|| "General".to_string()),
                title: r.title,
                metric_summary: format!("Latent Q: {:.2} (Var: {:.2})", mean, var),
                voi_index: voi,
                triage_routing: routing.to_string(),
                rationale: rationale.to_string(),
            });
        }
    } else {
        eprintln!("Error: Unrecognized JSON schema in file {:?}", file_path);
        std::process::exit(1);
    }

    if json_out {
        println!(
            "{}",
            serde_json::to_string_pretty(&evaluated_items).unwrap()
        );
        return;
    }

    if csv_out {
        println!("ID,Domain,Title,MetricSummary,VOIIndex,TriageRouting");
        for item in &evaluated_items {
            println!(
                "\"{}\",\"{}\",\"{}\",\"{}\",{:.4},\"{}\"",
                item.id,
                item.domain,
                item.title,
                item.metric_summary,
                item.voi_index,
                item.triage_routing
            );
        }
        return;
    }

    println!(
        "\n=== Dynamic Benchmark Evaluation: {} ===",
        file_path.file_name().unwrap().to_string_lossy()
    );
    println!("Total Records Processed: {}", evaluated_items.len());
    println!("\n| ID | Domain | Title | Metrics | VOI Index | Routing |");
    println!("| :--- | :--- | :--- | :---: | :---: | :--- |");

    let mut boundary_count = 0;
    let mut fast_count = 0;

    for item in &evaluated_items {
        if item.triage_routing.contains("DEEP")
            || item.triage_routing.contains("AUDIT")
            || item.triage_routing.contains("THROTTLE")
        {
            boundary_count += 1;
        } else {
            fast_count += 1;
        }

        println!(
            "| {} | {} | {} | {} | {:.3} | {} |",
            item.id,
            item.domain,
            item.title,
            item.metric_summary,
            item.voi_index,
            item.triage_routing
        );
    }

    let total = evaluated_items.len().max(1);
    let boundary_pct = (boundary_count as f64 / total as f64) * 100.0;
    let fast_pct = (fast_count as f64 / total as f64) * 100.0;

    println!(
        "\n[Summary] High-Attention Boundary Cases: {} ({:.1}%) | Fast-Filtered: {} ({:.1}%)",
        boundary_count, boundary_pct, fast_count, fast_pct
    );
    println!("[Impact]  Human Review Effort Saved: ~{:.0}%\n", fast_pct);
}

// =========================================================================
// MONTE CARLO BENCHMARK COMMAND
// =========================================================================

fn run_benchmark_cmd(args: &[String]) {
    let mut replications = 500;
    let mut json_out = false;
    let mut csv_out = false;

    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--replications" | "-r" => {
                if let Some(val) = args.get(i + 1) {
                    replications = val.parse().unwrap_or(500);
                    i += 1;
                }
            }
            "--json" => json_out = true,
            "--csv" => csv_out = true,
            _ => {}
        }
        i += 1;
    }

    let mut rng = thread_rng();
    let params = Parameters::default();
    let stats = run_benchmark_replications(&mut rng, replications, &params);

    if json_out {
        println!("{}", serde_json::to_string_pretty(&stats).unwrap());
        return;
    }

    if csv_out {
        println!("Regime,Arrivals,Acceptances,QualityThroughputMean,QualityThroughputRunInterval_Low,QualityThroughputRunInterval_High,MeanQualityMean,MeanQualityRunInterval_Low,MeanQualityRunInterval_High,UnconventionalRecallMean,UnconventionalRecallRunInterval_Low,UnconventionalRecallRunInterval_High,FalseDiscoveryRateMean,HumanReviewsMean,EstimatedHiddenMean");
        for r in &stats {
            println!(
                "\"{}\",{},{},{:.2},{:.2},{:.2},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4},{:.1},{}",
                r.regime_name,
                r.arrivals,
                r.acceptances,
                r.quality_throughput_mean,
                r.quality_throughput_run_interval.0,
                r.quality_throughput_run_interval.1,
                r.mean_accepted_quality_mean,
                r.mean_accepted_quality_run_interval.0,
                r.mean_accepted_quality_run_interval.1,
                r.unconventional_recall_mean,
                r.unconventional_recall_run_interval.0,
                r.unconventional_recall_run_interval.1,
                r.false_discovery_rate_mean,
                r.human_reviews_mean,
                r.estimated_hidden_unconventional_mean
                    .map(|h| format!("{:.1}", h))
                    .unwrap_or_else(|| "".to_string())
            );
        }
        return;
    }

    println!(
        "\n>>> Running Monte Carlo Simulation ({} replications)...\n",
        replications
    );
    println!("| Regime | Arrivals / Acceptances | Quality Throughput [central 95% run interval] | Mean Accepted Quality [central 95% run interval] | High-Value Unconventional Recall [central 95% run interval] | False Discovery Rate |");
    println!("| :--- | :---: | :---: | :---: | :---: | :---: |");

    for r in &stats {
        let hidden_str = if let Some(h) = r.estimated_hidden_unconventional_mean {
            format!(" (est. hidden: {:.1})", h)
        } else {
            String::new()
        };

        println!(
            "| **{}**{} | {} / {} | {:.1} [{:.1}, {:.1}] | {:.3} [{:.3}, {:.3}] | {:.3} [{:.3}, {:.3}] | {:.1}% |",
            r.regime_name,
            hidden_str,
            r.arrivals,
            r.acceptances,
            r.quality_throughput_mean,
            r.quality_throughput_run_interval.0,
            r.quality_throughput_run_interval.1,
            r.mean_accepted_quality_mean,
            r.mean_accepted_quality_run_interval.0,
            r.mean_accepted_quality_run_interval.1,
            r.unconventional_recall_mean,
            r.unconventional_recall_run_interval.0,
            r.unconventional_recall_run_interval.1,
            r.false_discovery_rate_mean * 100.0,
        );
    }
    println!("\n[OK] Benchmark completed successfully.\n");
}

// =========================================================================
// PROPOSITION 1 BOUND COMMAND
// =========================================================================

fn run_bound_cmd(args: &[String]) {
    let mut arrivals = 5000;
    let mut capacity = 200;
    let mut high_rate = 0.067;
    let mut json_out = false;
    let mut csv_out = false;

    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--arrivals" | "-a" => {
                if let Some(v) = args.get(i + 1) {
                    arrivals = v.parse().unwrap_or(5000);
                    i += 1;
                }
            }
            "--capacity" | "-k" => {
                if let Some(v) = args.get(i + 1) {
                    capacity = v.parse().unwrap_or(200);
                    i += 1;
                }
            }
            "--high-rate" | "-p" => {
                if let Some(v) = args.get(i + 1) {
                    high_rate = v.parse().unwrap_or(0.067);
                    i += 1;
                }
            }
            "--json" => json_out = true,
            "--csv" => csv_out = true,
            _ => {}
        }
        i += 1;
    }

    let bound = calculate_proposition_1_bound(arrivals, capacity, high_rate);
    let multipliers = vec![1.0, 2.0, 5.0, 10.0, 20.0, 50.0];
    let curve = generate_recall_scaling_curve(1000, capacity, high_rate, &multipliers);

    if json_out {
        #[derive(Serialize)]
        struct BoundOutput {
            bound: aetre_core::ThroughputRecallBound,
            scaling_curve: Vec<aetre_core::RecallScalingPoint>,
        }
        println!(
            "{}",
            serde_json::to_string_pretty(&BoundOutput {
                bound,
                scaling_curve: curve
            })
            .unwrap()
        );
        return;
    }

    if csv_out {
        println!("Multiplier,Arrivals,Capacity,MaxTheoreticalRecall");
        for pt in curve {
            println!(
                "{},{},{},{:.4}",
                pt.arrival_multiplier,
                pt.arrivals,
                pt.selection_capacity,
                pt.max_theoretical_recall
            );
        }
        return;
    }

    println!("\n=== Proposition 1: Throughput–Recall Bound Evaluation ===");
    println!(
        "Total Candidates (N):             {}",
        bound.total_candidates
    );
    println!(
        "Selection Capacity (K):           {}",
        bound.selection_capacity
    );
    println!(
        "High-Value Prior Share (p_H):     {:.2}%",
        bound.high_value_rate * 100.0
    );
    println!(
        "Expected High-Value Count (H_N):  {:.1}",
        bound.expected_high_value_count
    );
    println!(
        "Theoretical Max Recall (R_N):     {:.3} ({:.1}%)",
        bound.theoretical_max_recall,
        bound.theoretical_max_recall * 100.0
    );
    println!(
        "Is Capacity Constrained:          {}",
        bound.is_capacity_constrained
    );

    println!(
        "\n--- Scaling Trajectory under Fixed Capacity (K = {}) ---",
        capacity
    );
    println!("| Multiplier | Arrivals (N) | Max Theoretical Recall |");
    println!("| :---: | :---: | :---: |");
    for pt in curve {
        println!(
            "| {}x | {} | {:.3} ({:.1}%) |",
            pt.arrival_multiplier,
            pt.arrivals,
            pt.max_theoretical_recall,
            pt.max_theoretical_recall * 100.0
        );
    }
    println!();
}

// =========================================================================
// KINGMAN QUEUE COMMAND
// =========================================================================

fn run_queue_cmd(args: &[String]) {
    let mut arrival_rate = 95.0;
    let mut service_rate = 100.0;
    let mut cv_a = 1.0;
    let mut cv_s = 1.0;
    let mut json_out = false;
    let mut csv_out = false;

    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--arrival-rate" | "-l" => {
                if let Some(v) = args.get(i + 1) {
                    arrival_rate = v.parse().unwrap_or(95.0);
                    i += 1;
                }
            }
            "--service-rate" | "-m" => {
                if let Some(v) = args.get(i + 1) {
                    service_rate = v.parse().unwrap_or(100.0);
                    i += 1;
                }
            }
            "--cv-a" => {
                if let Some(v) = args.get(i + 1) {
                    cv_a = v.parse().unwrap_or(1.0);
                    i += 1;
                }
            }
            "--cv-s" => {
                if let Some(v) = args.get(i + 1) {
                    cv_s = v.parse().unwrap_or(1.0);
                    i += 1;
                }
            }
            "--json" => json_out = true,
            "--csv" => csv_out = true,
            _ => {}
        }
        i += 1;
    }

    let q = evaluate_stage_queue(arrival_rate, service_rate, cv_a, cv_s);
    let gov = calculate_governor_action(arrival_rate, service_rate, 0.80);

    if json_out {
        #[derive(Serialize)]
        struct QueueOutput {
            queue_metrics: aetre_core::StageQueueMetrics,
            governor_action: aetre_core::GovernorAction,
        }
        println!(
            "{}",
            serde_json::to_string_pretty(&QueueOutput {
                queue_metrics: q,
                governor_action: gov
            })
            .unwrap()
        );
        return;
    }

    if csv_out {
        println!("ArrivalRate,ServiceRate,Utilization,MeanWaitTime,InSystemBacklog,IsCongested,TargetRho,ExcessArrivals,RecommendTriage");
        println!(
            "{:.2},{:.2},{:.4},{:.4},{:.2},{},{:.2},{:.2},{}",
            q.arrival_rate,
            q.service_rate,
            q.utilization,
            q.mean_wait_time,
            q.mean_items_in_queue,
            q.is_congested,
            gov.target_utilization,
            gov.excess_arrival_rate,
            gov.recommend_automated_triage
        );
        return;
    }

    println!("\n=== Kingman Heavy-Traffic Queue Telemetry ===");
    println!(
        "Arrival Rate (lambda):        {:.1} items/period",
        q.arrival_rate
    );
    println!(
        "Service Capacity (mu):        {:.1} items/period",
        q.service_rate
    );
    println!(
        "Utilization (rho):            {:.3} ({:.1}%)",
        q.utilization,
        q.utilization * 100.0
    );
    println!(
        "Mean Waiting Time (E[W_q]):   {:.3} periods",
        q.mean_wait_time
    );
    println!(
        "Mean In-System Backlog (L):   {:.1} items",
        q.mean_items_in_queue
    );
    println!("Is Heavily Congested (>=85%): {}", q.is_congested);

    println!("\n=== Capacity Governor Recommendation ===");
    println!(
        "Target Sustainable Rho:       {:.2}",
        gov.target_utilization
    );
    println!(
        "Max Sustainable Arrivals:     {:.1} items/period",
        gov.max_human_throughput
    );
    println!(
        "Excess Load:                  {:.1} items/period",
        gov.excess_arrival_rate
    );
    println!(
        "Action:                       {}",
        if gov.recommend_automated_triage {
            "⚠️ THROTTLE / ACTIVATE MULTI-STAGE VOI TRIAGE"
        } else {
            "✅ STABLE: Direct human review sustainable"
        }
    );
    println!();
}

// =========================================================================
// EXPLORATION AUDIT COMMAND
// =========================================================================

fn run_audit_cmd(args: &[String]) {
    let mut pool = 4800;
    let mut sample = 25;
    let mut found = 1;
    let mut json_out = false;
    let mut csv_out = false;

    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--pool" | "-N" => {
                if let Some(v) = args.get(i + 1) {
                    pool = v.parse().unwrap_or(4800);
                    i += 1;
                }
            }
            "--sample" | "-n" => {
                if let Some(v) = args.get(i + 1) {
                    sample = v.parse().unwrap_or(25);
                    i += 1;
                }
            }
            "--found" | "-k" => {
                if let Some(v) = args.get(i + 1) {
                    found = v.parse().unwrap_or(1);
                    i += 1;
                }
            }
            "--json" => json_out = true,
            "--csv" => csv_out = true,
            _ => {}
        }
        i += 1;
    }

    let audit = calculate_exploration_audit(pool, sample, found);

    if json_out {
        println!("{}", serde_json::to_string_pretty(&audit).unwrap());
        return;
    }

    if csv_out {
        println!("DeprioritizedPool,SampleAudited,HighValueFound,EstimatedHiddenHV,StdErr,CILower95,CIUpper95");
        println!(
            "{},{},{},{:.2},{:.2},{:.2},{:.2}",
            audit.deprioritized_pool_size,
            audit.audited_sample_size,
            audit.audited_high_value_found,
            audit.estimated_hidden_high_value,
            audit.estimated_hidden_high_value_std_err,
            audit.confidence_interval_95.0,
            audit.confidence_interval_95.1
        );
        return;
    }

    println!("\n=== Randomized Exploration Audit (Horvitz-Thompson Estimator) ===");
    println!(
        "Deprioritized Pool Size (N_D):      {}",
        audit.deprioritized_pool_size
    );
    println!(
        "Random Audit Sample (n):            {}",
        audit.audited_sample_size
    );
    println!(
        "High-Value Unconventional Found (k): {}",
        audit.audited_high_value_found
    );
    println!(
        "Estimated Hidden High-Value (H_hat): {:.1}",
        audit.estimated_hidden_high_value
    );
    println!(
        "Standard Error:                     {:.2}",
        audit.estimated_hidden_high_value_std_err
    );
    println!(
        "95% Confidence Interval:            [{:.1}, {:.1}]",
        audit.confidence_interval_95.0, audit.confidence_interval_95.1
    );
    println!();
}

// =========================================================================
// STAKING ELASTICITY COMMAND
// =========================================================================

fn run_staking_cmd(args: &[String]) {
    let mut c_gen = 0.01;
    let mut c_sub = 2.0;
    let mut val = 100.0;
    let mut applicants = 5000;
    let mut capacity = 200;
    let mut json_out = false;
    let mut csv_out = false;

    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--c-gen" => {
                if let Some(v) = args.get(i + 1) {
                    c_gen = v.parse().unwrap_or(0.01);
                    i += 1;
                }
            }
            "--c-sub" => {
                if let Some(v) = args.get(i + 1) {
                    c_sub = v.parse().unwrap_or(2.0);
                    i += 1;
                }
            }
            "--val" => {
                if let Some(v) = args.get(i + 1) {
                    val = v.parse().unwrap_or(100.0);
                    i += 1;
                }
            }
            "--applicants" => {
                if let Some(v) = args.get(i + 1) {
                    applicants = v.parse().unwrap_or(5000);
                    i += 1;
                }
            }
            "--capacity" => {
                if let Some(v) = args.get(i + 1) {
                    capacity = v.parse().unwrap_or(200);
                    i += 1;
                }
            }
            "--json" => json_out = true,
            "--csv" => csv_out = true,
            _ => {}
        }
        i += 1;
    }

    let eq = evaluate_submitter_equilibrium(c_gen, c_sub, val, applicants, capacity);
    let curve = generate_staking_curve(c_gen, val, applicants, capacity, 20.0, 10);

    if json_out {
        #[derive(Serialize)]
        struct StakingOutput {
            equilibrium: aetre_core::SubmitterEquilibrium,
            sensitivity_curve: Vec<aetre_core::StakingCurvePoint>,
        }
        println!(
            "{}",
            serde_json::to_string_pretty(&StakingOutput {
                equilibrium: eq,
                sensitivity_curve: curve
            })
            .unwrap()
        );
        return;
    }

    if csv_out {
        println!("Fee,ThresholdProb,EstimatedEntry,SpamDeterredPct");
        for pt in curve {
            println!(
                "{:.2},{:.4},{:.0},{:.2}",
                pt.submission_fee,
                pt.threshold_acceptance_prob,
                pt.estimated_entry_volume,
                pt.low_quality_spam_deterred_pct
            );
        }
        return;
    }

    println!("\n=== Anti-Spam Staking & Entry Elasticity Analysis ===");
    println!(
        "Generation Cost (c_gen):           ${:.2}",
        eq.generation_cost
    );
    println!(
        "Submission Deposit/Fee (c_sub):    ${:.2}",
        eq.submission_fee
    );
    println!(
        "Private Value of Acceptance (V):   ${:.2}",
        eq.private_acceptance_value
    );
    println!(
        "Threshold Win Probability:         {:.3}%",
        eq.threshold_acceptance_prob * 100.0
    );
    println!(
        "Estimated Entry Volume:            {:.0} / {} applicants",
        eq.estimated_entry_volume, eq.total_potential_applicants
    );
    println!(
        "Low-Effort Spam Deterred:          {:.1}%",
        eq.low_quality_spam_deterred_pct
    );

    println!("\n--- Staking Fee Sensitivity Curve (c_sub: $0 to $20) ---");
    println!("| Fee ($) | Win Prob Threshold | Est. Entry Volume | Spam Deterred % |");
    println!("| :---: | :---: | :---: | :---: |");
    for pt in curve {
        println!(
            "| ${:.2} | {:.2}% | {:.0} | {:.1}% |",
            pt.submission_fee,
            pt.threshold_acceptance_prob * 100.0,
            pt.estimated_entry_volume,
            pt.low_quality_spam_deterred_pct
        );
    }
    println!();
}

fn get_license_dir() -> PathBuf {
    if let Ok(profile) = env::var("USERPROFILE") {
        PathBuf::from(profile).join(".aetre")
    } else if let Ok(home) = env::var("HOME") {
        PathBuf::from(home).join(".aetre")
    } else {
        env::temp_dir().join(".aetre")
    }
}

fn run_login_cmd(args: &[String]) {
    if !args.is_empty() {
        eprintln!("Do not pass license keys on the command line; they can leak through shell history and process listings.");
        eprintln!("Run 'aetre login' and paste the key when prompted, or set AETRE_LICENSE_KEY.");
        std::process::exit(2);
    }

    let key = if let Ok(key) = env::var("AETRE_LICENSE_KEY") {
        key
    } else {
        eprint!("Paste your AETRE license key, then press Enter: ");
        let _ = io::stderr().flush();
        let mut key = String::new();
        if let Err(error) = io::stdin().read_line(&mut key) {
            eprintln!("Error reading license key: {error}");
            std::process::exit(1);
        }
        key
    };

    let key = key.trim();
    if key.is_empty() {
        eprintln!("License key cannot be empty.");
        std::process::exit(1);
    }

    let dir = get_license_dir();
    let _ = fs::create_dir_all(&dir);
    let key_path = dir.join("license.key");

    if let Err(e) = fs::write(&key_path, key) {
        eprintln!("Error saving license key to {:?}: {}", key_path, e);
        std::process::exit(1);
    }

    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        if let Err(error) = fs::set_permissions(&key_path, fs::Permissions::from_mode(0o600)) {
            eprintln!("Error securing license key permissions: {error}");
            let _ = fs::remove_file(&key_path);
            std::process::exit(1);
        }
    }

    println!("\n✅ License key successfully activated!");
    println!("Saved to: {}", key_path.display());
    println!("Key prefix: {}", &key[..key.len().min(16)]);
    println!("AETRE CLI and MCP local binaries will now automatically use this license.\n");
}

fn run_license_cmd(_args: &[String]) {
    let dir = get_license_dir();
    let key_path = dir.join("license.key");

    let saved_key = fs::read_to_string(&key_path)
        .ok()
        .map(|s| s.trim().to_string());
    let env_key = env::var("AETRE_LICENSE_KEY")
        .or_else(|_| env::var("AETRE_API_KEY"))
        .ok();

    println!("\n╔═══════════════════════════════════════════════════════════════════════════╗");
    println!("║   AETRE: License Status & Cryptographic Key Inspector                     ║");
    println!("╚═══════════════════════════════════════════════════════════════════════════╝\n");

    if let Some(ref k) = env_key {
        println!("Active Key Source: Environment Variable (AETRE_LICENSE_KEY / AETRE_API_KEY)");
        println!("Key Prefix:        {}", &k[..k.len().min(20)]);
    } else if let Some(ref k) = saved_key {
        println!(
            "Active Key Source: Local Config File ({})",
            key_path.display()
        );
        println!("Key Prefix:        {}", &k[..k.len().min(20)]);
    } else {
        println!("Active Key Source: Default Community Tier (Free Open-Core)");
        println!("Status:            Keyless Mode (3 free pre-flight checks/month)");
        println!("To activate:       Run 'aetre login' or set AETRE_LICENSE_KEY");
    }
    println!();
}
