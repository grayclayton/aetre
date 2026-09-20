use crate::layer::{active_layer, tool_in_layer, Layer};
use serde_json::{json, Value};

pub fn list_tools() -> Value {
    let layer = active_layer();
    let mut tools = all_tools();
    if layer != Layer::All {
        if let Some(list) = tools.as_array_mut() {
            list.retain(|t| {
                t.get("name")
                    .and_then(|n| n.as_str())
                    .is_some_and(|n| tool_in_layer(n, layer))
            });
        }
    }
    tools
}

pub(crate) fn all_tools() -> Value {
    json!([
        {
            "name": "aetre_system_catalog",
            "description": "Comprehensive system introspection returning AETRE architecture, bundled synthetic fixtures, optional data adapters, connectors, mathematical tools, and institutional tiers.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "query_type": {
                        "type": "string",
                        "enum": ["all", "datasets", "tools", "database_connectors", "institutional_tiers", "resources", "prompts", "license"],
                        "description": "Category of system capability metadata to inspect. Defaults to 'all'."
                    },
                    "api_key": {
                        "type": "string",
                        "description": "Optional AETRE API or license key for tier verification."
                    }
                }
            }
        },
        {
            "name": "aetre_triage_proposal",
            "description": "Applies transparent, uncalibrated lexical routing indicators to proposal text, then calculates a VOI index and demonstration stage route (FAST-PASS, FAST-REJECT, or DEEP REVIEW). Not a validated estimate of scientific quality.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "The abstract, executive summary, or proposal body text to evaluate."
                    },
                    "title": {
                        "type": "string",
                        "description": "Optional title of the proposal."
                    },
                    "selection_boundary": {
                        "type": "number",
                        "description": "Decision cutoff boundary for acceptance. Defaults to 1.2."
                    },
                    "api_key": {
                        "type": "string",
                        "description": "Optional AETRE API or license key."
                    }
                },
                "required": ["text"]
            }
        },
        {
            "name": "aetre_calculate_voi",
            "description": "Calculates the exact Bayesian Value of Information (VOI) for crossing a top-K selection boundary under Gaussian conjugate updates.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "prior_source": { "type": "string", "description": "Where the estimates in this call came from, e.g. \"measured: 2026 cycle\", \"expert elicitation\", \"assumed\". Echoed in the result so a reader can tell a measurement from a guess." },
                    "posterior_mean": {
                        "type": "number",
                        "description": "Current expected latent quality (mu)."
                    },
                    "posterior_variance": {
                        "type": "number",
                        "description": "Current epistemic uncertainty / variance (sigma^2)."
                    },
                    "selection_boundary": {
                        "type": "number",
                        "description": "The threshold quality cutoff for acceptance (tau)."
                    },
                    "signal_noise": {
                        "type": "number",
                        "description": "Standard deviation of the additional review signal. Defaults to 0.8."
                    },
                    "review_cost": {
                        "type": "number",
                        "description": "Cost of conducting the review. Defaults to 0.5."
                    },
                    "api_key": {
                        "type": "string",
                        "description": "Optional AETRE API or license key."
                    }
                },
                "required": ["posterior_mean", "posterior_variance", "selection_boundary"]
            }
        },
        {
            "name": "aetre_check_governor",
            "description": "Evaluates evaluator queue load using Kingman's Heavy-Traffic approximation and returns governor throttle recommendations when utilization exceeds rho >= 0.85.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "arrival_rate": {
                        "type": "number",
                        "description": "Arrival rate of submissions (lambda), items per period."
                    },
                    "service_rate": {
                        "type": "number",
                        "description": "Review capacity of the committee/system (mu), items per period."
                    },
                    "cv_arrivals": {
                        "type": "number",
                        "description": "Coefficient of variation of arrivals (c_a). Defaults to 1.0."
                    },
                    "cv_service": {
                        "type": "number",
                        "description": "Coefficient of variation of review duration (c_s). Defaults to 1.0."
                    },
                    "target_utilization": {
                        "type": "number",
                        "description": "Target sustainable utilization ceiling (rho_target). Defaults to 0.85."
                    },
                    "api_key": {
                        "type": "string",
                        "description": "Enterprise license key required."
                    }
                },
                "required": ["arrival_rate", "service_rate"]
            }
        },
        {
            "name": "aetre_exploration_audit",
            "description": "Calculates the unbiased Horvitz-Thompson exploration audit estimator (H_hat_D) and 95% confidence intervals on deprioritized candidates to catch false negative breakthroughs.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "deprioritized_pool_size": {
                        "type": "integer",
                        "description": "Total size of the rejected or deprioritized candidate pool (N_D)."
                    },
                    "audited_sample_size": {
                        "type": "integer",
                        "description": "Number of randomly sampled candidates audited (m_D)."
                    },
                    "audited_high_value_found": {
                        "type": "integer",
                        "description": "Number of high-value unconventional breakthroughs found in the audit sample."
                    },
                    "api_key": {
                        "type": "string",
                        "description": "Enterprise license key required."
                    }
                },
                "required": ["deprioritized_pool_size", "audited_sample_size", "audited_high_value_found"]
            }
        },
        {
            "name": "aetre_evaluate_staking",
            "description": "Simulates submitter entry equilibrium under generative AI generation costs and refundable submission deposits to curb spam floods.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "generation_cost": {
                        "type": "number",
                        "description": "AI generation cost per candidate (c_gen), e.g. $0.05."
                    },
                    "submission_fee": {
                        "type": "number",
                        "description": "Required deposit or submission stake (c_sub), e.g. $5.00."
                    },
                    "private_acceptance_value": {
                        "type": "number",
                        "description": "Submitter's private value of winning acceptance (V), e.g. $100.00."
                    },
                    "total_potential_applicants": {
                        "type": "integer",
                        "description": "Total potential applicant pool (N), e.g. 5000."
                    },
                    "acceptance_capacity": {
                        "type": "integer",
                        "description": "Total available acceptance slots (K), e.g. 200."
                    },
                    "api_key": {
                        "type": "string",
                        "description": "Enterprise license key required."
                    }
                },
                "required": ["generation_cost", "submission_fee", "private_acceptance_value", "total_potential_applicants", "acceptance_capacity"]
            }
        },
        {
            "name": "aetre_proposition_1_bound",
            "description": "Calculates Proposition 1 theoretical recall ceiling R_N <= min(1, K_N / H_N) to determine if a pipeline is mathematically capacity-constrained.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "prior_source": { "type": "string", "description": "Where the estimates in this call came from, e.g. \"measured: 2026 cycle\", \"expert elicitation\", \"assumed\". Echoed in the result so a reader can tell a measurement from a guess." },
                    "total_candidates": {
                        "type": "integer",
                        "description": "Total candidate arrival volume (N)."
                    },
                    "selection_capacity": {
                        "type": "integer",
                        "description": "Available selection capacity (K)."
                    },
                    "high_value_rate": {
                        "type": "number",
                        "description": "Prior fraction of high-value ideas in population (p_H), e.g. 0.067."
                    },
                    "api_key": {
                        "type": "string",
                        "description": "Optional AETRE API or license key."
                    }
                },
                "required": ["total_candidates", "selection_capacity", "high_value_rate"]
            }
        },
        {
            "name": "aetre_correlated_posterior_update",
            "description": "Calculates Bayesian posterior mean and uncertainty under correlated multi-agent evaluator noise (rho_corr), preventing artificial overconfidence from redundant LLM outputs.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "prior_source": { "type": "string", "description": "Where the estimates in this call came from, e.g. \"measured: 2026 cycle\", \"expert elicitation\", \"assumed\". Echoed in the result so a reader can tell a measurement from a guess." },
                    "prior_mean": {
                        "type": "number",
                        "description": "Prior mean of candidate quality (mu_0)."
                    },
                    "prior_variance": {
                        "type": "number",
                        "description": "Prior variance of candidate quality (sigma_0^2)."
                    },
                    "evaluations": {
                        "type": "array",
                        "description": "List of evaluator agent scores and noise standard deviations.",
                        "items": {
                            "type": "object",
                            "properties": {
                                "agent_id": { "type": "string" },
                                "score": { "type": "number" },
                                "noise_sd": { "type": "number" }
                            },
                            "required": ["agent_id", "score", "noise_sd"]
                        }
                    },
                    "inter_agent_correlation": {
                        "type": "number",
                        "description": "Pairwise correlation coefficient between evaluator errors (rho in [0, 1)). Defaults to 0.5."
                    },
                    "api_key": {
                        "type": "string",
                        "description": "Enterprise license key required."
                    }
                },
                "required": ["prior_mean", "prior_variance", "evaluations"]
            }
        },
        {
            "name": "aetre_heavy_tailed_voi",
            "description": "Calculates Generalized Pareto / Heavy-Tailed Value of Information (VOI) to optimize selection pipelines for positive black swan breakthrough discovery.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "prior_source": { "type": "string", "description": "Where the estimates in this call came from, e.g. \"measured: 2026 cycle\", \"expert elicitation\", \"assumed\". Echoed in the result so a reader can tell a measurement from a guess." },
                    "posterior_mean": {
                        "type": "number",
                        "description": "Current expected candidate quality (mu)."
                    },
                    "posterior_variance": {
                        "type": "number",
                        "description": "Current epistemic uncertainty (sigma^2)."
                    },
                    "selection_boundary": {
                        "type": "number",
                        "description": "Threshold cutoff boundary for selection (tau)."
                    },
                    "tail_index_alpha": {
                        "type": "number",
                        "description": "Pareto tail index alpha > 1.0 (e.g. 1.5 for heavy-tailed scientific/biotech innovation). Defaults to 1.5."
                    },
                    "signal_noise": {
                        "type": "number",
                        "description": "Noise standard deviation of additional deep review. Defaults to 0.8."
                    },
                    "review_cost": {
                        "type": "number",
                        "description": "Cost of conducting review. Defaults to 0.5."
                    },
                    "api_key": {
                        "type": "string",
                        "description": "Enterprise license key required."
                    }
                },
                "required": ["posterior_mean", "posterior_variance", "selection_boundary"]
            }
        },
        {
            "name": "aetre_quadratic_staking",
            "description": "Calculates super-linear anti-sybil staking deposit requirements (Stake(m) = S_0 * m^gamma) to deter mass AI spam submissions while preserving human entry.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "base_fee": {
                        "type": "number",
                        "description": "Base deposit for a single submission (S_0), e.g. $5.00."
                    },
                    "escalation_exponent": {
                        "type": "number",
                        "description": "Escalation exponent gamma >= 1.0 (e.g. 2.0 for quadratic escalation). Defaults to 2.0."
                    },
                    "submission_count": {
                        "type": "integer",
                        "description": "Total submissions attempted by the entity within the time window (m)."
                    },
                    "generation_cost": {
                        "type": "number",
                        "description": "AI generation cost per submission (c_gen). Defaults to 0.05."
                    },
                    "private_acceptance_value": {
                        "type": "number",
                        "description": "Private monetary or prestige payoff if accepted (V). Defaults to 100.0."
                    },
                    "api_key": {
                        "type": "string",
                        "description": "Enterprise license key required."
                    }
                },
                "required": ["base_fee", "submission_count"]
            }
        },
        {
            "name": "aetre_heterogeneous_queues",
            "description": "Evaluates a multi-specialist heterogeneous reviewer network, identifying bottleneck domains and generating capacity rebalancing actions.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "pools": {
                        "type": "array",
                        "description": "List of domain queues with arrival and service parameters.",
                        "items": {
                            "type": "object",
                            "properties": {
                                "domain": { "type": "string" },
                                "arrival_rate": { "type": "number" },
                                "service_rate": { "type": "number" },
                                "cv_arrivals": { "type": "number" },
                                "cv_service": { "type": "number" }
                            },
                            "required": ["domain", "arrival_rate", "service_rate"]
                        }
                    },
                    "api_key": {
                        "type": "string",
                        "description": "Enterprise license key required."
                    }
                },
                "required": ["pools"]
            }
        },
        {
            "name": "aetre_author_preflight_benchmark",
            "description": "Comprehensive pre-submission diagnostic scorecard for authors and researchers, calculating crowd novelty percentile, reviewer disagreement risk, and prescriptive refinement actions.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "title": {
                        "type": "string",
                        "description": "Proposal or paper title."
                    },
                    "text": {
                        "type": "string",
                        "description": "Full proposal abstract or summary."
                    },
                    "selection_boundary": {
                        "type": "number",
                        "description": "Funding or acceptance cutoff threshold (tau). Defaults to 1.2."
                    },
                    "api_key": {
                        "type": "string",
                        "description": "Optional AETRE Pro or Enterprise license key for unlimited checks."
                    }
                },
                "required": ["text"]
            }
        },
        {
            "name": "aetre_simulate_benchmark",
            "description": "Runs a paired-cohort Monte Carlo simulation across all 4 screening regimes, comparing Quality Throughput, FDR, Unconventional Recall, and Human Reviews with central 95% run-to-run outcome intervals (not confidence intervals for the mean).",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "replications": {
                        "type": "integer",
                        "description": "Number of Monte Carlo simulation replicates (default: 50)."
                    },
                    "baseline_arrivals": {
                        "type": "integer",
                        "description": "Baseline arrival volume N (default: 1000)."
                    },
                    "ai_arrival_multiplier": {
                        "type": "number",
                        "description": "Multiplier for synthetic/AI flood regime (default: 5.0)."
                    },
                    "acceptance_capacity": {
                        "type": "integer",
                        "description": "Number of acceptance slots K (default: 200)."
                    },
                    "unconventional_share": {
                        "type": "number",
                        "description": "Prior share of unconventional/novel ideas (default: 0.10)."
                    },
                    "evaluation_budget": {
                        "type": "number",
                        "description": "Total available evaluation budget (default: 1000.0)."
                    },
                    "randomized_audit_budget_share": {
                        "type": "number",
                        "description": "Share of budget allocated to randomized Horvitz-Thompson exploration audits (default: 0.05)."
                    },
                    "api_key": {
                        "type": "string",
                        "description": "Enterprise license key required."
                    }
                }
            }
        },
        {
            "name": "aetre_batch_triage",
            "description": "Batch applies disclosed, uncalibrated lexical indicators to a cohort, computing heuristic ranks, VOI ranks, and demonstration stream allocation (Stream A Fast-Reject, Stream B Deep Review, Stream C Fast-Pass).",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "proposals": {
                        "type": "array",
                        "description": "List of proposals with title and text/abstract.",
                        "items": {
                            "type": "object",
                            "properties": {
                                "title": { "type": "string" },
                                "text": { "type": "string" }
                            },
                            "required": ["text"]
                        }
                    },
                    "selection_boundary": {
                        "type": "number",
                        "description": "Cutoff threshold boundary (default: 1.2)."
                    },
                    "api_key": {
                        "type": "string",
                        "description": "Optional license key."
                    }
                },
                "required": ["proposals"]
            }
        },
        {
            "name": "aetre_recall_scaling_curve",
            "description": "Calculates the Proposition 1 theoretical recall decay curve across arrival expansion scales (e.g. 1x, 2x, 5x, 10x, 20x, 50x) demonstrating capacity collapse points.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "prior_source": { "type": "string", "description": "Where the estimates in this call came from, e.g. \"measured: 2026 cycle\", \"expert elicitation\", \"assumed\". Echoed in the result so a reader can tell a measurement from a guess." },
                    "baseline_arrivals": {
                        "type": "integer",
                        "description": "Baseline candidate arrivals N (default: 1000)."
                    },
                    "selection_capacity": {
                        "type": "integer",
                        "description": "Available selection capacity K (default: 200)."
                    },
                    "high_value_rate": {
                        "type": "number",
                        "description": "Prior high-value fraction in population (default: 0.067)."
                    },
                    "multipliers": {
                        "type": "array",
                        "items": { "type": "number" },
                        "description": "List of arrival multipliers to sweep across (default: [1, 2, 5, 10, 20, 50])."
                    },
                    "api_key": {
                        "type": "string",
                        "description": "Optional license key."
                    }
                }
            }
        },
        {
            "name": "aetre_fit_boundary",
            "description": "Fits the decision boundary to a venue's own calibration data by sweeping thresholds and ranking candidates by boundary VOI at each. The boundary determines whether triage beats chance, and a default carried from another corpus generally does not.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "dataset": {
                        "type": "string",
                        "description": "Path to a JSON array of candidates: {id, split, label, pre_triage_data:{preliminary_mean, preliminary_variance}}. Required: fit against your own data."
                    },
                    "split": { "type": "string", "description": "Split to fit on. Defaults to 'calib'. Fit and evaluate on different splits." },
                    "budget": { "type": "integer", "description": "Review budget K the boundary is optimised for. Defaults to 200." },
                    "grid_min": { "type": "number", "description": "Lowest boundary to try. Defaults to 1.0." },
                    "grid_max": { "type": "number", "description": "Highest boundary to try. Defaults to 10.0." },
                    "grid_step": { "type": "number", "description": "Step between candidate boundaries. Defaults to 0.25." },
                    "api_key": { "type": "string", "description": "Optional license key." }
                },
                "required": ["dataset"]
            }
        },
        {
            "name": "aetre_heldout_backtest",
            "description": "Runs a multi-policy held-out review allocation backtest across 8 triage policies under fixed review budget K, evaluating true decision flips, precision, recall, and paired bootstrap intervals.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "dataset": {
                        "type": "string",
                        "description": "Path to a JSON array of candidates to evaluate: {id, split, label, pre_triage_data:{preliminary_mean, preliminary_variance}}. Omit to run against a bundled six-candidate sample, which illustrates the evaluation but is not evidence."
                    },
                    "budget": {
                        "type": "integer",
                        "description": "Fixed review capacity budget K (default: 50)."
                    },
                    "boundary": {
                        "type": "number",
                        "description": "Acceptance threshold boundary theta (default: 6.0)."
                    },
                    "split": {
                        "type": "string",
                        "description": "Evaluation split ('test', 'dev', 'calib', 'replication', 'all') (default: 'test')."
                    },
                    "api_key": {
                        "type": "string",
                        "description": "Optional license key."
                    }
                }
            }
        },
        {
            "name": "aetre_calibrate_scorer",
            "description": "Fits Platt logistic scaling on continuous model scores and binary labels, returning slope, intercept, Expected Calibration Error (ECE), and Brier score.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "scores": {
                        "type": "array",
                        "items": { "type": "number" },
                        "description": "Raw continuous candidate scores or VOI values."
                    },
                    "labels": {
                        "type": "array",
                        "items": { "type": "integer" },
                        "description": "Binary ground-truth labels (0 or 1)."
                    },
                    "iterations": {
                        "type": "integer",
                        "description": "Calibration optimization iterations (default: 500)."
                    },
                    "learning_rate": {
                        "type": "number",
                        "description": "Optimization learning rate (default: 0.05)."
                    },
                    "api_key": {
                        "type": "string",
                        "description": "Optional license key."
                    }
                },
                "required": ["scores", "labels"]
            }
        },
        {
            "name": "aetre_multi_attribute_voi",
            "description": "Computes multi-attribute Bayesian Value of Information across orthogonal proposal evaluation dimensions (Novelty, Rigor, Impact, Feasibility), outputting composite VOI and optimal dimension-specific review targets.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "dimensions": {
                        "type": "array",
                        "description": "List of evaluation dimensions with name, prior_mean, prior_variance, weight, and review_noise_sd.",
                        "items": {
                            "type": "object",
                            "properties": {
                                "name": { "type": "string" },
                                "prior_mean": { "type": "number" },
                                "prior_variance": { "type": "number" },
                                "weight": { "type": "number" },
                                "threshold": { "type": "number" },
                                "review_noise_sd": { "type": "number" }
                            },
                            "required": ["name", "prior_mean", "prior_variance", "weight"]
                        }
                    },
                    "composite_threshold": {
                        "type": "number",
                        "description": "Composite decision threshold cutoff (default: 6.0)."
                    },
                    "review_cost_per_dim": {
                        "type": "number",
                        "description": "Marginal review cost per dimension (default: 1.0)."
                    },
                    "api_key": {
                        "type": "string",
                        "description": "Optional license key."
                    }
                },
                "required": ["dimensions"]
            }
        },
        {
            "name": "aetre_congestion_matching",
            "description": "Optimizes reviewer-to-proposal assignment by maximizing domain/keyword affinity while enforcing Kingman queue utilization constraints (rho <= 0.85) on individual reviewer workloads.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "proposals": {
                        "type": "array",
                        "description": "List of candidate proposals with id, title, domain, voi_index, required_reviews, and keywords.",
                        "items": {
                            "type": "object",
                            "properties": {
                                "id": { "type": "string" },
                                "title": { "type": "string" },
                                "domain": { "type": "string" },
                                "voi_index": { "type": "number" },
                                "required_reviews": { "type": "integer" },
                                "keywords": { "type": "array", "items": { "type": "string" } }
                            },
                            "required": ["id", "domain", "voi_index"]
                        }
                    },
                    "reviewers": {
                        "type": "array",
                        "description": "List of reviewer profiles with id, name, domain, capacity, current_load, service_rate, arrival_rate, and expertise_tags.",
                        "items": {
                            "type": "object",
                            "properties": {
                                "id": { "type": "string" },
                                "name": { "type": "string" },
                                "domain": { "type": "string" },
                                "capacity": { "type": "integer" },
                                "current_load": { "type": "integer" },
                                "service_rate": { "type": "number" },
                                "arrival_rate": { "type": "number" },
                                "expertise_tags": { "type": "array", "items": { "type": "string" } }
                            },
                            "required": ["id", "name", "domain", "capacity", "service_rate"]
                        }
                    },
                    "target_utilization": {
                        "type": "number",
                        "description": "Maximum allowed reviewer utilization target (default: 0.85)."
                    },
                    "api_key": {
                        "type": "string",
                        "description": "Enterprise license key."
                    }
                },
                "required": ["proposals", "reviewers"]
            }
        },
        {
            "name": "aetre_sequential_stopping_rule",
            "description": "Calculates optimal dynamic Bayesian stopping boundaries for sequential reviews (Accept, Reject, or Solicit More Reviews) based on posterior decision confidence and boundary VOI.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "prior_source": { "type": "string", "description": "Where the estimates in this call came from, e.g. \"measured: 2026 cycle\", \"expert elicitation\", \"assumed\". Echoed in the result so a reader can tell a measurement from a guess." },
                    "prior_mean": {
                        "type": "number",
                        "description": "Baseline prior mean quality (e.g. 5.0)."
                    },
                    "prior_variance": {
                        "type": "number",
                        "description": "Baseline prior epistemic variance (e.g. 1.0)."
                    },
                    "threshold": {
                        "type": "number",
                        "description": "Decision acceptance threshold cutoff (e.g. 6.0)."
                    },
                    "reviews": {
                        "type": "array",
                        "description": "Ordered sequence of completed reviewer scores with noise_sd and cost.",
                        "items": {
                            "type": "object",
                            "properties": {
                                "step": { "type": "integer" },
                                "reviewer_id": { "type": "string" },
                                "score": { "type": "number" },
                                "noise_sd": { "type": "number" },
                                "cost": { "type": "number" }
                            },
                            "required": ["score", "noise_sd"]
                        }
                    },
                    "next_review_noise_sd": {
                        "type": "number",
                        "description": "Expected noise SD of a future review (default: 0.80)."
                    },
                    "next_review_cost": {
                        "type": "number",
                        "description": "Cost of soliciting an additional review (default: 1.0)."
                    },
                    "confidence_threshold": {
                        "type": "number",
                        "description": "Target confidence probability to stop early (default: 0.90)."
                    },
                    "api_key": {
                        "type": "string",
                        "description": "Optional license key."
                    }
                },
                "required": ["prior_mean", "prior_variance", "threshold", "reviews"]
            }
        },
        {
            "name": "governed_bellman_triage",
            "description": "Pillar I Bellman Governor: evaluates Bayesian dynamic programming stopping policy over multi-stage pass lattices under asymmetric loss stakes (L/R), returning optimal action (CONTINUE, HALT_AND_COMMIT, HALT_AND_REJECT), expected utility, VOI, and critical threshold p*.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "reward": { "type": "number", "description": "Conforming candidate net reward R (default: 0.02)." },
                    "loss": { "type": "number", "description": "Defective candidate loss penalty L (default: 0.10)." },
                    "prior": { "type": "number", "description": "Prior belief in conforming status (default: 0.50)." },
                    "stage": { "type": "integer", "description": "Current verification stage index (default: 0)." },
                    "consecutive_passes": { "type": "integer", "description": "Number of consecutive test passes observed (default: 0)." },
                    "max_stages": { "type": "integer", "description": "Maximum verification stages horizon H (default: 4)." },
                    "defect_leakage": { "type": "number", "description": "Defect leakage rate q (default: 0.5875)." },
                    "api_key": { "type": "string", "description": "Optional license key." }
                }
            }
        },
        {
            "name": "governed_review_boundary",
            "description": "Section 4.1 Tripartite Review Boundary: evaluates whether an autonomous coding candidate should be AUTO-admitted, sent to human REVIEW, or ABSTAINED based on reviewer effort cost and knapsack capacity shadow price lambda_K.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "prior_source": { "type": "string", "description": "Where the estimates in this call came from, e.g. \"measured: 2026 cycle\", \"expert elicitation\", \"assumed\". Echoed in the result so a reader can tell a measurement from a guess." },
                    "belief": { "type": "number", "description": "Current posterior belief probability in [0, 1]." },
                    "reward": { "type": "number", "description": "Conforming net reward R (default: 0.02)." },
                    "loss": { "type": "number", "description": "Defective loss penalty L (default: 0.10)." },
                    "review_cost": { "type": "number", "description": "Direct cost of human reviewer examination (default: 0.002)." },
                    "shadow_price_lambda": { "type": "number", "description": "Knapsack queue capacity congestion shadow price lambda_K (default: 0.0)." },
                    "review_accuracy": { "type": "number", "description": "Probability reviewer correctly verifies valid code (default: 1.0)." },
                    "api_key": { "type": "string", "description": "Optional license key." }
                },
                "required": ["belief"]
            }
        },
        {
            "name": "governed_knapsack_admit",
            "description": "Pillar V Knapsack Queue Controller: packs candidate pull requests into the review queue under capacity budget K using the c-mu rule (density rho_i = E[U_i] / k_i) and calculates the dual capacity shadow price lambda_K.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "candidates": {
                        "type": "array",
                        "description": "List of candidate submissions with candidate_id, posterior_belief, and optional reward, loss, review_cost.",
                        "items": {
                            "type": "object",
                            "properties": {
                                "candidate_id": { "type": "string" },
                                "task_id": { "type": "string" },
                                "posterior_belief": { "type": "number" },
                                "reward": { "type": "number" },
                                "loss": { "type": "number" },
                                "review_cost": { "type": "number" }
                            },
                            "required": ["candidate_id", "posterior_belief"]
                        }
                    },
                    "capacity_k": { "type": "number", "description": "Total reviewer capacity budget K (e.g. 3.0 review slots or hours)." },
                    "review_cost_k": { "type": "number", "description": "Default review effort cost per candidate (default: 1.0)." },
                    "api_key": { "type": "string", "description": "Optional license key." }
                },
                "required": ["candidates", "capacity_k"]
            }
        },
        {
            "name": "governed_gate_pr",
            "description": "Road A Tiered Verification Gate: evaluates host pre-checks (Tier 0 syntactic AST parsing and structural validation) on candidate Python code or pull request patches to short-circuit broken submissions before expensive container CI escalation.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "candidate_id": { "type": "string", "description": "Identifier for the pull request or patch." },
                    "code": { "type": "string", "description": "Source code or patch string to screen." },
                    "api_key": { "type": "string", "description": "Optional license key." }
                },
                "required": ["code"]
            }
        },
        {
            "name": "governed_evaluate_action",
            "description": "Micro-level loss evaluation comparing autonomous execution against deferral to human review, under an asymmetric loss matrix weighted by irreversibility.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "prior_source": { "type": "string", "description": "Where the estimates in this call came from, e.g. \"measured: 2026 cycle\", \"expert elicitation\", \"assumed\". Echoed in the result so a reader can tell a measurement from a guess." },
                    "action_name": { "type": "string", "description": "Identifier for the candidate agent action." },
                    "consequence_distribution": {
                        "type": "object",
                        "description": "Distribution over the consequences of acting autonomously.",
                        "properties": {
                            "p_failure": { "type": "number", "description": "Probability the action is wrong (0-1)." },
                            "severity_mean": { "type": "number", "description": "Mean loss conditional on failure." },
                            "severity_std": { "type": "number", "description": "Standard deviation of that loss." },
                            "reversibility": { "type": "number", "description": "Fraction of the loss recoverable after the fact (0-1)." }
                        }
                    },
                    "review_cost": { "type": "number", "description": "Cost of deferring the action to a human reviewer." },
                    "api_key": { "type": "string", "description": "Optional license key." }
                },
                "required": ["consequence_distribution"]
            }
        },
        {
            "name": "governed_stopping_policy",
            "description": "Solves the finite-horizon optimal stopping problem by backward induction: at each step the agent either takes the terminal payoff or pays the continuation cost for one more step of evidence.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "horizon_steps": { "type": "integer", "description": "Number of steps in the lattice. Defaults to the length of terminal_payoffs." },
                    "cost_per_step": { "type": "number", "description": "Cost charged for each additional step of evidence gathering." },
                    "terminal_payoffs": {
                        "type": "array",
                        "items": { "type": "number" },
                        "description": "Payoff from stopping at each step. The last value is carried forward if shorter than the horizon."
                    },
                    "api_key": { "type": "string", "description": "Optional license key." }
                },
                "required": ["terminal_payoffs"]
            }
        },
        {
            "name": "governed_invariant_check",
            "description": "Evaluates declarative safety invariants against a runtime state snapshot before a state change is committed, returning every violated rule rather than the first.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "state_snapshot": { "type": "object", "description": "Flat object of runtime state fields to test." },
                    "invariant_rules": {
                        "type": "array",
                        "items": { "type": "string" },
                        "description": "Rules of the form '<field> <op> <number>' with op in <=, <, >=, >, ==, != , or 'exists <field>'."
                    },
                    "api_key": { "type": "string", "description": "Optional license key." }
                },
                "required": ["state_snapshot", "invariant_rules"]
            }
        },
        {
            "name": "governed_shadow_price",
            "description": "Computes the marginal shadow price of scarce reviewer time (lambda_K) from queue backlog and review capacity under a heavy-tailed utility distribution, and the factor by which it elevates the admission cutoff.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "prior_source": { "type": "string", "description": "Where the estimates in this call came from, e.g. \"measured: 2026 cycle\", \"expert elicitation\", \"assumed\". Echoed in the result so a reader can tell a measurement from a guess." },
                    "queue_backlog": { "type": "number", "description": "Number of candidates awaiting review." },
                    "reviewer_headcount": { "type": "number", "description": "Number of available reviewers." },
                    "reviews_per_reviewer": { "type": "number", "description": "Review slots per reviewer in the period. Defaults to 8." },
                    "tail_index_alpha": { "type": "number", "description": "Pareto tail index of the utility distribution. Defaults to 1.25." },
                    "api_key": { "type": "string", "description": "Optional license key." }
                },
                "required": ["queue_backlog", "reviewer_headcount"]
            }
        },
        {
            "name": "governed_recall_scaling",
            "description": "Fits the recall saturation curve Recall(K) = 1 - exp(-gamma K) to observed budget and recall pairs by least squares, returning the fitted gamma and the budget beyond which marginal recall stops paying for itself.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "historical_budget_K": {
                        "type": "array",
                        "items": { "type": "number" },
                        "description": "Observed review budgets."
                    },
                    "recall_points": {
                        "type": "array",
                        "items": { "type": "number" },
                        "description": "Recall achieved at each budget (0-1), positionally paired with historical_budget_K."
                    },
                    "marginal_recall_threshold": { "type": "number", "description": "Marginal recall per unit budget below which spending stops. Defaults to 0.001." },
                    "api_key": { "type": "string", "description": "Optional license key." }
                },
                "required": ["historical_budget_K", "recall_points"]
            }
        },
        {
            "name": "governed_runtime_audit",
            "description": "Generates a tamper-evident SHA-256 decision receipt over an execution's identifier, payload and timestamp, so a gated agent action can be verified after the fact.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "execution_id": { "type": "string", "description": "Identifier of the agent execution being recorded." },
                    "decision_payload": { "type": "object", "description": "The decision record to bind into the receipt." },
                    "include_posterior_trace": { "type": "boolean", "description": "Echo the posterior trace from the payload into the receipt." },
                    "api_key": { "type": "string", "description": "Optional license key." }
                },
                "required": ["execution_id"]
            }
        },
        {
            "name": "aetre_investment_benchmark",
            "description": "Runs the venture dealflow triage benchmark: generates a synthetic heavy-tailed cohort of deals and compares status-quo preliminary-score screening against AETRE heavy-tailed VOI triage under a fixed diligence budget.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "n_deals": { "type": "integer", "description": "Deals in the synthetic cohort. Defaults to 1000, capped at 20000." },
                    "diligence_budget": { "type": "integer", "description": "Deals that can be taken to full diligence. Defaults to 50." },
                    "tail_alpha": { "type": "number", "description": "Pareto tail index of the return distribution. Defaults to 1.25." },
                    "wrapper_pct": { "type": "number", "description": "Share of the cohort that is well-packaged but low-substance. Defaults to 0.30." },
                    "selection_boundary": { "type": "number", "description": "Preliminary score boundary for selection. Defaults to 6.0." },
                    "hours_per_diligence": { "type": "number", "description": "Analyst hours consumed per deal diligenced. Defaults to 20.0." },
                    "api_key": { "type": "string", "description": "Optional license key." }
                },
                "required": []
            }
        },
        {
            "name": "aetre_staking_curve",
            "description": "Sweeps the submission fee across a range and returns the submitter equilibrium at each point, showing how entry volume and low-quality deterrence respond to the staking fee rather than evaluating a single fee.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "c_gen": { "type": "number", "description": "Marginal cost of generating a proposal. Defaults to 0.01." },
                    "private_acceptance_value": { "type": "number", "description": "Private value of acceptance to the submitter. Defaults to 100.0." },
                    "total_potential_applicants": { "type": "integer", "description": "Size of the applicant pool. Defaults to 5000." },
                    "acceptance_capacity": { "type": "integer", "description": "Slots available for acceptance. Defaults to 200." },
                    "max_fee": { "type": "number", "description": "Highest fee on the swept curve. Defaults to 20.0." },
                    "steps": { "type": "integer", "description": "Points on the curve. Defaults to 10, capped at 200." },
                    "api_key": { "type": "string", "description": "Optional license key." }
                },
                "required": []
            }
        }
    ])
}
