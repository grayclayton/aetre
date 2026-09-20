use crate::format;
use crate::helpers::*;
use crate::heuristics::analyze_text_heuristics;
use crate::layer::{active_layer, tool_in_layer, Layer};
use crate::license::{self, get_license_tier, get_quota_status};
use crate::prompts::list_prompts;
use crate::resources::list_resources;
use crate::schemas::list_tools;
use aetre_core::*;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::fmt::Write as _;

/// The bundled held-out fixture, embedded so `aetre_heldout_backtest` works from
/// any working directory. An MCP client spawns the server wherever it likes, and
/// a relative path only ever resolved inside a checkout of this repository.
///
/// It is six candidates. It demonstrates the shape of the evaluation; it is not
/// evidence, and the tool says so in `dataset_source`.
const BUNDLED_BACKTEST: &str = include_str!("../fixtures/openreview_heldout_backtest.json");

/// Rejects a call whose arguments do not match the tool's declared schema.
///
/// Without this a misspelled parameter is silently ignored and the default is
/// scored instead, so the caller gets a confident answer to a question it did
/// not ask. A model guessing `abstract` where the schema says `text` is exactly
/// the case this catches. Fail closed, which is what the rest of the system does.
fn schema_violation(name: &str, args: &Value) -> Option<Value> {
    let tools = crate::schemas::all_tools();
    let tool = tools
        .as_array()?
        .iter()
        .find(|t| t.get("name").and_then(|n| n.as_str()) == Some(name))?;
    let schema = tool.get("inputSchema")?;
    let properties = schema.get("properties")?.as_object()?;
    let required: Vec<&str> = schema
        .get("required")
        .and_then(|r| r.as_array())
        .map(|a| a.iter().filter_map(|v| v.as_str()).collect())
        .unwrap_or_default();

    let empty = serde_json::Map::new();
    let supplied = args.as_object().unwrap_or(&empty);

    let missing: Vec<&str> = required
        .iter()
        .copied()
        .filter(|key| !supplied.contains_key(*key))
        .collect();
    let unknown: Vec<&str> = supplied
        .keys()
        .map(String::as_str)
        .filter(|key| !properties.contains_key(*key))
        .collect();

    if missing.is_empty() && unknown.is_empty() {
        return None;
    }

    let mut accepted: Vec<&str> = properties.keys().map(String::as_str).collect();
    accepted.sort_unstable();

    let mut problems: Vec<String> = Vec::new();
    if !missing.is_empty() {
        problems.push(format!(
            "missing required argument(s): {}",
            missing.join(", ")
        ));
    }
    if !unknown.is_empty() {
        problems.push(format!("unrecognised argument(s): {}", unknown.join(", ")));
    }

    Some(json!({
        "content": [{
            "type": "text",
            "text": format!(
                "{} does not accept these arguments.
{}
Accepted: {}.",
                name,
                problems.join("
    "),
                accepted.join(", ")
            )
        }],
        "isError": true
    }))
}

/// A candidate as the held-out corpora store one.
#[allow(dead_code)]
#[derive(serde::Deserialize)]
pub(crate) struct BacktestCandidate {
    pub(crate) id: String,
    pub(crate) split: String,
    pub(crate) label: u8,
    pub(crate) pre_triage_data: BacktestPreTriage,
}

#[allow(dead_code)]
#[derive(serde::Deserialize)]
pub(crate) struct BacktestPreTriage {
    pub(crate) preliminary_mean: f64,
    pub(crate) preliminary_variance: f64,
    pub(crate) m_reviews_count: Option<usize>,
    pub(crate) preliminary_mean_confidence: Option<f64>,
}

pub(crate) struct BacktestScore {
    pub(crate) evaluated: usize,
    pub(crate) positives: usize,
    pub(crate) budget: usize,
    pub(crate) caught: usize,
    pub(crate) precision: f64,
    pub(crate) recall: f64,
}

/// Ranks candidates by boundary VOI and reports what the top K catches.
///
/// Shared by the backtest and the boundary fit so the fitted threshold is
/// chosen against exactly the procedure it will later be used for.
pub(crate) fn score_backtest(
    records: &[BacktestCandidate],
    split: &str,
    budget: usize,
    boundary: f64,
) -> BacktestScore {
    let eval: Vec<&BacktestCandidate> = records
        .iter()
        .filter(|r| r.split == split || split == "all")
        .collect();
    let n = eval.len();
    let positives = eval.iter().filter(|r| r.label == 1).count();

    let mut scores: Vec<(usize, f64)> = eval
        .iter()
        .enumerate()
        .map(|(idx, r)| {
            let m = r.pre_triage_data.preliminary_mean;
            let m_count = r.pre_triage_data.m_reviews_count.unwrap_or(2) as f64;
            let v = r.pre_triage_data.preliminary_variance.max(0.01);
            let conf = r
                .pre_triage_data
                .preliminary_mean_confidence
                .unwrap_or(3.0)
                .clamp(1.0, 5.0);
            let sig_noise = (2.0 / conf).max(0.3);
            let post_var = (v / m_count).max(0.01);
            (
                idx,
                aetre_core::calculate_boundary_voi(m, post_var, boundary, sig_noise, 0.50),
            )
        })
        .collect();
    scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    let k_eff = budget.min(n);
    let caught = scores[..k_eff]
        .iter()
        .filter(|(idx, _)| eval[*idx].label == 1)
        .count();

    BacktestScore {
        evaluated: n,
        positives,
        budget: k_eff,
        caught,
        precision: caught as f64 / k_eff.max(1) as f64,
        recall: caught as f64 / positives.max(1) as f64,
    }
}

/// Records, in the result, where the caller's estimates came from and which
/// arguments they never supplied.
///
/// A tool that takes a prior returns the same confident numbers whether the
/// prior was measured or invented, and a model filling a well-specified schema
/// will invent one readily. `prior_source` is what the caller claims;
/// `defaulted_arguments` is what actually happened, which cannot be overstated.
fn annotate_provenance(name: &str, args: &Value, mut result: Value) -> Value {
    if result.get("isError").and_then(Value::as_bool) == Some(true) {
        return result;
    }
    let tools = crate::schemas::all_tools();
    let Some(tool) = tools.as_array().and_then(|a| {
        a.iter()
            .find(|t| t.get("name").and_then(|n| n.as_str()) == Some(name))
    }) else {
        return result;
    };
    let declares_prior_source = tool
        .pointer("/inputSchema/properties/prior_source")
        .is_some();
    if !declares_prior_source {
        return result;
    }

    let empty = serde_json::Map::new();
    let supplied = args.as_object().unwrap_or(&empty);
    let mut defaulted: Vec<String> = tool
        .pointer("/inputSchema/properties")
        .and_then(Value::as_object)
        .map(|props| {
            props
                .keys()
                .filter(|k| k.as_str() != "api_key" && k.as_str() != "prior_source")
                .filter(|k| !supplied.contains_key(k.as_str()))
                .cloned()
                .collect()
        })
        .unwrap_or_default();
    defaulted.sort();

    let stated = supplied
        .get("prior_source")
        .and_then(Value::as_str)
        .unwrap_or("unstated: the caller did not say where these estimates came from");

    // The payload is JSON inside a text block; annotate it there if we can.
    let Some(text) = result.pointer("/content/0/text").and_then(Value::as_str) else {
        return result;
    };
    let Ok(mut body) = serde_json::from_str::<Value>(text) else {
        return result;
    };
    if let Some(map) = body.as_object_mut() {
        map.insert("prior_source".into(), json!(stated));
        if !defaulted.is_empty() {
            map.insert("defaulted_arguments".into(), json!(defaulted));
        }
    }
    if let Some(slot) = result.pointer_mut("/content/0/text") {
        *slot = json!(serde_json::to_string_pretty(&body).unwrap_or_default());
    }
    result
}

pub fn call_tool(name: &str, args: Value) -> Value {
    let layer = active_layer();
    if !tool_in_layer(name, layer) {
        return json!({
            "content": [{
                "type": "text",
                "text": format!(
                    "Tool {name} is not exposed: this server was started with --layer={}.",
                    match layer { Layer::Macro => "macro", Layer::Micro => "micro", Layer::All => "all" }
                )
            }],
            "isError": true
        });
    }
    if let Some(rejection) = schema_violation(name, &args) {
        return rejection;
    }

    let result = dispatch_tool(name, args.clone());
    annotate_provenance(name, &args, result)
}

fn dispatch_tool(name: &str, args: Value) -> Value {
    let tier = get_license_tier(&args);
    match name {
        "aetre_system_catalog" => {
            let query_type = get_str(&args, "query_type", "all");

            let out = match query_type {
                "license" | "pricing" | "quota" => json!({
                    "section": "License & Quota Status",
                    "license_details": get_quota_status(tier),
                    "active_license_resolution": license::resolve_license(&args)
                }),

                "datasets" => json!({
                    "section": "Benchmark Datasets",
                    "catalog_uri": "aetre://catalog/datasets",
                    "available_datasets": [
                        { "id": "openreview", "name": "Synthetic Peer-Review Fixture", "domain": "Machine Learning Peer Review", "target_test": "Bayesian VOI Triage & Reviewer Disagreement", "uri": "aetre://datasets/openreview" },
                        { "id": "nih", "name": "Synthetic Biomedical Grant Fixture", "domain": "Biomedical Grants", "target_test": "Horvitz-Thompson Exploration Audits", "uri": "aetre://datasets/nih" },
                        { "id": "uspto", "name": "Synthetic Patent Examination Fixture", "domain": "Utility Patents", "target_test": "Kingman Backlog & Governor Throttling", "uri": "aetre://datasets/uspto" },
                        { "id": "paperswithcode", "name": "Synthetic Code-Artifact Fixture", "domain": "AI Reproducibility", "target_test": "Artifact Routing", "uri": "aetre://datasets/paperswithcode" },
                        { "id": "arxiv_ssrn_live", "name": "External Preprint Adapter (No Bundled Data)", "domain": "User-Supplied Preprints", "target_test": "Optional Ingestion", "uri": "aetre://datasets/arxiv_ssrn_live" }
                    ]
                }),

                "database_connectors" => json!({
                    "section": "Database Integration & Connectors",
                    "schema_uri": "aetre://schemas/database-writeback",
                    "supported_sql_databases": ["PostgreSQL", "SQLite", "Snowflake", "Google BigQuery"],
                    "supported_crm_webhooks": ["Airtable", "Notion", "Affinity CRM", "Typeform"],
                    "batch_pipeline_script": "scripts/connect_external_db.py",
                    "writeback_columns": ["aetre_prior_mean", "aetre_variance", "aetre_novelty", "aetre_voi", "aetre_quality_rank", "aetre_voi_rank", "aetre_routing", "aetre_evaluation_fingerprint"]
                }),

                "tools" => json!({
                    "section": "Mathematical Engine Tools (20 Tools)",
                    "tool_count": 20,
                    "active_license_tier": tier.as_str(),
                    "active_license_resolution": license::resolve_license(&args),
                    "tools": list_tools()
                }),

                "institutional_tiers" => json!({
                    "section": "7 Institutional Deployment Tiers",
                    "tiers_uri": "aetre://institutional/tiers",
                    "tiers": [
                        "Tier 1: Researchers & Authors (Pre-flight diagnostic)",
                        "Tier 2: Venture Capital (Heavy-tailed Pareto VOI)",
                        "Tier 3: Academic Publishers (Multi-agent debiasing & Kingman governor)",
                        "Tier 4: Grant Agencies (5% Horvitz-Thompson audits)",
                        "Tier 5: Patent Offices (Specialist queue balancing)",
                        "Tier 6: Corporate R&D & TTOs (Epistemic portfolio triage)",
                        "Tier 7: Accelerators & Prizes (Anti-sybil quadratic staking)"
                    ]
                }),

                "resources" => json!({
                    "section": "MCP Native Resources",
                    "resources": list_resources()
                }),

                "prompts" => json!({
                    "section": "MCP Standard Prompts",
                    "prompts": list_prompts()
                }),

                _ => json!({
                    "system": "AETRE (Adaptive Epistemic Triage & Recall Engine)",
                    "version": env!("CARGO_PKG_VERSION"),
                    "author": "Clayton Gray (2026)",
                    "paper_reference": "The Innovation-Absorption Gap (SSRN: 7161458)",
                    "protocol": "Model Context Protocol (MCP 2024-11-05)",
                    "active_license_tier": tier.as_str(),
                    "active_license_resolution": license::resolve_license(&args),
                    "license_details": get_quota_status(tier),
                    "total_tools": 20,
                    "total_resources": 4,
                    "total_prompts": 3,
                    "benchmark_datasets": ["openreview", "nih", "uspto", "paperswithcode", "arxiv_ssrn_live"],
                    "database_connectors": ["PostgreSQL", "SQLite", "Snowflake", "BigQuery", "Airtable", "Notion", "Affinity"],
                    "institutional_tiers_supported": 7,
                    "quick_actions": {
                        "read_dataset_catalog": "resources/read with uri='aetre://catalog/datasets'",
                        "read_db_schema": "resources/read with uri='aetre://schemas/database-writeback'",
                        "read_math_specs": "resources/read with uri='aetre://specs/mathematical-formulations'",
                        "read_institutional_tiers": "resources/read with uri='aetre://institutional/tiers'",
                        "read_nih_dataset": "resources/read with uri='aetre://datasets/nih'"
                    }
                }),
            };

            json!({
                "content": [
                    {
                        "type": "text",
                        "text": serde_json::to_string_pretty(&out).unwrap_or_default()
                    }
                ],
                "isError": false
            })
        }

        "aetre_triage_proposal" => {
            let text = get_str(&args, "text", "");
            let title = get_str(&args, "title", "Untitled Proposal");
            let boundary = get_f64(&args, "selection_boundary", 1.2);

            let diagnostics = analyze_text_heuristics(text);
            let prior_mean = diagnostics.prior_mean;
            let prior_var = diagnostics.prior_variance;
            let novelty = diagnostics.novelty_score;

            let voi = calculate_boundary_voi(prior_mean, prior_var, boundary, 0.8, 0.5);

            let (routing, badge_class, rationale) = if voi > 0.15 && prior_var > 0.4 {
                (
                    "HIGH VOI: DEEP HUMAN REVIEW QUEUE",
                    "badge-accent",
                    format!(
                        "High epistemic variance (sigma_0^2 = {:.2}) near selection boundary ({:.2}). High crossing probability justifies human reviewer capacity (VOI = {:.3}).",
                        prior_var, boundary, voi
                    ),
                )
            } else if prior_mean >= boundary {
                (
                    "FAST-PASS: DIRECT PHASE 2",
                    "badge-success",
                    format!(
                        "Expected quality (mu_0 = {:.2}) meets or exceeds boundary ({:.2}). Fast-pass to downstream stage without preliminary review delay.",
                        prior_mean, boundary
                    ),
                )
            } else {
                (
                    "FAST-REJECT / SPAM FILTER",
                    "badge-danger",
                    format!(
                        "Expected quality (mu_0 = {:.2}) falls below selection boundary ({:.2}). Automated filtering preserves reviewer budget.",
                        prior_mean, boundary
                    ),
                )
            };

            let fingerprint_input =
                format!("{title}:{prior_mean:.6}:{prior_var:.6}:{novelty:.6}:{boundary:.6}");
            let hash = format!(
                "aetre-eval-v1-{:x}",
                Sha256::digest(fingerprint_input.as_bytes())
            );

            let md_scorecard = format::format_triage_markdown(
                title, prior_mean, prior_var, novelty, voi, boundary, routing, &rationale, &hash,
            );

            let out = json!({
                "title": title,
                "prior_mean_mu_0": (prior_mean * 1000.0).round() / 1000.0,
                "epistemic_variance_sigma_0_sq": (prior_var * 1000.0).round() / 1000.0,
                "novelty_score": (novelty * 1000.0).round() / 1000.0,
                "crowd_novelty_rank": novelty_rank_label(novelty),
                "crowd_novelty_percentile": novelty_rank_percentile(novelty),
                "crowd_novelty_reference": novelty_rank_reference(),
                "voi_index": (voi * 1000.0).round() / 1000.0,
                "selection_boundary": boundary,
                "predicted_triage_stream": routing,
                "recommended_routing": routing,
                "badge_class": badge_class,
                "decision_rationale": rationale,
                "evaluation_fingerprint": hash,
                "epistemic_diagnostics": diagnostics,
                "markdown_scorecard": md_scorecard
            });

            json!({
                "content": [
                    {
                        "type": "text",
                        "text": serde_json::to_string_pretty(&out).unwrap_or_default()
                    }
                ],
                "isError": false
            })
        }

        "aetre_calculate_voi" => {
            let mu = get_f64(&args, "posterior_mean", 0.0);
            let var = get_f64(&args, "posterior_variance", 1.0);
            let boundary = get_f64(&args, "selection_boundary", 1.2);
            let noise = get_f64(&args, "signal_noise", 0.8);
            let cost = get_f64(&args, "review_cost", 0.5);

            let voi = calculate_boundary_voi(mu, var, boundary, noise, cost);
            let new_var = 1.0 / (1.0 / var + 1.0 / (noise * noise));
            let mean_shift_sd = (var - new_var).max(1e-12).sqrt();
            let gap = (mu - boundary).abs();
            let z = gap / mean_shift_sd;
            let priority = if voi > 0.15 {
                "HIGH_VALUE_REVIEW"
            } else if voi > 0.05 {
                "MODERATE_VALUE_REVIEW"
            } else {
                "LOW_VALUE_REVIEW"
            };

            let md_scorecard = format::format_voi_markdown(
                mu,
                var,
                boundary,
                noise,
                cost,
                mean_shift_sd,
                z,
                voi,
                priority,
            );

            let out = json!({
                "posterior_mean": mu,
                "posterior_variance": var,
                "selection_boundary": boundary,
                "signal_noise": noise,
                "review_cost": cost,
                "mean_shift_std_dev": mean_shift_sd,
                "normalized_boundary_distance_z": z,
                "voi_index": voi,
                "priority_assessment": priority,
                "markdown_scorecard": md_scorecard
            });

            json!({
                "content": [
                    {
                        "type": "text",
                        "text": serde_json::to_string_pretty(&out).unwrap_or_default()
                    }
                ],
                "isError": false
            })
        }

        "aetre_check_governor" => {
            let lambda = get_f64(&args, "arrival_rate", 90.0);
            let mu = get_f64(&args, "service_rate", 100.0);
            let cv_a = get_f64(&args, "cv_arrivals", 1.0);
            let cv_s = get_f64(&args, "cv_service", 1.0);
            let target_rho = get_f64(&args, "target_utilization", 0.85);

            let q_metrics = evaluate_stage_queue(lambda, mu, cv_a, cv_s);
            let gov_action = calculate_governor_action(lambda, mu, target_rho);

            let drop_pct = if q_metrics.utilization > target_rho {
                ((q_metrics.utilization - target_rho) / q_metrics.utilization) * 100.0
            } else {
                0.0
            };
            let action_str = if gov_action.recommend_automated_triage {
                format!("THROTTLE_BOTTOM_{:.1}%", drop_pct)
            } else {
                "NO_ACTION_REQUIRED".to_string()
            };
            let explanation = if q_metrics.utilization >= 0.85 {
                format!("Kingman heavy-traffic queue utilization rho={:.2} exceeds 0.85 ceiling, causing asymptotic delay explosion. Auto-filtering bottom {:.1}% of candidate pool stabilizes reviewer queue.", q_metrics.utilization, drop_pct)
            } else {
                format!("Queue is in stable operating regime (rho={:.2} <= 0.85). Reviewer capacity is well-matched to current arrival volume.", q_metrics.utilization)
            };
            let md_scorecard = format::format_governor_markdown(
                lambda,
                mu,
                q_metrics.utilization,
                q_metrics.mean_wait_time,
                &action_str,
                drop_pct,
                &explanation,
            );

            let out = json!({
                "arrival_rate_lambda": lambda,
                "service_capacity_mu": mu,
                "utilization_rho": q_metrics.utilization,
                "mean_queue_delay": q_metrics.mean_wait_time,
                "in_system_backlog": q_metrics.mean_items_in_queue,
                "is_congested": q_metrics.is_congested,
                "target_utilization": target_rho,
                "excess_arrival_rate": gov_action.excess_arrival_rate,
                "recommend_automated_triage": gov_action.recommend_automated_triage,
                "governor_status": if q_metrics.utilization >= 0.95 {
                    "CRITICAL_SATURATION"
                } else if q_metrics.utilization >= 0.85 {
                    "HEAVY_CONGESTION_WARNING"
                } else {
                    "STABLE_OPERATION"
                },
                "markdown_scorecard": md_scorecard
            });

            json!({
                "content": [
                    {
                        "type": "text",
                        "text": serde_json::to_string_pretty(&out).unwrap_or_default()
                    }
                ],
                "isError": false
            })
        }

        "aetre_exploration_audit" => {
            let n_total = get_usize(&args, "deprioritized_pool_size", 0);
            let n_sample = get_usize(&args, "audited_sample_size", 0);
            let n_found = get_usize(&args, "audited_high_value_found", 0);

            let res = calculate_exploration_audit(n_total, n_sample, n_found);
            let md_scorecard = format::format_exploration_audit_markdown(
                res.deprioritized_pool_size,
                res.audited_sample_size,
                res.audited_high_value_found,
                res.estimated_hidden_high_value,
                res.estimated_hidden_high_value_std_err,
                res.confidence_interval_95.0,
                res.confidence_interval_95.1,
            );

            let out = json!({
                "deprioritized_pool_size_N_D": res.deprioritized_pool_size,
                "audited_sample_size_m_D": res.audited_sample_size,
                "audited_high_value_found": res.audited_high_value_found,
                "estimated_hidden_high_value_H_hat_D": res.estimated_hidden_high_value,
                "std_err": res.estimated_hidden_high_value_std_err,
                "confidence_interval_95": {
                    "lower": res.confidence_interval_95.0,
                    "upper": res.confidence_interval_95.1
                },
                "markdown_scorecard": md_scorecard
            });

            json!({
                "content": [
                    {
                        "type": "text",
                        "text": serde_json::to_string_pretty(&out).unwrap_or_default()
                    }
                ],
                "isError": false
            })
        }

        "aetre_evaluate_staking" => {
            let c_gen = get_f64(&args, "generation_cost", 0.05);
            let c_sub = get_f64(&args, "submission_fee", 5.0);
            let val = get_f64(&args, "private_acceptance_value", 100.0);
            let apps = get_usize(&args, "total_potential_applicants", 5000);
            let cap = get_usize(&args, "acceptance_capacity", 200);

            let eq = evaluate_submitter_equilibrium(c_gen, c_sub, val, apps, cap);
            let md_scorecard = format::format_submitter_equilibrium_markdown(
                eq.generation_cost,
                eq.submission_fee,
                eq.private_acceptance_value,
                eq.total_potential_applicants,
                cap,
                eq.threshold_acceptance_prob,
                eq.estimated_entry_volume,
                eq.low_quality_spam_deterred_pct,
            );

            let out = json!({
                "generation_cost_c_gen": eq.generation_cost,
                "submission_fee_c_sub": eq.submission_fee,
                "private_value_V": eq.private_acceptance_value,
                "total_potential_applicants_N": eq.total_potential_applicants,
                "acceptance_capacity_K": cap,
                "threshold_acceptance_probability": eq.threshold_acceptance_prob,
                "estimated_equilibrium_entry_volume": eq.estimated_entry_volume.round() as usize,
                "low_quality_spam_deterred_pct": (eq.low_quality_spam_deterred_pct * 10.0).round() / 10.0,
                "markdown_scorecard": md_scorecard
            });

            json!({
                "content": [
                    {
                        "type": "text",
                        "text": serde_json::to_string_pretty(&out).unwrap_or_default()
                    }
                ],
                "isError": false
            })
        }

        "aetre_proposition_1_bound" => {
            let n = get_usize(&args, "total_candidates", 5000);
            let k = get_usize(&args, "selection_capacity", 200);
            let p_h = get_f64(&args, "high_value_rate", 0.067);

            let bound = calculate_proposition_1_bound(n, k, p_h);

            let missed = bound.expected_high_value_count * (1.0 - bound.theoretical_max_recall);
            let md_scorecard = format::format_prop1_markdown(
                n,
                k,
                p_h,
                bound.expected_high_value_count,
                bound.theoretical_max_recall,
                missed,
                bound.is_capacity_constrained,
            );

            let out = json!({
                "total_candidates_N": bound.total_candidates,
                "selection_capacity_K": bound.selection_capacity,
                "high_value_rate_p_H": bound.high_value_rate,
                "expected_high_value_count_H_N": bound.expected_high_value_count.round() as usize,
                "theoretical_max_recall_R_N": (bound.theoretical_max_recall * 1000.0).round() / 1000.0,
                "theoretical_max_recall_pct": format!("{:.1}%", bound.theoretical_max_recall * 100.0),
                "is_capacity_constrained": bound.is_capacity_constrained,
                "missed_high_value_candidates": missed.round() as usize,
                "markdown_scorecard": md_scorecard
            });

            json!({
                "content": [
                    {
                        "type": "text",
                        "text": serde_json::to_string_pretty(&out).unwrap_or_default()
                    }
                ],
                "isError": false
            })
        }

        "aetre_correlated_posterior_update" => {
            let prior_mean = get_f64(&args, "prior_mean", 0.0);
            let prior_var = get_f64(&args, "prior_variance", 1.0);
            let rho = get_f64(&args, "inter_agent_correlation", 0.5);

            let mut evaluations = Vec::new();
            if let Some(evals) = args.get("evaluations").and_then(|v| v.as_array()) {
                for (idx, e) in evals.iter().enumerate() {
                    let agent_id =
                        get_str(e, "agent_id", &format!("agent_{}", idx + 1)).to_string();
                    let score = get_f64(e, "score", 0.0);
                    let noise_sd = get_f64(e, "noise_sd", 1.0);
                    evaluations.push(AgentEvaluation {
                        agent_id,
                        score,
                        noise_sd,
                    });
                }
            }

            let res = correlated_posterior_update(prior_mean, prior_var, &evaluations, rho);
            let md_scorecard = format::format_correlated_update_markdown(
                prior_mean,
                prior_var,
                evaluations.len(),
                rho,
                res.effective_evaluator_count,
                res.correlation_discount * 100.0,
                res.posterior_mean,
                res.posterior_variance,
            );

            let out = json!({
                "prior_mean": prior_mean,
                "prior_variance": prior_var,
                "raw_agent_count": evaluations.len(),
                "inter_agent_correlation_rho": rho,
                "effective_evaluator_count_M_eff": (res.effective_evaluator_count * 100.0).round() / 100.0,
                "redundancy_correlation_discount_pct": (res.correlation_discount * 100.0).round() / 100.0,
                "posterior_mean": (res.posterior_mean * 1000.0).round() / 1000.0,
                "posterior_variance": (res.posterior_variance * 1000.0).round() / 1000.0,
                "markdown_scorecard": md_scorecard
            });

            json!({
                "content": [
                    {
                        "type": "text",
                        "text": serde_json::to_string_pretty(&out).unwrap_or_default()
                    }
                ],
                "isError": false
            })
        }

        "aetre_heavy_tailed_voi" => {
            let mu = get_f64(&args, "posterior_mean", 0.0);
            let var = get_f64(&args, "posterior_variance", 1.0);
            let boundary = get_f64(&args, "selection_boundary", 1.2);
            let alpha = get_f64(&args, "tail_index_alpha", 1.5);
            let noise = get_f64(&args, "signal_noise", 0.8);
            let cost = get_f64(&args, "review_cost", 0.5);

            let res = calculate_heavy_tailed_voi(mu, var, boundary, alpha, noise, cost);
            let priority = if res.voi_index > 0.25 {
                "CRITICAL_BREAKTHROUGH_CANDIDATE"
            } else if res.voi_index > 0.10 {
                "HIGH_VALUE_TAIL_REVIEW"
            } else {
                "STANDARD_REVIEW"
            };

            let md_scorecard = format::format_heavy_tailed_voi_markdown(
                mu,
                var,
                boundary,
                res.tail_index,
                res.tail_probability,
                res.expected_excess_payoff,
                res.voi_index,
                priority,
            );

            let out = json!({
                "posterior_mean": mu,
                "posterior_variance": var,
                "selection_boundary": boundary,
                "pareto_tail_index_alpha": res.tail_index,
                "tail_crossing_probability": (res.tail_probability * 10000.0).round() / 10000.0,
                "expected_excess_breakthrough_payoff": (res.expected_excess_payoff * 100.0).round() / 100.0,
                "heavy_tail_voi_index": (res.voi_index * 1000.0).round() / 1000.0,
                "priority_assessment": priority,
                "markdown_scorecard": md_scorecard
            });

            json!({
                "content": [
                    {
                        "type": "text",
                        "text": serde_json::to_string_pretty(&out).unwrap_or_default()
                    }
                ],
                "isError": false
            })
        }

        "aetre_quadratic_staking" => {
            let base_fee = get_f64(&args, "base_fee", 5.0);
            let gamma = get_f64(&args, "escalation_exponent", 2.0);
            let count = get_usize(&args, "submission_count", 1);
            let c_gen = get_f64(&args, "generation_cost", 0.05);
            let val = get_f64(&args, "private_acceptance_value", 100.0);

            let res = evaluate_quadratic_staking(base_fee, gamma, count, c_gen, val);

            let md_scorecard = format::format_staking_markdown(
                res.base_fee,
                res.submission_count,
                res.total_stake_required,
                res.marginal_stake_for_next,
                res.spam_deterrence_pct > 80.0,
            );

            let out = json!({
                "base_fee_S_0": res.base_fee,
                "escalation_exponent_gamma": res.escalation_exponent,
                "submission_count_m": res.submission_count,
                "total_stake_required": (res.total_stake_required * 100.0).round() / 100.0,
                "marginal_stake_for_next": (res.marginal_stake_for_next * 100.0).round() / 100.0,
                "spam_deterrence_pct": (res.spam_deterrence_pct * 10.0).round() / 10.0,
                "markdown_scorecard": md_scorecard
            });

            json!({
                "content": [
                    {
                        "type": "text",
                        "text": serde_json::to_string_pretty(&out).unwrap_or_default()
                    }
                ],
                "isError": false
            })
        }

        "aetre_heterogeneous_queues" => {
            let mut pools_input = Vec::new();
            if let Some(pools) = args.get("pools").and_then(|v| v.as_array()) {
                for p in pools {
                    let domain = get_str(p, "domain", "General").to_string();
                    let arrival_rate = get_f64(p, "arrival_rate", 10.0);
                    let service_rate = get_f64(p, "service_rate", 15.0);
                    let cv_a = get_f64(p, "cv_arrivals", 1.0);
                    let cv_s = get_f64(p, "cv_service", 1.0);
                    pools_input.push((domain, arrival_rate, service_rate, cv_a, cv_s));
                }
            }

            let res = evaluate_heterogeneous_queues(pools_input);
            let md_scorecard = format::format_heterogeneous_queues_markdown(
                res.pools.len(),
                res.max_utilization,
                &res.bottleneck_domain,
                res.is_system_congested,
                &res.rebalancing_actions,
                &res.pools,
            );

            let out = json!({
                "pool_count": res.pools.len(),
                "max_utilization": (res.max_utilization * 100.0).round() / 100.0,
                "bottleneck_domain": res.bottleneck_domain,
                "is_system_congested": res.is_system_congested,
                "rebalancing_actions": res.rebalancing_actions,
                "pool_details": res.pools,
                "markdown_scorecard": md_scorecard
            });

            json!({
                "content": [
                    {
                        "type": "text",
                        "text": serde_json::to_string_pretty(&out).unwrap_or_default()
                    }
                ],
                "isError": false
            })
        }

        "aetre_author_preflight_benchmark" => {
            let text = get_str(&args, "text", "");
            let title = get_str(&args, "title", "Untitled Proposal");
            let boundary = get_f64(&args, "selection_boundary", 1.2);

            let diagnostics = analyze_text_heuristics(text);
            let prior_mean = diagnostics.prior_mean;
            let prior_var = diagnostics.prior_variance;
            let novelty = diagnostics.novelty_score;

            let report = evaluate_author_preflight(title, prior_mean, prior_var, novelty, boundary);

            let action_plan_str = report.prescriptive_action_plan.join("\n");
            let md_scorecard = format::format_triage_markdown(
                title,
                report.prior_mean,
                report.epistemic_variance,
                report.novelty_score,
                report.voi_index,
                boundary,
                &report.predicted_triage_stream,
                &action_plan_str,
                &report.evaluation_fingerprint,
            );

            let out = json!({
                "title": report.title,
                "prior_mean_mu_0": (report.prior_mean * 1000.0).round() / 1000.0,
                "epistemic_variance_sigma_0_sq": (report.epistemic_variance * 1000.0).round() / 1000.0,
                "novelty_score": (report.novelty_score * 1000.0).round() / 1000.0,
                "crowd_novelty_rank": report.crowd_novelty_rank,
                "crowd_novelty_percentile": report.crowd_novelty_percentile,
                "crowd_novelty_reference": report.crowd_novelty_reference,
                "reviewer_disagreement_risk": report.reviewer_disagreement_risk,
                "predicted_triage_stream": report.predicted_triage_stream,
                "voi_index": report.voi_index,
                "prescriptive_action_plan": report.prescriptive_action_plan,
                "variance_reduction_target": report.variance_reduction_target,
                "evaluation_fingerprint": report.evaluation_fingerprint,
                "markdown_badge": report.markdown_badge,
                "epistemic_diagnostics": diagnostics,
                "markdown_scorecard": md_scorecard,
                "license_tier": tier.as_str()
            });

            json!({
                "content": [
                    {
                        "type": "text",
                        "text": serde_json::to_string_pretty(&out).unwrap_or_default()
                    }
                ],
                "isError": false
            })
        }

        "aetre_simulate_benchmark" => {
            let replications = get_usize(&args, "replications", 50);
            let baseline_arrivals = get_usize(&args, "baseline_arrivals", 1000);
            let ai_multiplier = get_f64(&args, "ai_arrival_multiplier", 5.0);
            let capacity = get_usize(&args, "acceptance_capacity", 200);
            let unconventional_share = get_f64(&args, "unconventional_share", 0.10);
            let budget = get_f64(&args, "evaluation_budget", 1000.0);
            let audit_share = get_f64(&args, "randomized_audit_budget_share", 0.05);

            let params = Parameters {
                baseline_arrivals,
                ai_arrival_multiplier: ai_multiplier,
                acceptance_capacity: capacity,
                unconventional_share,
                evaluation_budget: budget,
                randomized_audit_budget_share: audit_share,
                ..Default::default()
            };

            let mut rng = rand::thread_rng();
            let regimes = run_benchmark_replications(&mut rng, replications, &params);
            let md_scorecard = format::format_benchmark_simulation_markdown(replications, &regimes);

            let out = json!({
                "replications": replications,
                "simulation_parameters": {
                    "baseline_arrivals": baseline_arrivals,
                    "ai_arrival_multiplier": ai_multiplier,
                    "acceptance_capacity": capacity,
                    "unconventional_share": unconventional_share,
                    "evaluation_budget": budget,
                    "randomized_audit_budget_share": audit_share
                },
                "regime_results": regimes,
                "markdown_scorecard": md_scorecard
            });

            json!({
                "content": [
                    {
                        "type": "text",
                        "text": serde_json::to_string_pretty(&out).unwrap_or_default()
                    }
                ],
                "isError": false
            })
        }

        "aetre_batch_triage" => {
            let boundary = get_f64(&args, "selection_boundary", 1.2);
            let mut proposals_out = Vec::new();

            if let Some(items) = args.get("proposals").and_then(|v| v.as_array()) {
                // A ceiling on work per call, not a licence check: the same for
                // everyone, and reported below when it truncates.
                const MAX_BATCH_ITEMS: usize = 5000;
                let items_to_process = &items[..items.len().min(MAX_BATCH_ITEMS)];

                for (idx, item) in items_to_process.iter().enumerate() {
                    let title =
                        get_str(item, "title", &format!("Proposal_{}", idx + 1)).to_string();
                    let text = get_str(item, "text", "");
                    let diag = analyze_text_heuristics(text);
                    let voi = calculate_boundary_voi(
                        diag.prior_mean,
                        diag.prior_variance,
                        boundary,
                        0.8,
                        0.5,
                    );

                    let stream = if voi > 0.15 && diag.prior_variance > 0.4 {
                        "HIGH VOI: DEEP HUMAN REVIEW QUEUE"
                    } else if diag.prior_mean >= boundary {
                        "FAST-PASS: DIRECT PHASE 2"
                    } else {
                        "FAST-REJECT / SPAM FILTER"
                    };

                    proposals_out.push((
                        title,
                        diag.prior_mean,
                        diag.prior_variance,
                        diag.novelty_score,
                        voi,
                        stream.to_string(),
                    ));
                }
            }

            let total_count = proposals_out.len();
            let fast_pass_count = proposals_out
                .iter()
                .filter(|p| p.5.contains("FAST-PASS"))
                .count();
            let deep_review_count = proposals_out
                .iter()
                .filter(|p| p.5.contains("HIGH VOI") || p.5.contains("DEEP"))
                .count();
            let fast_reject_count = proposals_out
                .iter()
                .filter(|p| p.5.contains("FAST-REJECT"))
                .count();

            // Sort by quality mean descending for ranking
            let mut ranked_items = proposals_out.clone();
            ranked_items.sort_unstable_by(|a, b| {
                b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
            });

            let top_items_for_md: Vec<(usize, &str, f64, f64, f64, &str)> = ranked_items
                .iter()
                .enumerate()
                .take(15)
                .map(|(rank, p)| (rank + 1, p.0.as_str(), p.1, p.2, p.4, p.5.as_str()))
                .collect();

            let md_scorecard = format::format_batch_triage_markdown(
                total_count,
                boundary,
                fast_pass_count,
                deep_review_count,
                fast_reject_count,
                &top_items_for_md,
            );

            let out_list: Vec<Value> = ranked_items
                .into_iter()
                .enumerate()
                .map(|(rank, p)| {
                    json!({
                        "global_quality_rank": rank + 1,
                        "title": p.0,
                        "prior_mean_mu_0": (p.1 * 1000.0).round() / 1000.0,
                        "epistemic_variance_sigma_0_sq": (p.2 * 1000.0).round() / 1000.0,
                        "novelty_score": (p.3 * 1000.0).round() / 1000.0,
                        "voi_index": (p.4 * 1000.0).round() / 1000.0,
                        "triage_stream": p.5
                    })
                })
                .collect();

            let out = json!({
                "total_proposals_evaluated": total_count,
                "selection_boundary": boundary,
                "cohort_allocation": {
                    "fast_pass_direct_count": fast_pass_count,
                    "high_voi_deep_review_count": deep_review_count,
                    "fast_reject_count": fast_reject_count
                },
                "ranked_proposals": out_list,
                "markdown_scorecard": md_scorecard
            });

            json!({
                "content": [
                    {
                        "type": "text",
                        "text": serde_json::to_string_pretty(&out).unwrap_or_default()
                    }
                ],
                "isError": false
            })
        }

        "aetre_recall_scaling_curve" => {
            let n = get_usize(&args, "baseline_arrivals", 1000);
            let k = get_usize(&args, "selection_capacity", 200);
            let p_h = get_f64(&args, "high_value_rate", 0.067);

            let multipliers: Vec<f64> =
                if let Some(arr) = args.get("multipliers").and_then(|v| v.as_array()) {
                    arr.iter()
                        .filter_map(|v| {
                            v.as_f64()
                                .or_else(|| v.as_str().and_then(|s| s.parse().ok()))
                        })
                        .collect()
                } else {
                    vec![1.0, 2.0, 5.0, 10.0, 20.0, 50.0]
                };

            let curve = generate_recall_scaling_curve(n, k, p_h, &multipliers);

            let out = json!({
                "baseline_arrivals_N": n,
                "selection_capacity_K": k,
                "high_value_rate_p_H": p_h,
                "scaling_curve_points": curve
            });

            json!({
                "content": [
                    {
                        "type": "text",
                        "text": serde_json::to_string_pretty(&out).unwrap_or_default()
                    }
                ],
                "isError": false
            })
        }

        "aetre_fit_boundary" => {
            let split = get_str(&args, "split", "calib");
            let budget = get_usize(&args, "budget", 200);
            let min = get_f64(&args, "grid_min", 1.0);
            let max = get_f64(&args, "grid_max", 10.0);
            let step = get_f64(&args, "grid_step", 0.25).max(0.01);

            let Some(name) = args.get("dataset").and_then(|v| v.as_str()) else {
                return json!({
                    "content": [{ "type": "text", "text":
                        "dataset is required: fit the boundary against your own calibration data,                          not against a sample." }],
                    "isError": true
                });
            };
            let path = std::path::PathBuf::from(name);
            if !path.exists() {
                return json!({
                    "content": [{ "type": "text", "text": format!("Dataset not found: {name}.") }],
                    "isError": true
                });
            }
            let Ok(raw) = std::fs::read(&path) else {
                return json!({
                    "content": [{ "type": "text", "text": format!("Could not read {name}.") }],
                    "isError": true
                });
            };
            let Ok(records) = serde_json::from_slice::<Vec<BacktestCandidate>>(&raw) else {
                return json!({
                    "content": [{ "type": "text", "text": format!("Could not parse {name} as a candidate array.") }],
                    "isError": true
                });
            };

            let mut curve = Vec::new();
            let mut best: Option<(f64, f64, f64)> = None;
            let mut boundary = min;
            while boundary <= max + 1e-9 {
                let s = score_backtest(&records, split, budget, boundary);
                curve.push(json!({
                    "boundary": (boundary * 1000.0).round() / 1000.0,
                    "precision": (s.precision * 1000.0).round() / 10.0,
                    "recall": (s.recall * 1000.0).round() / 10.0,
                }));
                // A match rather than is_none_or: that method landed in 1.82 and the
                // container builds on 1.80. map_or(true, ..) would trip clippy on
                // newer toolchains, so neither helper is safe across both.
                let improves = match best {
                    None => true,
                    Some((_, best_precision, _)) => s.precision > best_precision,
                };
                if improves {
                    best = Some((boundary, s.precision, s.recall));
                }
                boundary += step;
            }

            let Some((fitted, precision, recall)) = best else {
                return json!({
                    "content": [{ "type": "text", "text": "The grid was empty: check grid_min, grid_max and grid_step." }],
                    "isError": true
                });
            };

            let reference = score_backtest(&records, split, budget, 6.0);
            let base_rate = reference.positives as f64 / reference.evaluated.max(1) as f64;
            let at_edge = (fitted - min).abs() < 1e-9 || (fitted - max).abs() < 1e-9;

            let out = json!({
                "fitted_boundary": (fitted * 1000.0).round() / 1000.0,
                "fitted_on_split": split,
                "candidates": reference.evaluated,
                "positives": reference.positives,
                "base_rate_pct": (base_rate * 1000.0).round() / 10.0,
                "precision_at_k_pct": (precision * 1000.0).round() / 10.0,
                "recall_at_k_pct": (recall * 1000.0).round() / 10.0,
                "lift_over_random": if base_rate > 0.0 {
                    (precision / base_rate * 100.0).round() / 100.0
                } else { 0.0 },
                "budget_k": budget,
                "grid": { "min": min, "max": max, "step": step },
                "optimum_at_grid_edge": at_edge,
                "curve": curve,
                "caution": "Fitted on this split. Evaluate on a split you did not fit on before                             believing the number, and refit per venue: the optimum is sharp.",
                "dataset_source": path.display().to_string(),
            });

            json!({
                "content": [{ "type": "text", "text": serde_json::to_string_pretty(&out).unwrap_or_default() }],
                "isError": false
            })
        }

        "aetre_heldout_backtest" => {
            let budget = get_usize(&args, "budget", 50);
            let boundary = get_f64(&args, "boundary", 6.0);
            let boundary_source = if args.get("boundary").is_some() {
                "supplied by the caller".to_string()
            } else {
                "default 6.0, not fitted to this corpus - run aetre_fit_boundary".to_string()
            };
            let split = get_str(&args, "split", "test");
            // A caller who names a dataset gets that dataset or an error. Falling
            // back to the bundled sample would answer with six rows of fixture
            // data while looking like a result for the file they asked for.
            let supplied = args.get("dataset").and_then(|v| v.as_str());
            if let Some(name) = supplied {
                let path = std::path::PathBuf::from(name);
                if !path.exists() {
                    return json!({
                        "content": [{ "type": "text", "text": format!(
                            "Dataset not found: {name}. Pass a path to a JSON array of                              candidates, or omit `dataset` to run against the bundled                              six-candidate sample."
                        ) }],
                        "isError": true
                    });
                }
            }

            let candidates_files = [
                supplied.unwrap_or(""),
                "examples/datasets/openreview_heldout_backtest.json",
                "../../examples/datasets/openreview_heldout_backtest.json",
                "data/normalized/openreview_normalized.json",
            ];
            let mut file_path = None;
            for c in candidates_files {
                if c.is_empty() {
                    continue;
                }
                let p = std::path::PathBuf::from(c);
                if p.exists() {
                    file_path = Some(p);
                    break;
                }
            }

            // A real file on disk wins; otherwise fall back to the embedded
            // fixture so the tool still answers, and name the source either way.
            let (raw, dataset_source) = match &file_path {
                Some(p) => match std::fs::read(p) {
                    Ok(b) => (b, p.display().to_string()),
                    Err(e) => {
                        return json!({
                            "content": [{ "type": "text", "text": format!("Error reading dataset: {}", e) }],
                            "isError": true
                        });
                    }
                },
                None => (
                    BUNDLED_BACKTEST.as_bytes().to_vec(),
                    "bundled fixture (6 candidates, illustrative, not evidence)".to_string(),
                ),
            };

            {
                let Ok(records) = serde_json::from_slice::<Vec<BacktestCandidate>>(&raw) else {
                    return json!({
                        "content": [{ "type": "text", "text": format!(
                            "Failed to parse backtest dataset JSON schema from {}", dataset_source
                        ) }],
                        "isError": true
                    });
                };

                // The same scorer the boundary fit uses, so a fitted threshold is
                // evaluated by exactly the procedure that chose it.
                let scored = score_backtest(&records, split, budget, boundary);

                let out = json!({
                    "dataset_source": dataset_source,
                    "boundary": boundary,
                    "boundary_source": boundary_source,
                    "evaluation_split": split,
                    "candidates_evaluated": scored.evaluated,
                    "true_decision_flips": scored.positives,
                    "budget_allocated_K": scored.budget,
                    "aetre_voi_precision_at_k": (scored.precision * 1000.0).round() / 10.0,
                    "aetre_voi_recall_at_k": (scored.recall * 1000.0).round() / 10.0,
                    "aetre_discoveries_caught": scored.caught,
                    "reviewer_hours_per_discovery": if scored.caught > 0 {
                        (scored.budget as f64 * 4.0) / scored.caught as f64
                    } else {
                        scored.budget as f64 * 4.0
                    },
                    "status": "BACKTEST_EVALUATED_SUCCESSFULLY"
                });

                json!({
                    "content": [{ "type": "text", "text": serde_json::to_string_pretty(&out).unwrap_or_default() }],
                    "isError": false
                })
            }
        }

        "aetre_calibrate_scorer" => {
            let scores: Vec<f64> = args
                .get("scores")
                .and_then(|v| v.as_array())
                .map(|arr| arr.iter().filter_map(|x| x.as_f64()).collect())
                .unwrap_or_default();
            let labels: Vec<u8> = args
                .get("labels")
                .and_then(|v| v.as_array())
                .map(|arr| {
                    arr.iter()
                        .filter_map(|x| x.as_u64().map(|n| n as u8))
                        .collect()
                })
                .unwrap_or_default();

            if scores.is_empty() || labels.is_empty() || scores.len() != labels.len() {
                return json!({
                    "content": [{ "type": "text", "text": "Invalid scores/labels array: lengths must be non-empty and equal." }],
                    "isError": true
                });
            }

            let iterations = get_usize(&args, "iterations", 500);
            let lr = get_f64(&args, "learning_rate", 0.05);

            let calibrator = aetre_core::PlattCalibrator::fit(&scores, &labels, iterations, lr);
            let probs: Vec<f64> = scores
                .iter()
                .map(|&s| calibrator.predict_probability(s))
                .collect();
            let ece = aetre_core::calculate_expected_calibration_error(&probs, &labels, 10);
            let brier = aetre_core::calculate_brier_score(&probs, &labels);

            let out = json!({
                "calibration_method": "Platt_Logistic_Scaling",
                "training_samples_count": scores.len(),
                "calibrator_slope": (calibrator.slope * 10000.0).round() / 10000.0,
                "calibrator_intercept": (calibrator.intercept * 10000.0).round() / 10000.0,
                "expected_calibration_error_10_bin": (ece * 10000.0).round() / 10000.0,
                "brier_score": (brier * 10000.0).round() / 10000.0,
                "status": "CALIBRATOR_FITTED_SUCCESSFULLY"
            });

            json!({
                "content": [{ "type": "text", "text": serde_json::to_string_pretty(&out).unwrap_or_default() }],
                "isError": false
            })
        }

        "aetre_multi_attribute_voi" => {
            let dimensions: Vec<MultiAttributeDimension> =
                if let Some(arr) = args.get("dimensions").and_then(|v| v.as_array()) {
                    arr.iter()
                        .map(|d| {
                            let name = get_str(d, "name", "Unnamed Dimension").to_string();
                            let prior_mean = get_f64(d, "prior_mean", 5.0);
                            let prior_variance = get_f64(d, "prior_variance", 1.0);
                            let weight = get_f64(d, "weight", 1.0);
                            let threshold = d.get("threshold").and_then(|v| v.as_f64());
                            let review_noise_sd = get_f64(d, "review_noise_sd", 0.8);
                            MultiAttributeDimension {
                                name,
                                prior_mean,
                                prior_variance,
                                weight,
                                threshold,
                                review_noise_sd,
                            }
                        })
                        .collect()
                } else {
                    Vec::new()
                };

            if dimensions.is_empty() {
                return json!({
                    "content": [{ "type": "text", "text": "Error: 'dimensions' array must not be empty." }],
                    "isError": true
                });
            }

            let threshold = get_f64(&args, "composite_threshold", 6.0);
            let cost_per_dim = get_f64(&args, "review_cost_per_dim", 1.0);

            let result = evaluate_multi_attribute_voi(&dimensions, threshold, cost_per_dim);

            let out = json!({
                "composite_prior_mean": (result.composite_prior_mean * 1000.0).round() / 1000.0,
                "composite_prior_variance": (result.composite_prior_variance * 1000.0).round() / 1000.0,
                "composite_threshold": result.composite_threshold,
                "total_composite_voi": (result.composite_voi * 10000.0).round() / 10000.0,
                "suggested_routing": result.suggested_routing,
                "recommended_review_dimensions": result.recommended_review_dimensions,
                "dimension_breakdown": result.dimension_contributions.iter().map(|c| {
                    json!({
                        "dimension": c.dimension,
                        "weight": (c.weight * 1000.0).round() / 1000.0,
                        "marginal_voi": (c.marginal_voi * 10000.0).round() / 10000.0,
                        "variance_share_pct": (c.variance_share * 1000.0).round() / 10.0
                    })
                }).collect::<Vec<_>>()
            });

            json!({
                "content": [{ "type": "text", "text": serde_json::to_string_pretty(&out).unwrap_or_default() }],
                "isError": false
            })
        }

        "aetre_congestion_matching" => {
            let proposals: Vec<ProposalRequirement> = if let Some(arr) =
                args.get("proposals").and_then(|v| v.as_array())
            {
                arr.iter()
                    .filter_map(|p| {
                        let id = get_str(p, "id", "").to_string();
                        if id.is_empty() {
                            return None;
                        }
                        let title = get_str(p, "title", "Untitled Proposal").to_string();
                        let domain = get_str(p, "domain", "General").to_string();
                        let voi_index = get_f64(p, "voi_index", 0.5);
                        let required_reviews = get_usize(p, "required_reviews", 2);
                        let keywords = p
                            .get("keywords")
                            .and_then(|k| k.as_array())
                            .map(|arr| {
                                arr.iter()
                                    .filter_map(|s| s.as_str().map(|str_val| str_val.to_string()))
                                    .collect()
                            })
                            .unwrap_or_default();

                        Some(ProposalRequirement {
                            id,
                            title,
                            domain,
                            voi_index,
                            required_reviews,
                            keywords,
                        })
                    })
                    .collect()
            } else {
                Vec::new()
            };

            let reviewers: Vec<ReviewerProfile> = if let Some(arr) =
                args.get("reviewers").and_then(|v| v.as_array())
            {
                arr.iter()
                    .filter_map(|r| {
                        let id = get_str(r, "id", "").to_string();
                        if id.is_empty() {
                            return None;
                        }
                        let name = get_str(r, "name", "Anonymous Reviewer").to_string();
                        let domain = get_str(r, "domain", "General").to_string();
                        let capacity = get_usize(r, "capacity", 3);
                        let current_load = get_usize(r, "current_load", 0);
                        let service_rate = get_f64(r, "service_rate", 10.0);
                        let arrival_rate = get_f64(r, "arrival_rate", 5.0);
                        let expertise_tags = r
                            .get("expertise_tags")
                            .and_then(|k| k.as_array())
                            .map(|arr| {
                                arr.iter()
                                    .filter_map(|s| s.as_str().map(|str_val| str_val.to_string()))
                                    .collect()
                            })
                            .unwrap_or_default();

                        Some(ReviewerProfile {
                            id,
                            name,
                            domain,
                            capacity,
                            current_load,
                            service_rate,
                            arrival_rate,
                            expertise_tags,
                        })
                    })
                    .collect()
            } else {
                Vec::new()
            };

            if proposals.is_empty() || reviewers.is_empty() {
                return json!({
                    "content": [{ "type": "text", "text": "Error: 'proposals' and 'reviewers' arrays must not be empty." }],
                    "isError": true
                });
            }

            let target_utilization = get_f64(&args, "target_utilization", 0.85);
            let result =
                optimize_congestion_matching(&proposals, &reviewers, Some(target_utilization));

            let out = json!({
                "total_proposals": proposals.len(),
                "total_reviewers": reviewers.len(),
                "total_assignments_made": result.assignments.len(),
                "unassigned_proposals_count": result.unassigned_proposals.len(),
                "unassigned_proposal_ids": result.unassigned_proposals,
                "global_affinity_score": (result.global_objective_score * 100.0).round() / 100.0,
                "bottleneck_warnings": result.bottleneck_warnings,
                "assignments": result.assignments,
                "reviewer_utilizations": result.reviewer_utilizations
            });

            json!({
                "content": [{ "type": "text", "text": serde_json::to_string_pretty(&out).unwrap_or_default() }],
                "isError": false
            })
        }

        "aetre_sequential_stopping_rule" => {
            let prior_mean = get_f64(&args, "prior_mean", 5.0);
            let prior_variance = get_f64(&args, "prior_variance", 1.0);
            let threshold = get_f64(&args, "threshold", 6.0);

            let reviews: Vec<SequentialReviewStep> =
                if let Some(arr) = args.get("reviews").and_then(|v| v.as_array()) {
                    arr.iter()
                        .enumerate()
                        .map(|(idx, r)| {
                            let step = get_usize(r, "step", idx + 1);
                            let reviewer_id =
                                get_str(r, "reviewer_id", &format!("rev_{}", idx + 1)).to_string();
                            let score = get_f64(r, "score", 5.0);
                            let noise_sd = get_f64(r, "noise_sd", 0.8);
                            let cost = get_f64(r, "cost", 1.0);
                            SequentialReviewStep {
                                step,
                                reviewer_id,
                                score,
                                noise_sd,
                                cost,
                            }
                        })
                        .collect()
                } else {
                    Vec::new()
                };

            let next_noise = args.get("next_review_noise_sd").and_then(|v| v.as_f64());
            let next_cost = args.get("next_review_cost").and_then(|v| v.as_f64());
            let conf_thresh = args.get("confidence_threshold").and_then(|v| v.as_f64());

            let result = evaluate_sequential_stopping(
                prior_mean,
                prior_variance,
                threshold,
                &reviews,
                next_noise,
                next_cost,
                conf_thresh,
            );

            let out = json!({
                "completed_reviews_count": result.current_step,
                "posterior_mean": (result.posterior_mean * 1000.0).round() / 1000.0,
                "posterior_variance": (result.posterior_variance * 1000.0).round() / 1000.0,
                "posterior_std_dev": (result.posterior_variance.sqrt() * 1000.0).round() / 1000.0,
                "decision": result.decision,
                "decision_confidence_pct": (result.decision_confidence * 1000.0).round() / 10.0,
                "boundary_distance": (result.boundary_distance * 1000.0).round() / 1000.0,
                "prospective_voi_of_next_review": (result.current_voi * 10000.0).round() / 10000.0,
                "total_accumulated_cost": result.total_accumulated_cost,
                "stopping_rationale": result.stopping_rationale
            });

            json!({
                "content": [{ "type": "text", "text": serde_json::to_string_pretty(&out).unwrap_or_default() }],
                "isError": false
            })
        }

        "governed_bellman_triage" => {
            let reward = get_f64(&args, "reward", 0.02);
            let loss = get_f64(&args, "loss", 0.10);
            let prior = get_f64(&args, "prior", 0.50);
            let stage = get_usize(&args, "stage", 0);
            let passes = get_usize(&args, "consecutive_passes", 0);
            let max_stages = get_usize(&args, "max_stages", 4);
            let defect_leakage = get_f64(&args, "defect_leakage", 0.5875);

            let gov = match Governor::new(
                reward,
                loss,
                prior,
                max_stages,
                None,
                Some(defect_leakage),
                Some(1.0),
            ) {
                Ok(g) => g,
                Err(e) => {
                    return json!({
                        "content": [{ "type": "text", "text": format!("Governor error: {}", e) }],
                        "isError": true
                    });
                }
            };

            let decision = gov.evaluate_state(stage, passes);
            let out = json!({
                "action": decision.action,
                "current_stage": decision.stage,
                "consecutive_passes": passes,
                "posterior_belief": (decision.belief * 10000.0).round() / 100.0,
                "critical_threshold_p_star": (gov.p_star * 10000.0).round() / 100.0,
                "expected_utility": (decision.expected_utility * 100000.0).round() / 100000.0,
                "value_of_information_voi": (decision.voi * 100000.0).round() / 100000.0,
                "loss_to_reward_ratio": (loss / reward * 10.0).round() / 10.0,
                "governance_recommendation": match decision.action.as_str() {
                    "CONTINUE" => "PROCEED_TO_NEXT_VERIFICATION_PROBE: Value of Information justifies testing costs.",
                    "HALT_AND_COMMIT" => "ADMIT_AND_COMMIT: Posterior belief exceeds critical threshold p*.",
                    _ => "HALT_AND_REJECT: Candidate fails economic stopping threshold."
                }
            });

            json!({
                "content": [{ "type": "text", "text": serde_json::to_string_pretty(&out).unwrap_or_default() }],
                "isError": false
            })
        }

        "governed_review_boundary" => {
            let belief = get_f64(&args, "belief", 0.70);
            let reward = get_f64(&args, "reward", 0.02);
            let loss = get_f64(&args, "loss", 0.10);
            let review_cost = get_f64(&args, "review_cost", 0.002);
            let shadow_price = get_f64(&args, "shadow_price_lambda", 0.0);
            let accuracy = get_f64(&args, "review_accuracy", 1.0);

            let gov = match Governor::new(reward, loss, 0.50, 4, None, None, None) {
                Ok(g) => g,
                Err(e) => {
                    return json!({
                        "content": [{ "type": "text", "text": format!("Governor error: {}", e) }],
                        "isError": true
                    });
                }
            };

            let res = match gov.evaluate_review_boundary(
                belief,
                review_cost,
                shadow_price,
                accuracy,
            ) {
                Ok(r) => r,
                Err(e) => {
                    return json!({
                        "content": [{ "type": "text", "text": format!("Boundary evaluation error: {}", e) }],
                        "isError": true
                    });
                }
            };

            let out = json!({
                "boundary_action": res.action,
                "dominant_expected_utility": (res.dominant_utility * 100000.0).round() / 100000.0,
                "net_utility_auto": (res.u_auto * 100000.0).round() / 100000.0,
                "net_utility_review": (res.u_review * 100000.0).round() / 100000.0,
                "net_utility_abstain": res.u_abstain,
                "posterior_belief": (res.belief * 10000.0).round() / 100.0,
                "shadow_price_lambda_k": res.shadow_price_lambda,
                "tripartite_rationale": match res.action.as_str() {
                    "AUTO" => "Autonomous admission is optimal (E[U(auto)] dominates review & abstain).",
                    "REVIEW" => "Human review is economically viable (E[U(review)] > 0 and queue is unclogged).",
                    _ => "Abstain: Review costs or queue congestion (lambda_K) render human triage welfare-negative."
                }
            });

            json!({
                "content": [{ "type": "text", "text": serde_json::to_string_pretty(&out).unwrap_or_default() }],
                "isError": false
            })
        }

        "governed_knapsack_admit" => {
            let capacity_k = get_f64(&args, "capacity_k", 2.0);
            let review_cost_k = get_f64(&args, "review_cost_k", 1.0);

            let candidates_raw = args.get("candidates").and_then(|v| v.as_array());
            let candidates: Vec<CandidateSubmission> = match candidates_raw {
                Some(arr) => arr
                    .iter()
                    .enumerate()
                    .map(|(idx, c)| {
                        let cid =
                            get_str(c, "candidate_id", &format!("cand_{}", idx + 1)).to_string();
                        let tid = get_str(c, "task_id", "default_task").to_string();
                        let p = get_f64(c, "posterior_belief", 0.90);
                        let r = c.get("reward").and_then(|v| v.as_f64());
                        let l = c.get("loss").and_then(|v| v.as_f64());
                        let cost = c.get("review_cost").and_then(|v| v.as_f64());
                        CandidateSubmission::new(cid, tid, p, r, l, cost)
                    })
                    .collect(),
                None => Vec::new(),
            };

            let controller = match KnapsackController::new(capacity_k, review_cost_k) {
                Ok(ctrl) => ctrl,
                Err(e) => {
                    return json!({
                        "content": [{ "type": "text", "text": format!("Knapsack controller error: {}", e) }],
                        "isError": true
                    });
                }
            };

            let report = match controller.admit_batch(
                &candidates,
                Some(capacity_k),
                Some(review_cost_k),
            ) {
                Ok(rep) => rep,
                Err(e) => {
                    return json!({
                        "content": [{ "type": "text", "text": format!("Admission error: {}", e) }],
                        "isError": true
                    });
                }
            };

            let out = json!({
                "capacity_k": report.capacity_k,
                "review_cost_k": report.review_cost_k,
                "total_candidates": report.total_candidates,
                "total_admitted": report.total_admitted,
                "total_rejected": report.total_rejected,
                "total_admitted_cost": (report.total_admitted_cost * 1000.0).round() / 1000.0,
                "remaining_capacity": (report.remaining_capacity * 1000.0).round() / 1000.0,
                "total_welfare": (report.total_welfare * 10000.0).round() / 10000.0,
                "capacity_shadow_price_lambda": (report.shadow_price_lambda * 100000.0).round() / 100000.0,
                "admitted": report.admitted.iter().map(|c| json!({
                    "candidate_id": c.candidate_id,
                    "posterior_belief": c.posterior_belief,
                    "expected_utility": (c.expected_utility() * 10000.0).round() / 10000.0,
                    "review_cost": c.review_cost.unwrap_or(report.review_cost_k),
                    "value_density_rho": (c.density_with_cost(report.review_cost_k) * 10000.0).round() / 10000.0
                })).collect::<Vec<_>>(),
                "rejected": report.rejected.iter().map(|c| json!({
                    "candidate_id": c.candidate_id,
                    "posterior_belief": c.posterior_belief,
                    "expected_utility": (c.expected_utility() * 10000.0).round() / 10000.0,
                    "review_cost": c.review_cost.unwrap_or(report.review_cost_k),
                    "value_density_rho": (c.density_with_cost(report.review_cost_k) * 10000.0).round() / 10000.0
                })).collect::<Vec<_>>()
            });

            json!({
                "content": [{ "type": "text", "text": serde_json::to_string_pretty(&out).unwrap_or_default() }],
                "isError": false
            })
        }

        "governed_gate_pr" => {
            let candidate_id = get_str(&args, "candidate_id", "PR-candidate");
            let code = get_str(&args, "code", "");

            let mut paren_count: i32 = 0;
            let mut brace_count: i32 = 0;
            let mut bracket_count: i32 = 0;
            let mut in_single_quote = false;
            let mut in_double_quote = false;
            let mut escaped = false;
            let mut syntax_error = None;

            for (idx, ch) in code.char_indices() {
                if escaped {
                    escaped = false;
                    continue;
                }
                if ch == '\\' {
                    escaped = true;
                    continue;
                }
                if ch == '\'' && !in_double_quote {
                    in_single_quote = !in_single_quote;
                    continue;
                }
                if ch == '"' && !in_single_quote {
                    in_double_quote = !in_double_quote;
                    continue;
                }
                if in_single_quote || in_double_quote {
                    continue;
                }

                match ch {
                    '(' => paren_count += 1,
                    ')' => {
                        paren_count -= 1;
                        if paren_count < 0 {
                            syntax_error = Some(format!(
                                "Unexpected closing parenthesis at character {}",
                                idx
                            ));
                            break;
                        }
                    }
                    '{' => brace_count += 1,
                    '}' => {
                        brace_count -= 1;
                        if brace_count < 0 {
                            syntax_error =
                                Some(format!("Unexpected closing brace at character {}", idx));
                            break;
                        }
                    }
                    '[' => bracket_count += 1,
                    ']' => {
                        bracket_count -= 1;
                        if bracket_count < 0 {
                            syntax_error =
                                Some(format!("Unexpected closing bracket at character {}", idx));
                            break;
                        }
                    }
                    _ => {}
                }
            }

            if syntax_error.is_none() {
                if in_single_quote || in_double_quote {
                    syntax_error = Some("Unterminated string literal".to_string());
                } else if paren_count != 0 {
                    syntax_error = Some(format!(
                        "Unclosed parenthesis (unbalanced by {})",
                        paren_count
                    ));
                } else if brace_count != 0 {
                    syntax_error = Some(format!("Unclosed brace (unbalanced by {})", brace_count));
                } else if bracket_count != 0 {
                    syntax_error = Some(format!(
                        "Unclosed bracket (unbalanced by {})",
                        bracket_count
                    ));
                }
            }

            let passed = syntax_error.is_none();
            let out = json!({
                "candidate_id": candidate_id,
                "passed": passed,
                "terminal_tier": if passed { 1 } else { 0 },
                "action": if passed { "QUALIFIED_FOR_VERIFICATION" } else { "HALT_AND_REJECT" },
                "short_circuited": !passed,
                "docker_container_avoided": !passed,
                "estimated_compute_savings_usd": if !passed { 0.02 } else { 0.0 },
                "diagnostic": syntax_error.unwrap_or_else(|| "Tier 0 AST Passed: Balanced syntax tokens".to_string())
            });

            json!({
                "content": [{ "type": "text", "text": serde_json::to_string_pretty(&out).unwrap_or_default() }],
                "isError": false
            })
        }

        "governed_evaluate_action" => {
            let action_name = get_str(&args, "action_name", "unnamed_action");
            let dist = args
                .get("consequence_distribution")
                .cloned()
                .unwrap_or_else(|| json!({}));

            let p_failure = get_f64(&dist, "p_failure", 0.05).clamp(0.0, 1.0);
            let severity_mean = get_f64(&dist, "severity_mean", 1.0).max(0.0);
            let severity_std = get_f64(&dist, "severity_std", 0.0).max(0.0);
            let reversibility = get_f64(&dist, "reversibility", 0.5).clamp(0.0, 1.0);
            let review_cost = get_f64(&args, "review_cost", 0.002).max(0.0);

            // Only the unrecoverable share of the loss is a reason to defer.
            let unrecoverable = 1.0 - reversibility;
            let expected_loss = p_failure * severity_mean * unrecoverable;
            let tail_loss = p_failure * (severity_mean + 1.645 * severity_std) * unrecoverable;

            let requires_human_signoff = expected_loss > review_cost;
            let risk_tier = if tail_loss < 0.01 {
                "LOW"
            } else if tail_loss < 0.05 {
                "MODERATE"
            } else if tail_loss < 0.25 {
                "ELEVATED"
            } else {
                "CRITICAL"
            };

            let out = json!({
                "action_name": action_name,
                "risk_tier": risk_tier,
                "requires_human_signoff": requires_human_signoff,
                "expected_loss_autonomous": (expected_loss * 1_000_000.0).round() / 1_000_000.0,
                "tail_loss_p95": (tail_loss * 1_000_000.0).round() / 1_000_000.0,
                "deferral_cost": review_cost,
                "unrecoverable_fraction": unrecoverable,
                "rationale": if requires_human_signoff {
                    "Expected unrecoverable loss exceeds the cost of human review: defer."
                } else {
                    "Expected unrecoverable loss is below the cost of human review: execute autonomously."
                }
            });

            json!({
                "content": [{ "type": "text", "text": serde_json::to_string_pretty(&out).unwrap_or_default() }],
                "isError": false
            })
        }

        "governed_stopping_policy" => {
            let cost_per_step = get_f64(&args, "cost_per_step", 0.0).max(0.0);
            let payoffs = get_f64_array(&args, "terminal_payoffs");

            if payoffs.is_empty() {
                return json!({
                    "content": [{ "type": "text", "text": "terminal_payoffs must contain at least one value." }],
                    "isError": true
                });
            }

            let horizon = match get_usize(&args, "horizon_steps", 0) {
                0 => payoffs.len() - 1,
                h => h.min(512),
            };

            // The last supplied payoff is carried forward past the end of the vector.
            let payoff_at = |t: usize| payoffs[t.min(payoffs.len() - 1)];

            let mut values = vec![0.0_f64; horizon + 1];
            let mut stop_here = vec![true; horizon + 1];
            values[horizon] = payoff_at(horizon);

            for t in (0..horizon).rev() {
                let continuation = values[t + 1] - cost_per_step;
                let stopping = payoff_at(t);
                if stopping >= continuation {
                    values[t] = stopping;
                    stop_here[t] = true;
                } else {
                    values[t] = continuation;
                    stop_here[t] = false;
                }
            }

            let optimal_stopping_step = stop_here.iter().position(|&s| s).unwrap_or(horizon);
            let round = |v: f64| (v * 1_000_000.0).round() / 1_000_000.0;

            let out = json!({
                "horizon_steps": horizon,
                "cost_per_step": cost_per_step,
                "optimal_stopping_step": optimal_stopping_step,
                "expected_net_payoff": round(values[0]),
                "value_function": values.iter().map(|v| round(*v)).collect::<Vec<f64>>(),
                "stop_at_step": stop_here,
                "policy_rationale": if optimal_stopping_step == 0 {
                    "Stopping immediately dominates: no step of evidence pays for its own cost."
                } else {
                    "Continuation dominates until the marginal step stops covering cost_per_step."
                }
            });

            json!({
                "content": [{ "type": "text", "text": serde_json::to_string_pretty(&out).unwrap_or_default() }],
                "isError": false
            })
        }

        "governed_invariant_check" => {
            let state = args
                .get("state_snapshot")
                .cloned()
                .unwrap_or_else(|| json!({}));
            let rules: Vec<String> = args
                .get("invariant_rules")
                .and_then(|v| v.as_array())
                .map(|a| {
                    a.iter()
                        .filter_map(|x| x.as_str().map(|s| s.to_string()))
                        .collect()
                })
                .unwrap_or_default();

            let mut violated: Vec<String> = Vec::new();
            let mut unparsed: Vec<String> = Vec::new();
            let mut evaluated: Vec<Value> = Vec::new();

            for rule in &rules {
                let parts: Vec<&str> = rule.split_whitespace().collect();

                if parts.len() == 2 && parts[0].eq_ignore_ascii_case("exists") {
                    let present = state.get(parts[1]).is_some();
                    if !present {
                        violated.push(rule.clone());
                    }
                    evaluated
                        .push(json!({ "rule": rule, "passed": present, "actual": Value::Null }));
                    continue;
                }

                if parts.len() != 3 {
                    unparsed.push(rule.clone());
                    continue;
                }

                let bound: f64 = match parts[2].parse() {
                    Ok(v) => v,
                    Err(_) => {
                        unparsed.push(rule.clone());
                        continue;
                    }
                };

                // A rule about a field the snapshot does not carry is a violation,
                // not something to pass silently.
                let actual = match state.get(parts[0]).and_then(|v| v.as_f64()) {
                    Some(v) => v,
                    None => {
                        violated.push(rule.clone());
                        evaluated
                            .push(json!({ "rule": rule, "passed": false, "actual": Value::Null }));
                        continue;
                    }
                };

                let passed = match parts[1] {
                    "<=" => actual <= bound,
                    "<" => actual < bound,
                    ">=" => actual >= bound,
                    ">" => actual > bound,
                    "==" => (actual - bound).abs() < 1e-9,
                    "!=" => (actual - bound).abs() >= 1e-9,
                    _ => {
                        unparsed.push(rule.clone());
                        continue;
                    }
                };

                if !passed {
                    violated.push(rule.clone());
                }
                evaluated.push(json!({ "rule": rule, "passed": passed, "actual": actual }));
            }

            let valid = violated.is_empty() && unparsed.is_empty();
            let out = json!({
                "valid": valid,
                "violated_invariants": violated,
                "unparsed_rules": unparsed,
                "rules_evaluated": evaluated,
                "gate_decision": if valid { "COMMIT" } else { "BLOCK" }
            });

            json!({
                "content": [{ "type": "text", "text": serde_json::to_string_pretty(&out).unwrap_or_default() }],
                "isError": false
            })
        }

        "governed_shadow_price" => {
            let backlog = get_f64(&args, "queue_backlog", 0.0).max(0.0);
            let headcount = get_f64(&args, "reviewer_headcount", 0.0).max(0.0);
            let per_reviewer = get_f64(&args, "reviews_per_reviewer", 8.0).max(0.0);
            let alpha = get_f64(&args, "tail_index_alpha", 1.25).max(0.01);

            let capacity = headcount * per_reviewer;
            if capacity <= 0.0 {
                return json!({
                    "content": [{ "type": "text", "text": "reviewer_headcount and reviews_per_reviewer must give a capacity above zero." }],
                    "isError": true
                });
            }

            let utilization = backlog / capacity;

            // Under Pareto(alpha) the utility of the K-th best of N candidates scales as
            // (N/K)^(1/alpha). Below capacity nothing is scarce, so lambda is zero.
            let lambda = if backlog <= capacity {
                0.0
            } else {
                utilization.powf(1.0 / alpha) - 1.0
            };

            let out = json!({
                "review_capacity_K": capacity,
                "queue_backlog": backlog,
                "utilization_rho": (utilization * 10_000.0).round() / 10_000.0,
                "shadow_price_lambda": (lambda * 1_000_000.0).round() / 1_000_000.0,
                "cutoff_elevation_factor": ((1.0 + lambda) * 1_000_000.0).round() / 1_000_000.0,
                "tail_index_alpha": alpha,
                "interpretation": if lambda <= 0.0 {
                    "Capacity exceeds backlog: reviewer time is not scarce and the admission cutoff is unshifted."
                } else {
                    "Reviewer time is scarce: admit only candidates whose net VOI clears the elevated cutoff."
                }
            });

            json!({
                "content": [{ "type": "text", "text": serde_json::to_string_pretty(&out).unwrap_or_default() }],
                "isError": false
            })
        }

        "governed_recall_scaling" => {
            let budgets = get_f64_array(&args, "historical_budget_K");
            let recalls = get_f64_array(&args, "recall_points");
            let threshold = get_f64(&args, "marginal_recall_threshold", 0.001).max(1e-9);

            let pairs = budgets.len().min(recalls.len());
            if pairs == 0 {
                return json!({
                    "content": [{ "type": "text", "text": "historical_budget_K and recall_points must both be non-empty." }],
                    "isError": true
                });
            }

            // Linearise Recall(K) = 1 - exp(-gamma K) as -ln(1 - R) = gamma K, then fit
            // least squares through the origin.
            let mut numerator = 0.0;
            let mut denominator = 0.0;
            let mut linearised: Vec<(f64, f64)> = Vec::new();

            for i in 0..pairs {
                let k = budgets[i];
                if k <= 0.0 {
                    continue;
                }
                let r = recalls[i].clamp(0.0, 0.999);
                let y = -(1.0 - r).ln();
                numerator += k * y;
                denominator += k * k;
                linearised.push((k, y));
            }

            if denominator <= 0.0 {
                return json!({
                    "content": [{ "type": "text", "text": "historical_budget_K must contain at least one positive budget." }],
                    "isError": true
                });
            }

            let gamma = numerator / denominator;

            let mean_y = linearised.iter().map(|(_, y)| *y).sum::<f64>() / linearised.len() as f64;
            let ss_tot: f64 = linearised.iter().map(|(_, y)| (y - mean_y).powi(2)).sum();
            let ss_res: f64 = linearised
                .iter()
                .map(|(k, y)| (y - gamma * k).powi(2))
                .sum();
            let r_squared = if ss_tot > 0.0 {
                1.0 - ss_res / ss_tot
            } else {
                1.0
            };

            // dR/dK = gamma exp(-gamma K); spending stops where that falls under threshold.
            let optimal_budget = if gamma > threshold {
                (gamma / threshold).ln() / gamma
            } else {
                0.0
            };
            let predicted_recall = 1.0 - (-gamma * optimal_budget).exp();
            let round = |v: f64| (v * 1_000_000.0).round() / 1_000_000.0;

            let out = json!({
                "fitted_gamma": round(gamma),
                "optimal_budget_point": round(optimal_budget),
                "predicted_recall_at_optimum": round(predicted_recall),
                "r_squared": round(r_squared),
                "samples_used": linearised.len(),
                "marginal_recall_threshold": threshold,
                "model": "Recall(K) = 1 - exp(-gamma K)"
            });

            json!({
                "content": [{ "type": "text", "text": serde_json::to_string_pretty(&out).unwrap_or_default() }],
                "isError": false
            })
        }

        "governed_runtime_audit" => {
            let execution_id = get_str(&args, "execution_id", "");
            if execution_id.is_empty() {
                return json!({
                    "content": [{ "type": "text", "text": "execution_id is required." }],
                    "isError": true
                });
            }

            let include_trace = get_bool(&args, "include_posterior_trace", false);
            let payload = args
                .get("decision_payload")
                .cloned()
                .unwrap_or_else(|| json!({}));
            let timestamp = utc_timestamp_rfc3339();

            let receipt = json!({
                "execution_id": execution_id,
                "decision_payload": payload,
                "timestamp": timestamp
            });
            let canonical = serde_json::to_string(&receipt).unwrap_or_default();

            let mut hasher = Sha256::new();
            hasher.update(canonical.as_bytes());
            let proof_hash = hasher
                .finalize()
                .iter()
                .fold(String::new(), |mut acc, byte| {
                    let _ = write!(acc, "{byte:02x}");
                    acc
                });

            let mut out = json!({
                "execution_id": execution_id,
                "proof_hash": proof_hash,
                "timestamp": timestamp,
                "verified": true,
                "algorithm": "SHA-256",
                "canonical_bytes": canonical.len(),
                "verification_note": "Recompute SHA-256 over {execution_id, decision_payload, timestamp} to verify."
            });

            if include_trace {
                out["posterior_trace"] = payload
                    .get("posterior_trace")
                    .cloned()
                    .unwrap_or_else(|| json!([]));
            }

            json!({
                "content": [{ "type": "text", "text": serde_json::to_string_pretty(&out).unwrap_or_default() }],
                "isError": false
            })
        }

        "aetre_investment_benchmark" => {
            let n_deals = get_usize(&args, "n_deals", 1000).clamp(1, 20_000);
            let diligence_budget = get_usize(&args, "diligence_budget", 50).max(1);
            let tail_alpha = get_f64(&args, "tail_alpha", 1.25).max(0.01);
            let wrapper_pct = get_f64(&args, "wrapper_pct", 0.30).clamp(0.0, 1.0);
            let selection_boundary = get_f64(&args, "selection_boundary", 6.0);
            let hours_per_diligence = get_f64(&args, "hours_per_diligence", 20.0).max(0.0);

            let deals = generate_synthetic_venture_dealflow(
                n_deals,
                tail_alpha,
                wrapper_pct,
                selection_boundary,
            );
            let comparison = evaluate_venture_benchmark(
                &deals,
                diligence_budget,
                tail_alpha,
                hours_per_diligence,
            );

            let body = serde_json::to_value(&comparison).unwrap_or_else(|_| json!({}));
            let out = json!({
                "cohort_size": n_deals,
                "diligence_budget": diligence_budget,
                "tail_index_alpha": tail_alpha,
                "hours_per_diligence": hours_per_diligence,
                "benchmark": body
            });

            json!({
                "content": [{ "type": "text", "text": serde_json::to_string_pretty(&out).unwrap_or_default() }],
                "isError": false
            })
        }

        "aetre_staking_curve" => {
            let c_gen = get_f64(&args, "c_gen", 0.01).max(0.0);
            let value = get_f64(&args, "private_acceptance_value", 100.0).max(0.0);
            let applicants = get_usize(&args, "total_potential_applicants", 5000).max(1);
            let capacity = get_usize(&args, "acceptance_capacity", 200).max(1);
            let max_fee = get_f64(&args, "max_fee", 20.0).max(0.0);
            let steps = get_usize(&args, "steps", 10).clamp(2, 200);

            let curve = generate_staking_curve(c_gen, value, applicants, capacity, max_fee, steps);
            let points = serde_json::to_value(&curve).unwrap_or_else(|_| json!([]));

            // The knee: the fee that deters the most low-quality volume per unit of fee.
            let best = curve
                .iter()
                .filter(|p| p.submission_fee > 0.0)
                .max_by(|a, b| {
                    let da = a.low_quality_spam_deterred_pct / a.submission_fee;
                    let db = b.low_quality_spam_deterred_pct / b.submission_fee;
                    da.partial_cmp(&db).unwrap_or(std::cmp::Ordering::Equal)
                })
                .map(|p| p.submission_fee);

            let out = json!({
                "total_potential_applicants": applicants,
                "acceptance_capacity": capacity,
                "max_fee": max_fee,
                "steps": steps,
                "most_efficient_fee": best,
                "curve": points
            });

            json!({
                "content": [{ "type": "text", "text": serde_json::to_string_pretty(&out).unwrap_or_default() }],
                "isError": false
            })
        }

        _ => json!({
            "content": [
                {
                    "type": "text",
                    "text": format!("Unknown tool: {}", name)
                }
            ],
            "isError": true
        }),
    }
}

#[cfg(test)]
mod provenance_tests {
    use super::*;

    fn body(result: &Value) -> Value {
        serde_json::from_str(result.pointer("/content/0/text").unwrap().as_str().unwrap()).unwrap()
    }

    #[test]
    fn unstated_source_and_defaults_are_recorded() {
        let out = call_tool(
            "governed_review_boundary",
            json!({ "belief": 0.7, "reward": 0.05, "loss": 0.3 }),
        );
        let b = body(&out);
        assert!(b["prior_source"].as_str().unwrap().starts_with("unstated"));
        // The caller supplied three of six; the rest must be named.
        let defaulted = b["defaulted_arguments"].as_array().unwrap();
        assert!(defaulted.iter().any(|v| v == "review_cost"));
    }

    #[test]
    fn a_stated_source_is_echoed_and_nothing_is_defaulted() {
        let out = call_tool(
            "governed_review_boundary",
            json!({
                "belief": 0.7, "reward": 0.05, "loss": 0.3, "review_cost": 0.02,
                "shadow_price_lambda": 0.0, "review_accuracy": 0.9,
                "prior_source": "measured: 2026 cycle"
            }),
        );
        let b = body(&out);
        assert_eq!(b["prior_source"], "measured: 2026 cycle");
        assert!(b.get("defaulted_arguments").is_none());
    }

    #[test]
    fn tools_without_estimates_are_left_alone() {
        let out = call_tool(
            "aetre_check_governor",
            json!({ "arrival_rate": 96.0, "service_rate": 100.0 }),
        );
        assert!(body(&out).get("prior_source").is_none());
    }
}

#[cfg(test)]
mod boundary_fit_tests {
    use super::*;

    /// Candidates whose true flips sit near 5.5, so a boundary there should rank
    /// them above a boundary at 8.0.
    fn corpus() -> Vec<BacktestCandidate> {
        (0..200)
            .map(|i| {
                let mean = 3.0 + (i % 10) as f64 * 0.5;
                let flips = (5.0..6.0).contains(&mean);
                BacktestCandidate {
                    id: format!("c{i}"),
                    split: "calib".to_string(),
                    label: u8::from(flips),
                    pre_triage_data: BacktestPreTriage {
                        preliminary_mean: mean,
                        preliminary_variance: 1.0,
                        m_reviews_count: Some(2),
                        preliminary_mean_confidence: Some(3.0),
                    },
                }
            })
            .collect()
    }

    #[test]
    fn a_boundary_near_the_flips_beats_one_far_away() {
        let records = corpus();
        let near = score_backtest(&records, "calib", 40, 5.5);
        let far = score_backtest(&records, "calib", 40, 8.0);
        assert!(
            near.precision > far.precision,
            "near {} should beat far {}",
            near.precision,
            far.precision
        );
    }

    #[test]
    fn scoring_respects_the_split_and_the_budget() {
        let records = corpus();
        let s = score_backtest(&records, "calib", 40, 5.5);
        assert_eq!(s.evaluated, 200);
        assert_eq!(s.budget, 40);
        assert!(s.caught <= s.budget);
        assert_eq!(score_backtest(&records, "test", 40, 5.5).evaluated, 0);
    }

    #[test]
    fn fitting_requires_a_dataset_rather_than_guessing_one() {
        let out = call_tool("aetre_fit_boundary", json!({}));
        assert_eq!(out["isError"], true);
    }
}
