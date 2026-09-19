#![recursion_limit = "256"]

mod dispatch;
mod helpers;
mod heuristics;
mod layer;
mod license;
mod prompts;
mod protocol;
mod resources;
mod schemas;

pub mod format;
pub mod server;

pub use dispatch::call_tool;
pub use helpers::{get_bool, get_f64, get_f64_array, get_str, get_usize, utc_timestamp_rfc3339};
pub use heuristics::{analyze_text_heuristics, EpistemicDiagnostics};
pub use layer::{active_layer, layer_from_args, set_layer, tool_in_layer, Layer};
pub use prompts::{get_prompt, list_prompts};
pub use resources::{list_resource_templates, list_resources, read_resource};
pub use schemas::list_tools;

use protocol::{JsonRpcError, JsonRpcRequest, JsonRpcResponse};
use serde_json::{json, Value};
use std::io::{self, BufRead, Write};

fn main() -> io::Result<()> {
    let args: Vec<String> = std::env::args().collect();

    match layer_from_args(&args) {
        Ok(layer) => set_layer(layer),
        Err(message) => {
            eprintln!("ERROR: {message}");
            std::process::exit(2);
        }
    }

    let has_explicit_http_env =
        std::env::var("AETRE_HTTP_SERVER_TOKEN").is_ok() || std::env::var("PORT").is_ok();
    let http_requested_by_flag = args
        .iter()
        .any(|a| a == "--studio" || a == "--serve" || a == "studio" || a == "--web");
    let is_studio_mode = http_requested_by_flag || has_explicit_http_env;
    let no_browser = args
        .iter()
        .any(|a| a == "--no-browser" || a == "--headless")
        || has_explicit_http_env;

    // Only start embedded web server and open browser when explicitly in studio mode
    if is_studio_mode {
        let port = std::env::var("PORT")
            .ok()
            .and_then(|value| value.parse::<u16>().ok())
            .unwrap_or(8080);
        if let Err(err) = server::start_embedded_server(port, !no_browser) {
            eprintln!("ERROR: could not start the AETRE HTTP server on port {port}: {err}");
            eprintln!("HINT: set AETRE_HTTP_SERVER_TOKEN when AETRE_BIND_ADDRESS is not");
            eprintln!(
                "      loopback (that is the case inside a container), or bind to 127.0.0.1."
            );
            if http_requested_by_flag {
                // HTTP was asked for by name, so failing to provide it is fatal.
                eprintln!("FATAL: exiting rather than idling with nothing listening.");
                std::process::exit(1);
            }
            // Studio mode was only inferred from PORT / AETRE_HTTP_SERVER_TOKEN being
            // set. Keep serving MCP over stdio, which is what `docker run -i` relies on.
            eprintln!("NOTE: continuing in stdio MCP mode; no HTTP listener is available.");
        }
    }

    let stdin = io::stdin();
    let stdout = io::stdout();
    let mut stdout_handle = stdout.lock();

    for line in stdin.lock().lines() {
        if let Ok(raw_line) = line {
            let trimmed = raw_line.trim();
            if trimmed.is_empty() {
                continue;
            }

            if let Ok(req) = serde_json::from_str::<JsonRpcRequest>(trimmed) {
                if let Some(resp) = handle_request(req) {
                    if let Ok(json_str) = serde_json::to_string(&resp) {
                        let _ = writeln!(stdout_handle, "{}", json_str);
                        let _ = stdout_handle.flush();
                    }
                }
            }
        } else {
            break;
        }
    }

    // Keep embedded web studio alive when running in studio mode
    if is_studio_mode {
        loop {
            std::thread::sleep(std::time::Duration::from_secs(3600));
        }
    }

    Ok(())
}

fn handle_request(req: JsonRpcRequest) -> Option<JsonRpcResponse> {
    match req.method.as_str() {
        "initialize" => Some(JsonRpcResponse {
            jsonrpc: "2.0".to_string(),
            id: req.id,
            result: Some(json!({
                "protocolVersion": "2024-11-05",
                "capabilities": {
                    "tools": { "listChanged": false },
                    "resources": { "subscribe": false, "listChanged": false },
                    "prompts": { "listChanged": false }
                },
                "serverInfo": {
                    "name": "aetre-mcp",
                    "version": env!("CARGO_PKG_VERSION")
                }
            })),
            error: None,
        }),

        "notifications/initialized" => None,

        "ping" => Some(JsonRpcResponse {
            jsonrpc: "2.0".to_string(),
            id: req.id,
            result: Some(json!({})),
            error: None,
        }),

        "resources/list" => Some(JsonRpcResponse {
            jsonrpc: "2.0".to_string(),
            id: req.id,
            result: Some(json!({
                "resources": list_resources()
            })),
            error: None,
        }),

        "resources/templates/list" => Some(JsonRpcResponse {
            jsonrpc: "2.0".to_string(),
            id: req.id,
            result: Some(json!({
                "resourceTemplates": list_resource_templates()
            })),
            error: None,
        }),

        "resources/subscribe" => Some(JsonRpcResponse {
            jsonrpc: "2.0".to_string(),
            id: req.id,
            result: Some(json!({})),
            error: None,
        }),

        "resources/unsubscribe" => Some(JsonRpcResponse {
            jsonrpc: "2.0".to_string(),
            id: req.id,
            result: Some(json!({})),
            error: None,
        }),

        "logging/setLevel" => Some(JsonRpcResponse {
            jsonrpc: "2.0".to_string(),
            id: req.id,
            result: Some(json!({})),
            error: None,
        }),

        "completion/complete" => Some(JsonRpcResponse {
            jsonrpc: "2.0".to_string(),
            id: req.id,
            result: Some(json!({
                "completion": {
                    "values": [],
                    "total": 0,
                    "hasMore": false
                }
            })),
            error: None,
        }),

        "resources/read" => {
            let params = req.params.unwrap_or(Value::Null);
            let uri = params.get("uri").and_then(|v| v.as_str()).unwrap_or("");
            match read_resource(uri) {
                Ok(res) => Some(JsonRpcResponse {
                    jsonrpc: "2.0".to_string(),
                    id: req.id,
                    result: Some(res),
                    error: None,
                }),
                Err(err_msg) => Some(JsonRpcResponse {
                    jsonrpc: "2.0".to_string(),
                    id: req.id,
                    result: None,
                    error: Some(JsonRpcError {
                        code: -32602,
                        message: err_msg,
                        data: None,
                    }),
                }),
            }
        }

        "prompts/list" => Some(JsonRpcResponse {
            jsonrpc: "2.0".to_string(),
            id: req.id,
            result: Some(json!({
                "prompts": list_prompts()
            })),
            error: None,
        }),

        "prompts/get" => {
            let params = req.params.unwrap_or(Value::Null);
            let prompt_name = params.get("name").and_then(|v| v.as_str()).unwrap_or("");
            let arguments = params.get("arguments").cloned().unwrap_or(json!({}));
            match get_prompt(prompt_name, arguments) {
                Ok(res) => Some(JsonRpcResponse {
                    jsonrpc: "2.0".to_string(),
                    id: req.id,
                    result: Some(res),
                    error: None,
                }),
                Err(err_msg) => Some(JsonRpcResponse {
                    jsonrpc: "2.0".to_string(),
                    id: req.id,
                    result: None,
                    error: Some(JsonRpcError {
                        code: -32602,
                        message: err_msg,
                        data: None,
                    }),
                }),
            }
        }

        "tools/list" => Some(JsonRpcResponse {
            jsonrpc: "2.0".to_string(),
            id: req.id,
            result: Some(json!({
                "tools": list_tools()
            })),
            error: None,
        }),

        "tools/call" => {
            let params = req.params.unwrap_or(Value::Null);
            let tool_name = params.get("name").and_then(|v| v.as_str()).unwrap_or("");
            let arguments = params.get("arguments").cloned().unwrap_or(json!({}));

            let result = call_tool(tool_name, arguments);
            Some(JsonRpcResponse {
                jsonrpc: "2.0".to_string(),
                id: req.id,
                result: Some(result),
                error: None,
            })
        }

        _ => Some(JsonRpcResponse {
            jsonrpc: "2.0".to_string(),
            id: req.id,
            result: None,
            error: Some(JsonRpcError {
                code: -32601,
                message: format!("Method '{}' not found", req.method),
                data: None,
            }),
        }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_list_tools() {
        let tools = list_tools();
        let arr = tools.as_array().unwrap();
        assert_eq!(arr.len(), 32);
    }

    #[test]
    fn test_resilient_parameter_parsing() {
        // Stringified numbers should parse cleanly
        let args = json!({
            "posterior_mean": "1.15",
            "posterior_variance": "0.80",
            "selection_boundary": "1.20",
            "total_candidates": "5000",
            "is_active": "true"
        });

        assert_eq!(get_f64(&args, "posterior_mean", 0.0), 1.15);
        assert_eq!(get_f64(&args, "posterior_variance", 0.0), 0.80);
        assert_eq!(get_usize(&args, "total_candidates", 0), 5000);
        assert!(get_bool(&args, "is_active", false));
    }

    #[test]
    fn test_list_and_read_resources() {
        let res_list = list_resources();
        let arr = res_list.as_array().unwrap();
        assert_eq!(arr.len(), 4);

        let read_datasets = read_resource("aetre://catalog/datasets");
        assert!(read_datasets.is_ok());
        let val = read_datasets.unwrap();
        let content_text = val["contents"][0]["text"].as_str().unwrap();
        assert!(content_text.contains("openreview"));
        assert!(content_text.contains("nih"));

        let read_db = read_resource("aetre://schemas/database-writeback");
        assert!(read_db.is_ok());

        let read_specs = read_resource("aetre://specs/mathematical-formulations");
        assert!(read_specs.is_ok());

        let read_tiers = read_resource("aetre://institutional/tiers");
        assert!(read_tiers.is_ok());

        // Test Dynamic Template Resolution
        let read_nih_template = read_resource("aetre://datasets/nih");
        assert!(read_nih_template.is_ok());
        let nih_val = read_nih_template.unwrap();
        assert!(nih_val["contents"][0]["text"]
            .as_str()
            .unwrap()
            .contains("NIH"));

        let read_openreview_template = read_resource("aetre://datasets/openreview");
        assert!(read_openreview_template.is_ok());

        let read_proposal_template = read_resource("aetre://proposals/prop_test_99");
        assert!(read_proposal_template.is_ok());

        let invalid = read_resource("aetre://invalid/uri");
        assert!(invalid.is_err());
    }

    #[test]
    fn test_list_and_get_prompts() {
        let prompt_list = list_prompts();
        let arr = prompt_list.as_array().unwrap();
        assert_eq!(arr.len(), 3);

        let get_preflight = get_prompt(
            "author_preflight_review",
            json!({
                "title": "Test Title",
                "abstract": "Test Abstract",
                "boundary": "1.2"
            }),
        );
        assert!(get_preflight.is_ok());
        let val = get_preflight.unwrap();
        assert!(val["messages"][0]["content"]["text"]
            .as_str()
            .unwrap()
            .contains("Test Title"));

        let invalid_prompt = get_prompt("unknown_prompt", json!({}));
        assert!(invalid_prompt.is_err());
    }

    #[test]
    fn test_call_system_catalog() {
        let res = call_tool(
            "aetre_system_catalog",
            json!({
                "query_type": "all"
            }),
        );
        let is_err = res.get("isError").and_then(|v| v.as_bool()).unwrap();
        assert!(!is_err);
        let content_text = res["content"][0]["text"].as_str().unwrap();
        assert!(content_text.contains("AETRE"));
        assert!(content_text.contains("openreview"));

        let res_datasets = call_tool(
            "aetre_system_catalog",
            json!({
                "query_type": "datasets"
            }),
        );
        assert!(!res_datasets
            .get("isError")
            .and_then(|v| v.as_bool())
            .unwrap());
    }

    #[test]
    fn test_call_triage_proposal() {
        let res = call_tool(
            "aetre_triage_proposal",
            json!({
                "text": "We propose a novel hybrid quantum variational eigensolver for solid-state battery electrolyte synthesis with preliminary density functional validation.",
                "selection_boundary": "1.2"
            }),
        );
        let is_err = res.get("isError").and_then(|v| v.as_bool()).unwrap();
        assert!(!is_err);
        let content_text = res["content"][0]["text"].as_str().unwrap();
        assert!(content_text.contains("markdown_scorecard"));
        assert!(content_text.contains("epistemic_diagnostics"));
    }

    #[test]
    fn test_call_calculate_voi() {
        let res = call_tool(
            "aetre_calculate_voi",
            json!({
                "posterior_mean": "1.15",
                "posterior_variance": "0.8",
                "selection_boundary": "1.2"
            }),
        );
        let is_err = res.get("isError").and_then(|v| v.as_bool()).unwrap();
        assert!(!is_err);
        let content_text = res["content"][0]["text"].as_str().unwrap();
        assert!(content_text.contains("markdown_scorecard"));
    }

    #[test]
    fn test_call_check_governor() {
        let res_locked = call_tool(
            "aetre_check_governor",
            json!({
                "arrival_rate": 96.0,
                "service_rate": 100.0
            }),
        );
        // No licence supplied: the tool must still compute.
        let text_locked = res_locked["content"][0]["text"].as_str().unwrap();
        assert!(!text_locked.contains("TIER_LOCKED"));
        assert!(!res_locked["isError"].as_bool().unwrap_or(false));

        let res_unlocked = call_tool(
            "aetre_check_governor",
            json!({
                "api_key": "aetre_ent_test_key",
                "arrival_rate": "96.0",
                "service_rate": "100.0"
            }),
        );
        let text_unlocked = res_unlocked["content"][0]["text"].as_str().unwrap();
        assert!(text_unlocked.contains("CRITICAL_SATURATION"));
        assert!(text_unlocked.contains("markdown_scorecard"));
    }

    #[test]
    fn test_call_exploration_audit() {
        let res_locked = call_tool(
            "aetre_exploration_audit",
            json!({
                "deprioritized_pool_size": 4800,
                "audited_sample_size": 25,
                "audited_high_value_found": 1
            }),
        );
        // No licence supplied: the tool must still compute.
        assert!(!res_locked["content"][0]["text"]
            .as_str()
            .unwrap()
            .contains("TIER_LOCKED"));
        assert!(!res_locked["isError"].as_bool().unwrap_or(false));

        let res_unlocked = call_tool(
            "aetre_exploration_audit",
            json!({
                "api_key": "aetre_ent_test_key",
                "deprioritized_pool_size": "4800",
                "audited_sample_size": "25",
                "audited_high_value_found": "1"
            }),
        );
        assert!(res_unlocked["content"][0]["text"]
            .as_str()
            .unwrap()
            .contains("estimated_hidden_high_value_H_hat_D"));
        assert!(res_unlocked["content"][0]["text"]
            .as_str()
            .unwrap()
            .contains("markdown_scorecard"));
    }

    #[test]
    fn test_call_simulate_benchmark() {
        let res_locked = call_tool(
            "aetre_simulate_benchmark",
            json!({
                "replications": 10
            }),
        );
        // No licence supplied: the tool must still compute.
        assert!(!res_locked["content"][0]["text"]
            .as_str()
            .unwrap()
            .contains("TIER_LOCKED"));
        assert!(!res_locked["isError"].as_bool().unwrap_or(false));

        let res_unlocked = call_tool(
            "aetre_simulate_benchmark",
            json!({
                "api_key": "aetre_ent_test_key",
                "replications": 5,
                "baseline_arrivals": 100,
                "acceptance_capacity": 20
            }),
        );
        let text = res_unlocked["content"][0]["text"].as_str().unwrap();
        assert!(text.contains("regime_results"));
        assert!(text.contains("markdown_scorecard"));
    }

    #[test]
    fn test_call_batch_triage() {
        let res = call_tool(
            "aetre_batch_triage",
            json!({
                "proposals": [
                    { "title": "Quantum Battery", "text": "Novel quantum variational eigensolver for solid-state battery electrolytes." },
                    { "title": "Wrapper App", "text": "Simple prompt chaining wrapper for customer service on salesforce." },
                    { "title": "CRISPR Therapy", "text": "Synthetic microRNA epigenetic silencing for glioblastoma with empirical in-vitro proofs." }
                ],
                "selection_boundary": "1.2"
            }),
        );
        let is_err = res.get("isError").and_then(|v| v.as_bool()).unwrap();
        assert!(!is_err);
        let text = res["content"][0]["text"].as_str().unwrap();
        assert!(text.contains("ranked_proposals"));
        assert!(text.contains("cohort_allocation"));
        assert!(text.contains("markdown_scorecard"));
    }

    #[test]
    fn test_call_recall_scaling_curve() {
        let res = call_tool(
            "aetre_recall_scaling_curve",
            json!({
                "baseline_arrivals": "1000",
                "selection_capacity": "200",
                "high_value_rate": "0.067"
            }),
        );
        let is_err = res.get("isError").and_then(|v| v.as_bool()).unwrap();
        assert!(!is_err);
        let text = res["content"][0]["text"].as_str().unwrap();
        assert!(text.contains("scaling_curve_points"));
    }

    #[test]
    fn test_call_heldout_backtest() {
        let res = call_tool(
            "aetre_heldout_backtest",
            json!({
                "budget": 20,
                "boundary": 6.0,
                "split": "test"
            }),
        );
        let is_err = res.get("isError").and_then(|v| v.as_bool()).unwrap();
        assert!(!is_err);
        let text = res["content"][0]["text"].as_str().unwrap();
        assert!(text.contains("BACKTEST_EVALUATED_SUCCESSFULLY"));
        assert!(text.contains("aetre_voi_recall_at_k"));
    }

    #[test]
    fn test_call_calibrate_scorer() {
        let res = call_tool(
            "aetre_calibrate_scorer",
            json!({
                "scores": [0.1, 0.2, 0.3, 0.8, 0.9, 1.0],
                "labels": [0, 0, 0, 1, 1, 1],
                "iterations": 200,
                "learning_rate": 0.05
            }),
        );
        let is_err = res.get("isError").and_then(|v| v.as_bool()).unwrap();
        assert!(!is_err);
        let text = res["content"][0]["text"].as_str().unwrap();
        assert!(text.contains("CALIBRATOR_FITTED_SUCCESSFULLY"));
        assert!(text.contains("calibrator_slope"));
    }

    #[test]
    fn test_call_multi_attribute_voi() {
        let res = call_tool(
            "aetre_multi_attribute_voi",
            json!({
                "dimensions": [
                    { "name": "Novelty", "prior_mean": 6.5, "prior_variance": 1.2, "weight": 0.4 },
                    { "name": "Rigor", "prior_mean": 5.2, "prior_variance": 0.8, "weight": 0.4 },
                    { "name": "Impact", "prior_mean": 5.8, "prior_variance": 0.3, "weight": 0.2 }
                ],
                "composite_threshold": 6.0,
                "review_cost_per_dim": 1.0
            }),
        );
        let is_err = res.get("isError").and_then(|v| v.as_bool()).unwrap();
        assert!(!is_err);
        let text = res["content"][0]["text"].as_str().unwrap();
        assert!(text.contains("composite_prior_mean"));
        assert!(text.contains("total_composite_voi"));
        assert!(text.contains("dimension_breakdown"));
    }

    #[test]
    fn test_call_congestion_matching() {
        let res_locked = call_tool(
            "aetre_congestion_matching",
            json!({
                "proposals": [{ "id": "p1", "title": "P1", "domain": "AI", "voi_index": 0.8, "required_reviews": 1 }],
                "reviewers": [{ "id": "r1", "name": "Alice", "domain": "AI", "capacity": 2, "service_rate": 10.0 }]
            }),
        );
        // No licence supplied: the tool must still compute.
        assert!(!res_locked["content"][0]["text"]
            .as_str()
            .unwrap()
            .contains("TIER_LOCKED"));
        assert!(!res_locked["isError"].as_bool().unwrap_or(false));

        let res_unlocked = call_tool(
            "aetre_congestion_matching",
            json!({
                "api_key": "aetre_ent_test_key",
                "proposals": [{ "id": "p1", "title": "P1", "domain": "AI", "voi_index": 0.8, "required_reviews": 1 }],
                "reviewers": [{ "id": "r1", "name": "Alice", "domain": "AI", "capacity": 2, "service_rate": 10.0, "current_load": 0, "arrival_rate": 5.0 }]
            }),
        );
        let is_err = res_unlocked
            .get("isError")
            .and_then(|v| v.as_bool())
            .unwrap();
        assert!(!is_err);
        let text = res_unlocked["content"][0]["text"].as_str().unwrap();
        assert!(text.contains("total_assignments_made"));
        assert!(text.contains("global_affinity_score"));
    }

    #[test]
    fn test_call_sequential_stopping_rule() {
        let res = call_tool(
            "aetre_sequential_stopping_rule",
            json!({
                "prior_mean": 5.0,
                "prior_variance": 1.0,
                "threshold": 6.0,
                "reviews": [
                    { "step": 1, "reviewer_id": "r1", "score": 8.5, "noise_sd": 0.5, "cost": 1.0 },
                    { "step": 2, "reviewer_id": "r2", "score": 8.0, "noise_sd": 0.5, "cost": 1.0 }
                ]
            }),
        );
        let is_err = res.get("isError").and_then(|v| v.as_bool()).unwrap();
        assert!(!is_err);
        let text = res["content"][0]["text"].as_str().unwrap();
        assert!(text.contains("completed_reviews_count"));
        assert!(text.contains("Accept"));
        assert!(text.contains("stopping_rationale"));
    }

    #[test]
    fn test_call_governed_bellman_triage() {
        let res = call_tool(
            "governed_bellman_triage",
            json!({
                "reward": 0.02,
                "loss": 0.10,
                "prior": 0.50,
                "stage": 0,
                "consecutive_passes": 0
            }),
        );
        assert!(!res["isError"].as_bool().unwrap());
        let text = res["content"][0]["text"].as_str().unwrap();
        assert!(text.contains("CONTINUE"));
        assert!(text.contains("critical_threshold_p_star"));
    }

    #[test]
    fn test_call_governed_review_boundary() {
        let res = call_tool(
            "governed_review_boundary",
            json!({
                "belief": 0.98,
                "reward": 0.02,
                "loss": 0.10,
                "review_cost": 0.005,
                "shadow_price_lambda": 0.0
            }),
        );
        assert!(!res["isError"].as_bool().unwrap());
        let text = res["content"][0]["text"].as_str().unwrap();
        assert!(text.contains("AUTO"));
        assert!(text.contains("dominant_expected_utility"));
    }

    #[test]
    fn test_call_governed_knapsack_admit() {
        let res = call_tool(
            "governed_knapsack_admit",
            json!({
                "capacity_k": 2.0,
                "review_cost_k": 1.0,
                "candidates": [
                    { "candidate_id": "PR-1", "posterior_belief": 0.95 },
                    { "candidate_id": "PR-2", "posterior_belief": 0.90 },
                    { "candidate_id": "PR-3", "posterior_belief": 0.88 },
                    { "candidate_id": "PR-4", "posterior_belief": 0.20 }
                ]
            }),
        );
        assert!(!res["isError"].as_bool().unwrap());
        let text = res["content"][0]["text"].as_str().unwrap();
        assert!(text.contains("total_admitted"));
        assert!(text.contains("PR-1"));
        assert!(text.contains("PR-2"));
        assert!(text.contains("capacity_shadow_price_lambda"));
    }

    #[test]
    fn test_call_governed_gate_pr() {
        let res_clean = call_tool(
            "governed_gate_pr",
            json!({
                "candidate_id": "PR-good",
                "code": "def solve(x):\n    return x + 1\n"
            }),
        );
        assert!(!res_clean["isError"].as_bool().unwrap());
        let text_clean = res_clean["content"][0]["text"].as_str().unwrap();
        assert!(text_clean.contains("QUALIFIED_FOR_VERIFICATION"));
        assert!(text_clean.contains("\"passed\": true"));

        let res_bad = call_tool(
            "governed_gate_pr",
            json!({
                "candidate_id": "PR-bad",
                "code": "def broken(\n    return 42\n"
            }),
        );
        assert!(!res_bad["isError"].as_bool().unwrap());
        let text_bad = res_bad["content"][0]["text"].as_str().unwrap();
        assert!(text_bad.contains("HALT_AND_REJECT"));
        assert!(text_bad.contains("\"passed\": false"));
        assert!(text_bad.contains("Unclosed parenthesis"));
    }
}
