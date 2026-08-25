use std::fs;
use std::io::{self, BufRead, Write};
use std::path::PathBuf;

mod heuristics;
mod license;
pub use heuristics::{analyze_text_heuristics, EpistemicDiagnostics};
use license::{
    current_year_month, generate_quota_exceeded_payload, generate_tier_locked_payload,
    get_license_tier, get_preflight_usage, get_quota_status, increment_preflight_usage,
    LicenseTier, COMMUNITY_PREFLIGHT_LIMIT,
};

pub mod format;
pub mod server;

use aetre_core::{
    calculate_boundary_voi, calculate_exploration_audit, calculate_governor_action,
    calculate_heavy_tailed_voi, calculate_proposition_1_bound, correlated_posterior_update,
    evaluate_author_preflight, evaluate_heterogeneous_queues, evaluate_multi_attribute_voi,
    evaluate_quadratic_staking, evaluate_sequential_stopping, evaluate_stage_queue,
    evaluate_submitter_equilibrium, generate_recall_scaling_curve, optimize_congestion_matching,
    run_benchmark_replications, AgentEvaluation, MultiAttributeDimension, Parameters,
    ProposalRequirement, ReviewerProfile, SequentialReviewStep,
};

use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};

// ----------------------------------------------------------------------------
// Resilient Parameter Extraction Helpers (Coerce Strings, Numbers & Booleans)
// ----------------------------------------------------------------------------

pub fn get_f64(args: &Value, key: &str, default: f64) -> f64 {
    args.get(key)
        .and_then(|v| {
            v.as_f64()
                .or_else(|| v.as_str().and_then(|s| s.parse().ok()))
        })
        .unwrap_or(default)
}

pub fn get_usize(args: &Value, key: &str, default: usize) -> usize {
    args.get(key)
        .and_then(|v| {
            v.as_u64()
                .map(|n| n as usize)
                .or_else(|| v.as_str().and_then(|s| s.parse().ok()))
        })
        .unwrap_or(default)
}

pub fn get_str<'a>(args: &'a Value, key: &str, default: &'a str) -> &'a str {
    args.get(key).and_then(|v| v.as_str()).unwrap_or(default)
}

#[allow(dead_code)]
pub fn get_bool(args: &Value, key: &str, default: bool) -> bool {
    args.get(key)
        .and_then(|v| {
            v.as_bool()
                .or_else(|| v.as_str().and_then(|s| s.parse().ok()))
        })
        .unwrap_or(default)
}

#[derive(Debug, Serialize, Deserialize)]
struct JsonRpcRequest {
    jsonrpc: String,
    id: Option<Value>,
    method: String,
    params: Option<Value>,
}

#[derive(Debug, Serialize, Deserialize)]
struct JsonRpcResponse {
    jsonrpc: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    id: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    result: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    error: Option<JsonRpcError>,
}

#[derive(Debug, Serialize, Deserialize)]
struct JsonRpcError {
    code: i32,
    message: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    data: Option<Value>,
}

fn main() -> io::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let is_http_only = args.iter().any(|a| a == "--serve");
    let is_studio_mode = args
        .iter()
        .any(|a| a == "--studio" || a == "--serve" || a == "studio" || a == "--web");
    let no_browser = args
        .iter()
        .any(|a| a == "--no-browser" || a == "--headless");

    // Only start embedded web server and open browser when explicitly in studio mode
    if is_studio_mode {
        let port = std::env::var("PORT")
            .ok()
            .and_then(|value| value.parse::<u16>().ok())
            .unwrap_or(8080);
        server::start_embedded_server(port, !no_browser)
            .map_err(|error| io::Error::other(error.to_string()))?;
        if is_http_only {
            loop {
                std::thread::park();
            }
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
                    "version": "0.1.0"
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

pub fn list_resources() -> Value {
    json!([
        {
            "uri": "aetre://catalog/datasets",
            "name": "AETRE Benchmark Dataset Catalog",
            "description": "Catalog of bundled synthetic fixtures and optional external-data adapters.",
            "mimeType": "application/json"
        },
        {
            "uri": "aetre://schemas/database-writeback",
            "name": "AETRE SQL Database Write-Back Schema",
            "description": "SQL DDL and column definitions for PostgreSQL/SQLite/Snowflake write-back integration.",
            "mimeType": "text/markdown"
        },
        {
            "uri": "aetre://specs/mathematical-formulations",
            "name": "Mathematical & Operations Research Specifications",
            "description": "Core formulas: Proposition 1 Bound, Kingman Heavy-Traffic, Gaussian & Pareto VOI, Horvitz-Thompson Estimator.",
            "mimeType": "text/markdown"
        },
        {
            "uri": "aetre://institutional/tiers",
            "name": "7 Institutional Deployment Tiers Matrix",
            "description": "Cross-tier institutional matrix: Authors, VCs, Publishers, Grant Agencies, Patent Offices, Corporate R&D, Accelerators.",
            "mimeType": "application/json"
        }
    ])
}

pub fn list_resource_templates() -> Value {
    json!([
        {
            "uriTemplate": "aetre://datasets/{dataset_name}",
            "name": "Benchmark Dataset by Identifier",
            "description": "Dynamic resource template for inspecting specific peer-review datasets (openreview, nih, uspto, arxiv, ssrn).",
            "mimeType": "application/json"
        },
        {
            "uriTemplate": "aetre://proposals/{proposal_id}",
            "name": "Pre-flight Benchmark Proposal",
            "description": "Dynamic resource template for retrieving stored proposal evaluations, epistemic priors, and cryptographic verification receipts.",
            "mimeType": "application/json"
        }
    ])
}

fn try_read_file_or_fallback(paths: &[&str], fallback_json: Value) -> String {
    for p in paths {
        if let Ok(content) = fs::read_to_string(PathBuf::from(p)) {
            if !content.trim().is_empty() {
                return content;
            }
        }
    }
    serde_json::to_string_pretty(&fallback_json).unwrap_or_default()
}

pub fn read_resource(uri: &str) -> Result<Value, String> {
    // 1. Static Catalog URIs
    match uri {
        "aetre://catalog/datasets" => {
            let data = json!({
                "catalog_version": "1.0.0",
                "datasets": [
                    {
                        "id": "openreview",
                        "name": "Synthetic Peer-Review Fixture",
                        "source": "AETRE_SYNTHETIC_FIXTURE",
                        "target_test": "Bayesian VOI Triage & Reviewer Disagreement",
                        "records_file": "examples/datasets/openreview_peer_review.json",
                        "resource_uri": "aetre://datasets/openreview",
                        "description": "Fictional review scores and confidence distributions for parser and routing tests.",
                        "key_fields": ["review_scores", "reviewer_confidence", "mean_score", "score_variance", "historical_decision"]
                    },
                    {
                        "id": "nih",
                        "name": "Synthetic Biomedical Grant Fixture",
                        "source": "AETRE_SYNTHETIC_FIXTURE",
                        "target_test": "5% Randomized Horvitz-Thompson Audits (H_hat_D) & Selective-Label Recovery",
                        "records_file": "examples/datasets/nih_grant_proposals.json",
                        "resource_uri": "aetre://datasets/nih",
                        "description": "Fictional grant fields for parser and routing tests.",
                        "key_fields": ["initial_priority_percentile", "epistemic_variance", "requested_budget_usd", "historical_funding_outcome"]
                    },
                    {
                        "id": "uspto",
                        "name": "Synthetic Patent Examination Fixture",
                        "source": "AETRE_SYNTHETIC_FIXTURE",
                        "target_test": "Kingman Heavy-Traffic Backlog & Capacity Governor Throttling",
                        "records_file": "examples/datasets/uspto_patent_applications.json",
                        "resource_uri": "aetre://datasets/uspto",
                        "description": "Fictional examination fields for parser and queue-governor tests.",
                        "key_fields": ["cpc_class", "claims_count", "historical_pendency_months", "examiner_utilization_rho", "office_action_count"]
                    },
                    {
                        "id": "paperswithcode",
                        "name": "Synthetic Code-Artifact Fixture",
                        "source": "AETRE_SYNTHETIC_FIXTURE",
                        "target_test": "Sandboxed WASI Code Execution & Proof Verification",
                        "records_file": "examples/datasets/papers_with_code.json",
                        "resource_uri": "aetre://datasets/paperswithcode",
                        "description": "Fictional code artifacts and non-resolving URLs for parser tests.",
                        "key_fields": ["repository_url", "code_language", "claimed_throughput_speedup", "sandboxed_execution_status"]
                    },
                    {
                        "id": "arxiv_ssrn_live",
                        "name": "External Preprint Feed (Not Bundled)",
                        "source": "User-supplied records subject to source terms",
                        "target_test": "Optional ingestion integration",
                        "records_file": null,
                        "resource_uri": "aetre://datasets/arxiv_ssrn_live",
                        "description": "No arXiv or SSRN records are bundled. Users must supply appropriately licensed records.",
                        "key_fields": ["arxiv_id", "title", "abstract", "crowd_novelty_percentile", "predicted_triage_stream"]
                    }
                ]
            });

            return Ok(json!({
                "contents": [
                    {
                        "uri": uri,
                        "mimeType": "application/json",
                        "text": serde_json::to_string_pretty(&data).unwrap_or_default()
                    }
                ]
            }));
        }

        "aetre://schemas/database-writeback" => {
            let markdown_schema = r#"# AETRE Enterprise Database Write-Back Schema

## Direct SQL Integration (PostgreSQL, SQLite, Snowflake, BigQuery)

### Step 1: Add Triage Columns
```sql
ALTER TABLE submissions ADD COLUMN aetre_prior_mean REAL;
ALTER TABLE submissions ADD COLUMN aetre_variance REAL;
ALTER TABLE submissions ADD COLUMN aetre_novelty REAL;
ALTER TABLE submissions ADD COLUMN aetre_voi REAL;
ALTER TABLE submissions ADD COLUMN aetre_quality_rank INTEGER;
ALTER TABLE submissions ADD COLUMN aetre_voi_rank INTEGER;
ALTER TABLE submissions ADD COLUMN aetre_routing TEXT;
ALTER TABLE submissions ADD COLUMN aetre_evaluation_fingerprint TEXT;
ALTER TABLE submissions ADD COLUMN reviewed_at TIMESTAMP;
```

### Step 2: Batch Pipeline Connector
Run `python scripts/connect_external_db.py --sqlite enterprise_grants.db --boundary 1.20`.

### Step 3: Column Semantics
* `aetre_prior_mean`: Latent expected quality mu_0 in [-1.0, 3.0].
* `aetre_variance`: Epistemic uncertainty sigma_0^2 in [0.1, 1.5].
* `aetre_voi`: Marginal Value of Information boundary crossing gain.
* `aetre_quality_rank`: Global cohort ranking sorted by mu_0 desc.
* `aetre_voi_rank`: Review priority ranking sorted by VOI desc.
* `aetre_routing`: Stream A (Fast-Reject), Stream B (High-VOI Deep Review), or Stream C (Fast-Pass).
* `aetre_evaluation_fingerprint`: Reproducibility fingerprint for the input and score (not a signed receipt).
"#;

            return Ok(json!({
                "contents": [
                    {
                        "uri": uri,
                        "mimeType": "text/markdown",
                        "text": markdown_schema
                    }
                ]
            }));
        }

        "aetre://specs/mathematical-formulations" => {
            let math_specs = r#"# Mathematical & Operations Research Specifications (Gray, 2026)

## 1. Proposition 1: Throughput-Recall Ceiling
When candidate arrival volume N outpaces selection capacity K_N (K_N = o(N)):
$$R_N \le \min\left\{ 1, \frac{K_N}{H_N} \right\} \xrightarrow[N \to \infty]{} 0$$

## 2. Kingman Heavy-Traffic Approximation
Queue delay explodes non-linearly as utilization rho = lambda / mu approaches 1.0:
$$E[W_q] \approx \frac{\rho}{1-\rho} \cdot \frac{c_a^2 + c_s^2}{2} \cdot \frac{1}{\mu}$$
The Kingman Governor triggers automated triage whenever rho >= 0.85.

## 3. Gaussian Boundary Value-of-Information (VOI)
$$\text{VOI} = \int_{-\infty}^\infty \max(0, \mu' - \tau) \, p(\mu') \, d\mu' - \max(0, \mu_0 - \tau) - c_{\text{rev}}$$

## 4. Generalized Pareto Tail VOI (Power-Law Breakthroughs)
$$P(V > x) \propto x^{-\alpha}, \quad \alpha \in (1.0, 2.0]$$
Explicitly rewards high epistemic variance near boundary thresholds.

## 5. Correlated Multi-Agent Debiasing
$$M_{\text{eff}} = \frac{M}{1 + (M-1)\rho_{\text{corr}}}$$
Prevents artificial overconfidence from shared LLM pretraining bias.

## 6. Horvitz-Thompson Exploration Audits
$$\hat{H}_D = \sum_{i \in S_D} \frac{Y_i}{\pi_i} = \frac{N_D}{m_D} \cdot k_D$$
Unbiased recovery of false negatives from rejected candidate pools.

## 7. Anti-Sybil Quadratic Staking
$$\text{Stake}(m) = S_0 \cdot m^\gamma, \quad \gamma \ge 2.0$$
Escalates submission deposit requirements super-linearly to deter AI spam swarms.
"#;

            return Ok(json!({
                "contents": [
                    {
                        "uri": uri,
                        "mimeType": "text/markdown",
                        "text": math_specs
                    }
                ]
            }));
        }

        "aetre://institutional/tiers" => {
            let tiers_data = json!({
                "institutional_tiers": [
                    {
                        "tier": 1,
                        "name": "Researchers, Authors & Grant Applicants",
                        "inbound_flood": "Competitor preprints & grant drafts",
                        "bottleneck": "Reviewer consensus skepticism",
                        "core_tool": "aetre_author_preflight_benchmark",
                        "impact": "Eliminates blind rejections; provides empirical flight plan."
                    },
                    {
                        "tier": 2,
                        "name": "Venture Capital & DeepTech Angel Funds",
                        "inbound_flood": "Pitch decks & startup applications (5,000+/year)",
                        "bottleneck": "Partner consensus arithmetic averaging",
                        "core_tool": "aetre_heavy_tailed_voi",
                        "impact": "Catches 100x fund-returning positive black swans."
                    },
                    {
                        "tier": 3,
                        "name": "Academic Publishers & Conference Committees",
                        "inbound_flood": "Conference & journal submissions (NeurIPS, ICLR)",
                        "bottleneck": "Finite volunteer reviewer pool",
                        "core_tool": "aetre_correlated_posterior_update, aetre_check_governor",
                        "impact": "Prevents queue saturation; debiases AI reviewer panels."
                    },
                    {
                        "tier": 4,
                        "name": "Government Grant Agencies & Sovereign R&D",
                        "inbound_flood": "Grant proposals (NIH R01, NSF, DARPA, ARIA)",
                        "bottleneck": "Study section payline bandwidth",
                        "core_tool": "aetre_exploration_audit, aetre_calculate_voi",
                        "impact": "Unbiased discovery of overlooked breakthrough science."
                    },
                    {
                        "tier": 5,
                        "name": "Patent Offices & Intellectual Property Regulators",
                        "inbound_flood": "Synthetic patent claim filings (USPTO, EPO)",
                        "bottleneck": "Examiner pendency & time per claim",
                        "core_tool": "aetre_heterogeneous_queues",
                        "impact": "Resolves multi-year backlogs; protects true prior art."
                    },
                    {
                        "tier": 6,
                        "name": "Corporate R&D Portfolios & University TTOs",
                        "inbound_flood": "Internal invention disclosures",
                        "bottleneck": "Phase 1 / Phase 2 validation capital",
                        "core_tool": "aetre_triage_proposal",
                        "impact": "Optimizes multi-million dollar R&D budget allocation."
                    },
                    {
                        "tier": 7,
                        "name": "Startup Accelerators & Grand Challenge Prizes",
                        "inbound_flood": "Open online prize applications (25,000+ apps)",
                        "bottleneck": "Admissions screening capacity",
                        "core_tool": "aetre_quadratic_staking",
                        "impact": "Stops automated AI application swarms with 0 friction."
                    }
                ]
            });

            return Ok(json!({
                "contents": [
                    {
                        "uri": uri,
                        "mimeType": "application/json",
                        "text": serde_json::to_string_pretty(&tiers_data).unwrap_or_default()
                    }
                ]
            }));
        }

        _ => {}
    }

    // 2. Dynamic Template Resolution: aetre://datasets/{name}
    if let Some(ds_name) = uri.strip_prefix("aetre://datasets/") {
        let clean_name = ds_name.trim_end_matches(".json").to_lowercase();
        let content_str = match clean_name.as_str() {
            "nih" | "nih_grant_proposals" => try_read_file_or_fallback(
                &[
                    "examples/datasets/nih_grant_proposals.json",
                    "datasets/nih_grant_proposals.json",
                ],
                json!([
                    { "id": "NIH-R01-CA294810", "title": "Epigenetic Reprogramming of Glioblastoma", "requested_budget_usd": 1850000, "initial_priority_percentile": 14.5, "epistemic_variance": 0.82 }
                ]),
            ),
            "openreview" | "openreview_peer_review" | "peerread" => try_read_file_or_fallback(
                &[
                    "examples/datasets/openreview_peer_review.json",
                    "datasets/openreview_peer_review.json",
                ],
                json!([
                    { "id": "ICLR-2026-Sub-841", "title": "Equivariant Graph Neural Diffusion on Non-Euclidean Manifolds", "review_scores": [8.0, 3.0, 7.0], "variance": 0.74 }
                ]),
            ),
            "uspto" | "uspto_patent_applications" | "patents" => try_read_file_or_fallback(
                &[
                    "examples/datasets/uspto_patent_applications.json",
                    "datasets/uspto_patent_applications.json",
                ],
                json!([
                    { "application_id": "US18/924,102", "title": "Solid-State Polymer-Ceramic Electrolyte Matrix", "cpc_class": "H01M", "examiner_utilization_rho": 0.94 }
                ]),
            ),
            "paperswithcode" | "papers_with_code" => try_read_file_or_fallback(
                &[
                    "examples/datasets/papers_with_code.json",
                    "datasets/papers_with_code.json",
                ],
                json!([
                    { "paper_title": "Fast Sub-Quadratic Attention via Block-Sparse Approximations", "repository_url": "https://github.com/aetre-bench/sparse-attn", "sandboxed_execution_status": "Verified" }
                ]),
            ),
            "arxiv" | "arxiv_ssrn_live" | "ssrn" => json!({
                "status": "not_bundled",
                "message": "Supply records only after reviewing the source terms and paper licenses."
            })
            .to_string(),
            _ => try_read_file_or_fallback(
                &[
                    &format!("examples/datasets/{}.json", clean_name),
                    &format!("examples/{}.json", clean_name),
                ],
                json!({ "dataset": clean_name, "status": "custom_dataset", "records": [] }),
            ),
        };

        return Ok(json!({
            "contents": [
                {
                    "uri": uri,
                    "mimeType": "application/json",
                    "text": content_str
                }
            ]
        }));
    }

    // 3. Dynamic Template Resolution: aetre://proposals/{id}
    if let Some(proposal_id) = uri.strip_prefix("aetre://proposals/") {
        let proposal_data = json!({
            "proposal_id": proposal_id,
            "evaluation_fingerprint": format!("aetre-eval-demo-{:x}", Sha256::digest(proposal_id.as_bytes())),
            "title": format!("Proposal Benchmark ({})", proposal_id),
            "status": "SYNTHETIC_EXAMPLE_EVALUATED",
            "protocol": "AETRE deterministic evaluation example; not a signed receipt",
            "retrieval_uri": uri
        });

        return Ok(json!({
            "contents": [
                {
                    "uri": uri,
                    "mimeType": "application/json",
                    "text": serde_json::to_string_pretty(&proposal_data).unwrap_or_default()
                }
            ]
        }));
    }

    Err(format!("Resource with URI '{}' not found", uri))
}

pub fn list_prompts() -> Value {
    json!([
        {
            "name": "author_preflight_review",
            "description": "Pre-submission diagnostic flight simulator: benchmarks paper draft against crowd distributions, flags reviewer split risks, and provides a prescriptive variance reduction plan.",
            "arguments": [
                {
                    "name": "title",
                    "description": "Title of the research paper or grant proposal.",
                    "required": true
                },
                {
                    "name": "abstract",
                    "description": "Abstract, executive summary, or proposal body text.",
                    "required": true
                },
                {
                    "name": "boundary",
                    "description": "Selection or payline threshold (default: 1.2).",
                    "required": false
                }
            ]
        },
        {
            "name": "pipeline_congestion_audit",
            "description": "Audits review pipeline traffic intensity, wait times, and backlog under Kingman's Heavy-Traffic approximation.",
            "arguments": [
                {
                    "name": "arrival_rate",
                    "description": "Proposals arriving per period (lambda).",
                    "required": true
                },
                {
                    "name": "service_rate",
                    "description": "Review capacity of the system per period (mu).",
                    "required": true
                },
                {
                    "name": "target_utilization",
                    "description": "Target sustainable utilization ceiling (default: 0.85).",
                    "required": false
                }
            ]
        },
        {
            "name": "multi_agent_panel_debiasing",
            "description": "Debiases multi-LLM reviewer panels by computing effective evaluator sample size (M_eff) under shared training correlation.",
            "arguments": [
                {
                    "name": "scores",
                    "description": "Comma-separated scores from LLM evaluators (e.g. '1.6, 1.8, 1.5').",
                    "required": true
                },
                {
                    "name": "correlation",
                    "description": "Inter-agent error correlation rho in [0, 1) (default: 0.6).",
                    "required": false
                }
            ]
        }
    ])
}

pub fn get_prompt(name: &str, args: Value) -> Result<Value, String> {
    match name {
        "author_preflight_review" => {
            let title = args
                .get("title")
                .and_then(|v| v.as_str())
                .unwrap_or("Untitled Proposal");
            let text = args.get("abstract").and_then(|v| v.as_str()).unwrap_or("");
            let boundary = args
                .get("boundary")
                .and_then(|v| v.as_str())
                .unwrap_or("1.2");

            let prompt_text = format!(
                "You are evaluating a research or grant proposal draft before official submission:\n\n**Title**: {}\n**Abstract / Summary**:\n\"\"\"\n{}\n\"\"\"\n\nPlease run the `aetre_author_preflight_benchmark` tool with `selection_boundary = {}` to perform empirical crowd benchmarking, evaluate reviewer split risk, and generate a prescriptive epistemic flight plan.",
                title, text, boundary
            );

            Ok(json!({
                "description": "Pre-submission benchmark diagnostic and variance reduction flight simulator.",
                "messages": [
                    {
                        "role": "user",
                        "content": {
                            "type": "text",
                            "text": prompt_text
                        }
                    }
                ]
            }))
        }

        "pipeline_congestion_audit" => {
            let arrival_rate = args
                .get("arrival_rate")
                .and_then(|v| v.as_str())
                .unwrap_or("95.0");
            let service_rate = args
                .get("service_rate")
                .and_then(|v| v.as_str())
                .unwrap_or("100.0");
            let target = args
                .get("target_utilization")
                .and_then(|v| v.as_str())
                .unwrap_or("0.85");

            let prompt_text = format!(
                "Please evaluate our evaluation pipeline capacity using AETRE's Kingman Heavy-Traffic Governor (`aetre_check_governor`):\n- Arrival Rate (lambda): {} arrivals/period\n- Service Capacity (mu): {} reviews/period\n- Target Utilization Ceiling: {}\n\nPlease analyze whether the system is at risk of delay explosion and recommend the exact automated triage throttling required to stabilize reviewer workload.",
                arrival_rate, service_rate, target
            );

            Ok(json!({
                "description": "Kingman heavy-traffic queue congestion audit and throttle recommendations.",
                "messages": [
                    {
                        "role": "user",
                        "content": {
                            "type": "text",
                            "text": prompt_text
                        }
                    }
                ]
            }))
        }

        "multi_agent_panel_debiasing" => {
            let scores_str = args
                .get("scores")
                .and_then(|v| v.as_str())
                .unwrap_or("1.6, 1.8, 1.5");
            let corr = args
                .get("correlation")
                .and_then(|v| v.as_str())
                .unwrap_or("0.6");

            let prompt_text = format!(
                "We collected evaluations from multiple LLM evaluators on a candidate proposal with raw scores: [{}].\nAssuming an inter-agent error correlation rho = {}:\n\nUse `aetre_correlated_posterior_update` to calculate:\n1. The effective evaluator sample size (M_eff)\n2. The redundancy correlation discount percentage\n3. The true debiased posterior mean and epistemic variance.",
                scores_str, corr
            );

            Ok(json!({
                "description": "Debiasing multi-LLM reviewer panels against shared correlation.",
                "messages": [
                    {
                        "role": "user",
                        "content": {
                            "type": "text",
                            "text": prompt_text
                        }
                    }
                ]
            }))
        }

        _ => Err(format!("Prompt with name '{}' not found", name)),
    }
}

pub fn list_tools() -> Value {
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
            "name": "aetre_heldout_backtest",
            "description": "Runs a multi-policy held-out review allocation backtest across 8 triage policies under fixed review budget K, evaluating true decision flips, precision, recall, and paired bootstrap intervals.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "dataset": {
                        "type": "string",
                        "description": "Dataset identifier or path (default: 'openreview')."
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
        }
    ])
}

pub fn call_tool(name: &str, args: Value) -> Value {
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
                    "version": "0.1.0",
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
                "crowd_novelty_percentile": format!("Top {:.1}%", (1.0 - novelty) * 100.0),
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
            if tier != LicenseTier::Enterprise {
                let paywall = generate_tier_locked_payload(
                    "aetre_check_governor",
                    "Kingman Heavy-Traffic Capacity Governor & Auto-Throttling",
                    "Grant Agencies & Academic Publishers ($25,000–$250,000/yr)",
                );
                return json!({
                    "content": [
                        {
                            "type": "text",
                            "text": serde_json::to_string_pretty(&paywall).unwrap_or_default()
                        }
                    ],
                    "isError": false
                });
            }

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
            if tier != LicenseTier::Enterprise {
                let paywall = generate_tier_locked_payload(
                    "aetre_exploration_audit",
                    "5% Horvitz-Thompson Counterfactual Exploration Audits",
                    "Government Grant Agencies & Sovereign R&D Funds ($100,000–$500,000/yr)",
                );
                return json!({
                    "content": [
                        {
                            "type": "text",
                            "text": serde_json::to_string_pretty(&paywall).unwrap_or_default()
                        }
                    ],
                    "isError": false
                });
            }

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
            if tier != LicenseTier::Enterprise {
                let paywall = generate_tier_locked_payload(
                    "aetre_evaluate_staking",
                    "Endogenous Submitter Entry Equilibrium Simulation",
                    "Startup Accelerators & Grand Challenge Prizes ($15,000–$50,000/yr)",
                );
                return json!({
                    "content": [
                        {
                            "type": "text",
                            "text": serde_json::to_string_pretty(&paywall).unwrap_or_default()
                        }
                    ],
                    "isError": false
                });
            }

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
            if tier != LicenseTier::Enterprise {
                let paywall = generate_tier_locked_payload(
                    "aetre_correlated_posterior_update",
                    "Multi-Agent Reviewer Debiasing & Correlation Removal",
                    "Publishers, Conferences & Institutional Enterprise ($25,000–$250,000/yr)",
                );
                return json!({
                    "content": [
                        {
                            "type": "text",
                            "text": serde_json::to_string_pretty(&paywall).unwrap_or_default()
                        }
                    ],
                    "isError": false
                });
            }

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
            if tier != LicenseTier::Enterprise {
                let paywall = generate_tier_locked_payload(
                    "aetre_heavy_tailed_voi",
                    "Heavy-Tailed Pareto Black Swan Discovery",
                    "VC & Institutional Enterprise ($15,000–$40,000/yr)",
                );
                return json!({
                    "content": [
                        {
                            "type": "text",
                            "text": serde_json::to_string_pretty(&paywall).unwrap_or_default()
                        }
                    ],
                    "isError": false
                });
            }

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
            if tier != LicenseTier::Enterprise {
                let paywall = generate_tier_locked_payload(
                    "aetre_quadratic_staking",
                    "Super-Linear Anti-Sybil Quadratic Staking",
                    "Startup Accelerators & Grand Challenge Prizes ($15,000–$50,000/yr)",
                );
                return json!({
                    "content": [
                        {
                            "type": "text",
                            "text": serde_json::to_string_pretty(&paywall).unwrap_or_default()
                        }
                    ],
                    "isError": false
                });
            }

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
            if tier != LicenseTier::Enterprise {
                let paywall = generate_tier_locked_payload(
                    "aetre_heterogeneous_queues",
                    "Heterogeneous Specialist Reviewer Queue Balancer",
                    "Patent Offices & Corporate R&D ($75,000–$250,000/yr)",
                );
                return json!({
                    "content": [
                        {
                            "type": "text",
                            "text": serde_json::to_string_pretty(&paywall).unwrap_or_default()
                        }
                    ],
                    "isError": false
                });
            }

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

            let month = current_year_month();
            let current_usage = get_preflight_usage(&month);

            if tier == LicenseTier::Community && current_usage >= COMMUNITY_PREFLIGHT_LIMIT {
                let stream_pred = if prior_mean >= boundary {
                    "FAST-PASS: DIRECT PHASE 2"
                } else if prior_var > 0.4 {
                    "HIGH VOI: DEEP REVIEW QUEUE"
                } else {
                    "FAST-REJECT / SPAM FILTER"
                };

                let paywall_payload =
                    generate_quota_exceeded_payload(title, prior_mean, prior_var, stream_pred);

                return json!({
                    "content": [
                        {
                            "type": "text",
                            "text": serde_json::to_string_pretty(&paywall_payload).unwrap_or_default()
                        }
                    ],
                    "isError": false
                });
            }

            let new_usage = if tier == LicenseTier::Community {
                increment_preflight_usage(&month)
            } else {
                current_usage
            };

            let report = evaluate_author_preflight(title, prior_mean, prior_var, novelty, boundary);

            let remaining_checks = if tier == LicenseTier::Community {
                COMMUNITY_PREFLIGHT_LIMIT.saturating_sub(new_usage)
            } else {
                999999
            };

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
                "crowd_novelty_percentile": format!("Top {:.1}%", 100.0 - report.crowd_novelty_percentile),
                "reviewer_disagreement_risk": report.reviewer_disagreement_risk,
                "predicted_triage_stream": report.predicted_triage_stream,
                "voi_index": report.voi_index,
                "prescriptive_action_plan": report.prescriptive_action_plan,
                "variance_reduction_target": report.variance_reduction_target,
                "evaluation_fingerprint": report.evaluation_fingerprint,
                "markdown_badge": report.markdown_badge,
                "epistemic_diagnostics": diagnostics,
                "markdown_scorecard": md_scorecard,
                "license_tier": tier.as_str(),
                "monthly_quota_info": {
                    "tier": tier.display_name(),
                    "checks_used_this_month": new_usage,
                    "checks_remaining": if tier == LicenseTier::Community { format!("{}/{}", remaining_checks, COMMUNITY_PREFLIGHT_LIMIT) } else { "UNLIMITED (Pro/Enterprise)".to_string() }
                }
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
            if tier != LicenseTier::Enterprise {
                let paywall = generate_tier_locked_payload(
                    "aetre_simulate_benchmark",
                    "Monte Carlo Multi-Regime Benchmark Simulator",
                    "VC & Institutional Enterprise ($15,000–$250,000/yr)",
                );
                return json!({
                    "content": [
                        {
                            "type": "text",
                            "text": serde_json::to_string_pretty(&paywall).unwrap_or_default()
                        }
                    ],
                    "isError": false
                });
            }

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
                // Tier limit check: Community/Pro limited to 10 batch items, Enterprise unlimited
                let limit = if tier == LicenseTier::Enterprise {
                    5000
                } else {
                    10
                };
                let items_to_process = &items[..items.len().min(limit)];

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

        "aetre_heldout_backtest" => {
            let budget = get_usize(&args, "budget", 50);
            let boundary = get_f64(&args, "boundary", 6.0);
            let split = get_str(&args, "split", "test");
            let dataset_str = get_str(&args, "dataset", "openreview");

            let candidates_files = [
                dataset_str,
                "examples/datasets/openreview_heldout_backtest.json",
                "../../examples/datasets/openreview_heldout_backtest.json",
                "data/normalized/openreview_normalized.json",
            ];
            let mut file_path = None;
            for c in candidates_files {
                let p = std::path::PathBuf::from(c);
                if p.exists() {
                    file_path = Some(p);
                    break;
                }
            }

            if let Some(p) = file_path {
                let raw = match std::fs::read(&p) {
                    Ok(b) => b,
                    Err(e) => {
                        return json!({
                            "content": [{ "type": "text", "text": format!("Error reading dataset: {}", e) }],
                            "isError": true
                        });
                    }
                };

                #[allow(dead_code)]
                #[derive(serde::Deserialize)]
                struct TempCandidate {
                    id: String,
                    split: String,
                    label: u8,
                    pre_triage_data: PreTriageRaw,
                }
                #[allow(dead_code)]
                #[derive(serde::Deserialize)]
                struct PreTriageRaw {
                    preliminary_mean: f64,
                    preliminary_variance: f64,
                    m_reviews_count: Option<usize>,
                    preliminary_mean_confidence: Option<f64>,
                }

                if let Ok(records) = serde_json::from_slice::<Vec<TempCandidate>>(&raw) {
                    let eval_records: Vec<&TempCandidate> = records
                        .iter()
                        .filter(|r| r.split == split || split == "all")
                        .collect();
                    let n = eval_records.len();
                    let total_pos = eval_records.iter().filter(|r| r.label == 1).count();

                    let mut aetre_scores: Vec<(usize, f64)> = eval_records
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
                            let voi = aetre_core::calculate_boundary_voi(
                                m, post_var, boundary, sig_noise, 0.50,
                            );
                            (idx, voi)
                        })
                        .collect();

                    aetre_scores
                        .sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
                    let k_eff = budget.min(n);
                    let mut aetre_tp = 0;
                    for &(idx, _) in &aetre_scores[..k_eff] {
                        if eval_records[idx].label == 1 {
                            aetre_tp += 1;
                        }
                    }

                    let recall = aetre_tp as f64 / total_pos.max(1) as f64;
                    let precision = aetre_tp as f64 / k_eff.max(1) as f64;

                    let out = json!({
                        "dataset": p.to_string_lossy(),
                        "evaluation_split": split,
                        "candidates_evaluated": n,
                        "true_decision_flips": total_pos,
                        "budget_allocated_K": k_eff,
                        "aetre_voi_precision_at_k": (precision * 1000.0).round() / 10.0,
                        "aetre_voi_recall_at_k": (recall * 1000.0).round() / 10.0,
                        "aetre_discoveries_caught": aetre_tp,
                        "reviewer_hours_per_discovery": if aetre_tp > 0 { (k_eff as f64 * 4.0) / aetre_tp as f64 } else { k_eff as f64 * 4.0 },
                        "status": "BACKTEST_EVALUATED_SUCCESSFULLY"
                    });

                    json!({
                        "content": [{ "type": "text", "text": serde_json::to_string_pretty(&out).unwrap_or_default() }],
                        "isError": false
                    })
                } else {
                    json!({
                        "content": [{ "type": "text", "text": "Failed to parse backtest dataset JSON schema" }],
                        "isError": true
                    })
                }
            } else {
                json!({
                    "content": [{ "type": "text", "text": format!("Could not locate dataset file: {}", dataset_str) }],
                    "isError": true
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
            if tier != LicenseTier::Enterprise {
                let paywall = generate_tier_locked_payload(
                    "aetre_congestion_matching",
                    "Congestion-Aware Reviewer-to-Proposal Matching & Kingman Load Balancer",
                    "Enterprise Conference & Grant Agencies ($25,000–$250,000/yr)",
                );
                return json!({
                    "content": [
                        {
                            "type": "text",
                            "text": serde_json::to_string_pretty(&paywall).unwrap_or_default()
                        }
                    ],
                    "isError": false
                });
            }

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
mod tests {
    use super::*;

    #[test]
    fn test_list_tools() {
        let tools = list_tools();
        let arr = tools.as_array().unwrap();
        assert_eq!(arr.len(), 20);
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
        let text_locked = res_locked["content"][0]["text"].as_str().unwrap();
        assert!(text_locked.contains("TIER_LOCKED"));

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
        assert!(res_locked["content"][0]["text"]
            .as_str()
            .unwrap()
            .contains("TIER_LOCKED"));

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
        assert!(res_locked["content"][0]["text"]
            .as_str()
            .unwrap()
            .contains("TIER_LOCKED"));

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
        assert!(res_locked["content"][0]["text"]
            .as_str()
            .unwrap()
            .contains("TIER_LOCKED"));

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
}
