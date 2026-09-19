use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::fs;
use std::path::PathBuf;

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

pub(crate) fn try_read_file_or_fallback(paths: &[&str], fallback_json: Value) -> String {
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
                        "target_test": "Reported Code-Artifact Status Parsing",
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
