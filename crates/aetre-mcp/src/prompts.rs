use serde_json::{json, Value};

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
