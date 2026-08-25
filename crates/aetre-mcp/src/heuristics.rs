//! Transparent lexical indicators used by the demonstration MCP tools.
//!
//! These values are routing features, not calibrated estimates of scientific quality.
//! Production users should replace or calibrate this module on a frozen, time-split corpus.

use serde::{Deserialize, Serialize};
use std::collections::HashSet;

pub const SCORING_METHOD: &str = "transparent_lexical_heuristic_v1";
pub const CALIBRATION_STATUS: &str = "UNCALIBRATED_RESEARCH_PROTOTYPE";

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EpistemicDiagnostics {
    pub scoring_method: String,
    pub calibration_status: String,
    pub prior_mean: f64,
    pub prior_variance: f64,
    pub novelty_score: f64,
    pub empirical_rigor_score: f64,
    pub novelty_density: f64,
    pub uncertainty_ambiguity_index: f64,
    pub wrapper_risk_score: f64,
    pub vocabulary_diversity: f64,
    pub detected_frontier_keywords: Vec<String>,
}

fn defaults(prior_mean: f64, prior_variance: f64, novelty_score: f64) -> EpistemicDiagnostics {
    EpistemicDiagnostics {
        scoring_method: SCORING_METHOD.to_string(),
        calibration_status: CALIBRATION_STATUS.to_string(),
        prior_mean,
        prior_variance,
        novelty_score,
        empirical_rigor_score: 0.0,
        novelty_density: 0.0,
        uncertainty_ambiguity_index: 0.0,
        wrapper_risk_score: 0.0,
        vocabulary_diversity: 0.0,
        detected_frontier_keywords: Vec::new(),
    }
}

/// Returns explainable, uncalibrated lexical routing indicators.
/// Feature-family contributions are capped to reduce keyword-stacking attacks.
pub fn analyze_text_heuristics(text: &str) -> EpistemicDiagnostics {
    if text.is_empty() {
        return defaults(0.0, 0.5, 0.2);
    }

    let lower = text.to_lowercase();
    let words: Vec<&str> = lower
        .split(|c: char| !c.is_alphabetic())
        .filter(|word| word.len() >= 3)
        .collect();
    if words.len() < 5 {
        let mut result = defaults(0.2, 0.4, 0.1);
        result.vocabulary_diversity = 0.2;
        return result;
    }

    let unique_words: HashSet<&str> = words.iter().copied().collect();
    let diversity = unique_words.len() as f64 / words.len() as f64;
    let novelty_keywords = [
        "quantum",
        "qubit",
        "variational eigensolver",
        "vqe",
        "superconducting",
        "topological",
        "qaoa",
        "optically",
        "transport",
        "non-linear",
        "breakthrough",
        "paradigm",
        "non-gaussian",
        "novel",
        "synthetic",
        "frontier",
        "mechanistic",
        "circuit",
        "non-markovian",
        "monolithic",
        "epigenetic",
        "crispr",
        "cas9",
        "micro-rna",
        "mrna",
        "protein folding",
        "allosteric",
        "immunotherapy",
        "gene therapy",
        "car-t",
        "aptamer",
        "biosensor",
        "atomic structure",
        "semiconductor",
        "heterojunction",
        "electrolytes",
        "dendrite",
        "solid-state",
        "perovskite",
        "photovoltaic",
        "supercapacitor",
        "hydrogen storage",
        "transformer",
        "attention mechanism",
        "residual learning",
        "diffusion",
        "rlhf",
        "multi-agent",
        "mixture-of-experts",
        "state space model",
        "mamba",
        "symbolic regression",
        "zero-knowledge",
        "zk-snark",
        "zk-stark",
        "homomorphic",
        "proof-of-work",
        "proof-of-stake",
        "peer-to-peer",
        "fault-tolerance",
        "queueing",
        "kingman",
        "innovation-absorption",
        "horvitz-thompson",
    ];
    let rigor_keywords = [
        "bleu",
        "imagenet",
        "casp",
        "rmsd",
        "state-of-the-art",
        "sota",
        "1st place",
        "benchmark",
        "experiments show",
        "empirical evidence",
        "theorem",
        "we prove",
        "proof",
        "exact",
        "falsifiable",
        "ablation",
        "confidence interval",
        "p-value",
        "statistical significance",
    ];
    let wrapper_keywords = [
        "wrapper",
        "prompt-chaining",
        "prompt chaining",
        "zendesk",
        "salesforce",
        "boilerplate",
        "routine",
        "plugin",
        "airdrop",
        "crypto token",
        "whitepaper",
        "bag-of-words",
        "tf-idf",
        "scikit-learn",
        "small csv",
        "streamlit",
        "langchain",
        "basic ui",
        "simple app",
    ];
    let uncertainty_keywords = [
        "preliminary",
        "ambiguous",
        "hypothesis",
        "variance",
        "validation",
        "uncertain",
        "dry-room",
        "hplc",
        "disagreement",
        "noisy",
        "disputed",
        "simulation",
        "testing",
        "kinetics",
        "dft",
        "density functional",
        "proof of concept",
        "exploratory",
        "in-vitro",
        "pilot study",
    ];

    let all_detected: Vec<String> = novelty_keywords
        .iter()
        .filter(|&&keyword| lower.contains(keyword))
        .map(|keyword| (*keyword).to_string())
        .collect();
    let novelty_hits = all_detected.len().min(3) as f64;
    let count_capped = |keywords: &[&str]| {
        keywords
            .iter()
            .filter(|&&keyword| lower.contains(keyword))
            .count()
            .min(3) as f64
    };
    let rigor_hits = count_capped(&rigor_keywords);
    let wrapper_hits = count_capped(&wrapper_keywords);
    let uncertainty_hits = count_capped(&uncertainty_keywords);

    let novelty = (0.20 + novelty_hits * 0.18 + rigor_hits * 0.08 - wrapper_hits * 0.15
        + diversity * 0.20)
        .clamp(0.05, 0.95);
    let prior_variance = (0.25 + uncertainty_hits * 0.20 + novelty_hits * 0.05
        - rigor_hits * 0.08
        - wrapper_hits * 0.05)
        .clamp(0.10, 1.20);
    let mut raw_mean =
        0.50 + novelty_hits * 0.35 + rigor_hits * 0.35 - wrapper_hits * 0.45 + diversity * 0.35;
    if wrapper_hits >= 2.0 && novelty_hits == 0.0 {
        raw_mean -= 0.70;
    }

    EpistemicDiagnostics {
        scoring_method: SCORING_METHOD.to_string(),
        calibration_status: CALIBRATION_STATUS.to_string(),
        prior_mean: raw_mean.clamp(-0.80, 2.20),
        prior_variance,
        novelty_score: novelty,
        empirical_rigor_score: (rigor_hits * 0.25).clamp(0.0, 1.0),
        novelty_density: (novelty_hits * 0.20).clamp(0.0, 1.0),
        uncertainty_ambiguity_index: (uncertainty_hits * 0.25).clamp(0.0, 1.0),
        wrapper_risk_score: (wrapper_hits * 0.35).clamp(0.0, 1.0),
        vocabulary_diversity: (diversity * 100.0).round() / 100.0,
        detected_frontier_keywords: all_detected.into_iter().take(8).collect(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn diagnostics_disclose_uncalibrated_method() {
        let result = analyze_text_heuristics("A sufficiently long ordinary proposal description");
        assert_eq!(result.scoring_method, SCORING_METHOD);
        assert_eq!(result.calibration_status, CALIBRATION_STATUS);
    }

    #[test]
    fn feature_family_contributions_are_capped() {
        let three = analyze_text_heuristics(
            "quantum qubit topological study with a controlled experimental design and results",
        );
        let many = analyze_text_heuristics(
            "quantum qubit topological transformer diffusion frontier mechanistic circuit study with a controlled experimental design and results",
        );
        assert_eq!(three.novelty_density, many.novelty_density);
    }
}
