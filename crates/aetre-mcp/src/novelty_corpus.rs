//! Generator for the bundled novelty reference distribution.
//!
//! This lives behind `--emit-novelty-reference` in the shipped binary rather than
//! in a separate script on purpose: the quantiles must be produced by the exact
//! `analyze_text_heuristics` the runtime uses, or the percentile is a rank against
//! a distribution nothing is actually drawn from. Reimplementing the scorer in a
//! second language would reintroduce precisely that drift.

use crate::heuristics::{analyze_text_heuristics, SCORING_METHOD};
use aetre_core::novelty_reference::{build_quantiles, QUANTILE_COUNT, REFERENCE_SCHEMA};
use serde_json::{json, Value};

/// Field names searched, in order, for the text of each corpus record.
const TEXT_FIELDS: [&str; 4] = ["abstract_text", "abstract", "text", "summary"];

/// Reads a corpus of documents and prints the reference distribution as JSON.
///
/// The corpus is a JSON array of objects, each carrying its text in one of
/// [`TEXT_FIELDS`]. Exits non-zero with a message on any failure: emitting a
/// reference built from fewer documents than the caller supplied would silently
/// weaken every percentile computed against it.
pub fn emit(path: &str, corpus_id: &str, description: &str) -> Result<String, String> {
    let raw = std::fs::read_to_string(path)
        .map_err(|err| format!("could not read corpus at {path}: {err}"))?;
    let parsed: Value = serde_json::from_str(&raw)
        .map_err(|err| format!("corpus at {path} is not valid JSON: {err}"))?;
    let records = parsed
        .as_array()
        .ok_or_else(|| format!("corpus at {path} must be a JSON array of objects"))?;

    let mut scores: Vec<f64> = Vec::with_capacity(records.len());
    let mut skipped = 0usize;
    for record in records {
        match TEXT_FIELDS
            .iter()
            .find_map(|field| record.get(field).and_then(Value::as_str))
        {
            Some(text) if !text.trim().is_empty() => {
                scores.push(analyze_text_heuristics(text).novelty_score);
            }
            _ => skipped += 1,
        }
    }

    if scores.len() < QUANTILE_COUNT {
        return Err(format!(
            "corpus yielded only {} scorable documents; a reference needs at least {} \
             so that each stored quantile rests on a distinct observation",
            scores.len(),
            QUANTILE_COUNT
        ));
    }
    if skipped > 0 {
        eprintln!(
            "NOTE: skipped {skipped} record(s) with no text in any of {TEXT_FIELDS:?}; \
             the reference describes the {} that were scored.",
            scores.len()
        );
    }

    let n = scores.len() as u64;
    let quantiles = build_quantiles(&mut scores);
    let rounded: Vec<f64> = quantiles
        .iter()
        .map(|value| (value * 100000.0).round() / 100000.0)
        .collect();

    let document = json!({
        "schema": REFERENCE_SCHEMA,
        "corpus": corpus_id,
        "corpus_description": description,
        "scoring_method": SCORING_METHOD,
        "n": n,
        "quantile_step_percent": 100.0 / (QUANTILE_COUNT - 1) as f64,
        "quantiles": rounded,
    });

    serde_json::to_string_pretty(&document)
        .map_err(|err| format!("could not serialise the reference: {err}"))
}
