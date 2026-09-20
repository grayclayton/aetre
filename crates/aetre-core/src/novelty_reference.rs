//! Empirical reference distribution for lexical novelty scores.
//!
//! A percentile is only meaningful relative to a population. Earlier versions of
//! AETRE reported a "crowd novelty percentile" that was either a linear rescale of
//! the raw score or the CDF of an invented `N(0.35, 0.20^2)` distribution. Neither
//! was a rank against anything, so the number could not be audited and should not
//! have been read as one.
//!
//! This module replaces that with a measured distribution: the bundled quantiles
//! are the scores `analyze_text_heuristics` actually assigns to a fixed corpus of
//! real submission abstracts. `novelty_percentile` is then an ordinary empirical
//! CDF lookup, and every reported figure carries the corpus and sample size it was
//! computed against.
//!
//! Regenerate the bundled file with:
//!
//! ```text
//! aetre-mcp --emit-novelty-reference <corpus.json>
//! ```

use serde::{Deserialize, Serialize};
use std::sync::OnceLock;

/// The bundled reference, embedded at compile time.
pub const REFERENCE_JSON: &str = include_str!("../reference/novelty_reference_v1.json");

pub const REFERENCE_SCHEMA: &str = "aetre-novelty-reference-v1";

/// Number of stored quantile breakpoints: one per 0.1 percentile, inclusive of both ends.
pub const QUANTILE_COUNT: usize = 1001;

/// A distribution of novelty scores measured over a named corpus.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NoveltyReference {
    pub schema: String,
    /// Stable identifier for the corpus the quantiles were measured on.
    pub corpus: String,
    /// Human-readable description, reported alongside every percentile.
    pub corpus_description: String,
    /// The scoring function whose output was measured. A percentile is only valid
    /// for scores produced by this same method.
    pub scoring_method: String,
    /// Number of documents scored.
    pub n: u64,
    /// Percentile spacing between adjacent `quantiles` entries.
    pub quantile_step_percent: f64,
    /// `quantiles[i]` is the score at percentile `i * quantile_step_percent`,
    /// ascending. Length is [`QUANTILE_COUNT`].
    pub quantiles: Vec<f64>,
}

/// A rank for one score against a named reference corpus.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NoveltyPercentile {
    /// Share of the corpus scoring at or below this score, 0.0 to 100.0.
    pub percentile: f64,
    /// `100.0 - percentile`: the share of the corpus scoring higher.
    pub top_percent: f64,
    pub corpus: String,
    pub corpus_description: String,
    pub reference_n: u64,
    pub scoring_method: String,
}

impl NoveltyPercentile {
    /// Prose form, e.g. `"Top 34.2% of 6413 openreview-iclr-neurips"`.
    pub fn label(&self) -> String {
        format!(
            "Top {:.1}% of {} {}",
            self.top_percent, self.reference_n, self.corpus
        )
    }
}

fn parse_reference() -> Option<&'static NoveltyReference> {
    static CACHE: OnceLock<Option<NoveltyReference>> = OnceLock::new();
    CACHE
        .get_or_init(|| {
            let parsed: NoveltyReference = serde_json::from_str(REFERENCE_JSON).ok()?;
            if parsed.schema != REFERENCE_SCHEMA || parsed.quantiles.len() != QUANTILE_COUNT {
                return None;
            }
            Some(parsed)
        })
        .as_ref()
}

/// The bundled reference distribution, or `None` if it is absent or malformed.
pub fn reference() -> Option<&'static NoveltyReference> {
    parse_reference()
}

/// Ranks `score` against the bundled reference corpus.
///
/// Returns `None` when no usable reference is bundled. Callers must report the
/// absence rather than substituting an uncalibrated number: a percentile with no
/// population behind it is the defect this module exists to remove.
pub fn novelty_percentile(score: f64) -> Option<NoveltyPercentile> {
    let reference = parse_reference()?;
    let percentile = percentile_in(&reference.quantiles, reference.quantile_step_percent, score);
    Some(NoveltyPercentile {
        percentile: (percentile * 10.0).round() / 10.0,
        top_percent: ((100.0 - percentile) * 10.0).round() / 10.0,
        corpus: reference.corpus.clone(),
        corpus_description: reference.corpus_description.clone(),
        reference_n: reference.n,
        scoring_method: reference.scoring_method.clone(),
    })
}

/// Empirical CDF lookup with linear interpolation between stored breakpoints.
fn percentile_in(quantiles: &[f64], step: f64, score: f64) -> f64 {
    let last = quantiles.len() - 1;
    if score <= quantiles[0] {
        return 0.0;
    }
    if score >= quantiles[last] {
        return 100.0;
    }
    // First index whose stored score is >= `score`.
    let mut low = 0usize;
    let mut high = last;
    while low < high {
        let mid = low + (high - low) / 2;
        if quantiles[mid] < score {
            low = mid + 1;
        } else {
            high = mid;
        }
    }
    let upper = low;
    let lower = upper.saturating_sub(1);
    let span = quantiles[upper] - quantiles[lower];
    let within = if span.abs() < f64::EPSILON {
        0.0
    } else {
        (score - quantiles[lower]) / span
    };
    ((lower as f64 + within) * step).clamp(0.0, 100.0)
}

/// Builds the quantile breakpoints for a set of observed scores.
///
/// Used by the `--emit-novelty-reference` generator so that the bundled file and
/// the runtime lookup share one definition of what a quantile is.
pub fn build_quantiles(scores: &mut [f64]) -> Vec<f64> {
    scores.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let n = scores.len();
    if n == 0 {
        return vec![0.0; QUANTILE_COUNT];
    }
    (0..QUANTILE_COUNT)
        .map(|i| {
            let position = (i as f64 / (QUANTILE_COUNT - 1) as f64) * (n - 1) as f64;
            let lower = position.floor() as usize;
            let upper = position.ceil() as usize;
            if lower == upper {
                scores[lower]
            } else {
                let weight = position - lower as f64;
                scores[lower] * (1.0 - weight) + scores[upper] * weight
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bundled_reference_parses_and_is_well_formed() {
        let reference = reference().expect("bundled novelty reference must parse");
        assert_eq!(reference.schema, REFERENCE_SCHEMA);
        assert_eq!(reference.quantiles.len(), QUANTILE_COUNT);
        assert!(reference.n > 0, "reference must describe a real corpus");
        assert!(
            !reference.corpus_description.is_empty(),
            "every reported percentile carries this description"
        );
        assert!(
            reference.quantiles.windows(2).all(|w| w[0] <= w[1]),
            "quantiles must be ascending"
        );
    }

    #[test]
    fn percentile_is_monotonic_in_score() {
        let low = novelty_percentile(0.10).expect("reference bundled");
        let mid = novelty_percentile(0.50).expect("reference bundled");
        let high = novelty_percentile(0.90).expect("reference bundled");
        assert!(low.percentile <= mid.percentile);
        assert!(mid.percentile <= high.percentile);
        assert!((low.percentile + low.top_percent - 100.0).abs() < 0.2);
    }

    #[test]
    fn percentile_recovers_the_reference_quantiles() {
        // Ranking the corpus against itself must return roughly the input percentile.
        let reference = reference().expect("reference bundled");
        for target in [10usize, 250, 500, 750, 900] {
            let score = reference.quantiles[target];
            let measured = novelty_percentile(score)
                .expect("reference bundled")
                .percentile;
            let expected = target as f64 * reference.quantile_step_percent;
            assert!(
                (measured - expected).abs() <= 5.0,
                "score {score} at expected percentile {expected} measured as {measured}"
            );
        }
    }

    #[test]
    fn build_quantiles_spans_the_observed_range() {
        let mut scores: Vec<f64> = (0..1000).map(|i| i as f64 / 1000.0).collect();
        let quantiles = build_quantiles(&mut scores);
        assert_eq!(quantiles.len(), QUANTILE_COUNT);
        assert!((quantiles[0] - 0.0).abs() < 1e-9);
        assert!((quantiles[QUANTILE_COUNT - 1] - 0.999).abs() < 1e-9);
        assert!(quantiles.windows(2).all(|w| w[0] <= w[1]));
    }
}
