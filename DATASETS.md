# Dataset provenance and terms

The file `openreview_heldout_backtest.json` is a synthetic fixture created to
exercise AETRE's parsers and algorithms. It is a replacement for a previously
committed processed research corpus. It is distributed under the same
AGPL-3.0-or-later terms as the source code. Do not infer empirical validation
from results produced with this fixture.

The other JSON files under `examples/datasets/` are also fictional fixtures.
Their identifiers use a `SYNTH-` prefix, their text identifies them as
fictional, and any example URLs use the reserved `.invalid` top-level domain.
The general `examples/proposals.json` file contains fictional proposals for CLI
demonstrations.

Previously bundled arXiv/SSRN-derived examples and mixed real-paper metadata
are intentionally excluded from the release snapshot.

Real OpenReview, PaperCopilot, NIH, USPTO, arXiv, SSRN, Papers with Code, or
other third-party records are not relicensed by this repository. Before
downloading, publishing, or redistributing such records, verify the source's
current terms, applicable paper licenses, privacy requirements, and conference
policies. Prefer publishing an ingestion script, frozen identifiers, hashes,
and aggregate metrics instead of republishing full text when rights are not
clear.
