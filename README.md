# AETRE: Adaptive Epistemic Triage & Recall Engine

[![Zenodo DOI](https://zenodo.org/badge/1346232534.svg)](https://doi.org/10.5281/zenodo.22098366)
[![SSRN: 7161458](https://img.shields.io/badge/SSRN-7161458-blue.svg)](https://ssrn.com/abstract=7161458)
[![License: AGPL v3](https://img.shields.io/badge/License-AGPL%20v3-green.svg)](LICENSE)
[![Rust: 1.75+](https://img.shields.io/badge/Rust-1.75%2B-orange.svg)](https://www.rust-lang.org/)
[![Model Context Protocol](https://img.shields.io/badge/MCP-20%20Tools-purple.svg)](https://modelcontextprotocol.io/)
[![Live Portal](https://img.shields.io/badge/Portal-lithiumeel.com%2Faetre-emerald.svg)](https://www.lithiumeel.com/aetre)

> **"Empowering Breakthrough Ideas in the Age of Abundance."**  
> A high-performance, mathematically rigorous operations-research engine that optimizes academic peer-review pipelines, grant study sections, and venture capital dealflow.

> **Release status: experimental public alpha.** The software and mathematical
> simulations are testable, but the bundled data are synthetic and do not
> establish prospective effectiveness in a live conference, grant, or
> investment workflow. Use outputs as decision-support diagnostics, not as
> autonomous acceptance, rejection, funding, or investment decisions.

Based on the working paper:  
**The Innovation-Absorption Gap: How Artificial Intelligence Can Accelerate Idea Production Faster Than Complementary Institutions Adapt**  
*Clayton Gray (2026)* — [SSRN: 7161458](https://ssrn.com/abstract=7161458)

---

## The Problem: The Innovation-Absorption Gap

When Artificial Intelligence makes idea generation cheap ($c_{\text{gen}} \to 0$), proposal volume ($N$) explodes. However, downstream evaluation, laboratory validation, and human review capacity ($K$) remain strictly finite. 

This creates three critical pipeline pathologies:
1. **The Kingman Delay Explosion:** When evaluator utilization $\rho = \lambda / \mu$ approaches saturation ($\rho > 0.85$), wait times shoot up non-linearly according to Kingman's Heavy-Traffic equation:
   $$E[W_q] \approx \frac{\rho}{1-\rho} \cdot \frac{c_a^2 + c_s^2}{2} \cdot \frac{1}{\mu}$$
2. **The Asymmetric Payoff Trap:** In heavy-tailed domains like venture capital and breakthrough scientific discovery (Pareto index $\alpha \approx 1.25$), consensus-seeking scoring systems penalize high-variance, transformative outliers in favor of safe, incremental proposals.
3. **The Finite-Capacity Recall Ceiling (Proposition 1):** Without active epistemic triage, true breakthrough recall asymptotically decays towards zero as arrival rates surge:
   $$R_N \le \min\left(1, \frac{K_N}{H_N}\right) \to 0 \quad \text{as } N \to \infty$$

---

## The Solution: The AETRE 4-Pillar Pipeline

```text
               INCOMING PROPOSAL STREAM (N)
                            │
                            ▼
    ┌───────────────────────────────────────────────────┐
    │ 1. Bayesian Value-of-Information (VOI) Triage     │
    │    Routes attention strictly where it changes     │
    │    the downstream decision (μ_q, σ_q^2).          │
    └───────────────────────────────────────────────────┘
                            │
        ┌───────────────────┼───────────────────┐
        ▼                   ▼                   ▼
  [ Fast-Drop ]       [ VOI Queue ]       [ Auto-Pass ]
  Low Q, Low Var      High Uncertainty    High Q, Low Var
  (Quick reject)      (Deep review)       (Direct accept)
                            │
                            ▼
    ┌───────────────────────────────────────────────────┐
    │ 2. Kingman Heavy-Traffic Capacity Governor        │
    │    Dynamically throttles queues to preserve       │
    │    reviewer quality and prevent burnout (ρ ≤ 0.85)│
    └───────────────────────────────────────────────────┘
                            │
        ┌───────────────────┴───────────────────┐
        ▼                                       ▼
  [ Selected Cohort (K) ]             [ 3. Exploration Audit Pool ]
  Optimal High-Conviction             Randomized Non-Consensus Ideas
                                                │
                                                ▼
                                      [ 4. Counterfactual Tracker ]
                                      Unbiased Horvitz-Thompson H_hat_D
```

---

## Repository Structure

```text
.
├── Cargo.toml                  # Workspace manifest (AGPL-3.0)
├── crates/
│   ├── aetre-core/             # Pure Rust decision engine (VOI, Kingman, Pareto, Staking)
│   ├── aetre-cli/              # Command-line interface, VC benchmark & validation tool
│   └── aetre-mcp/              # Model Context Protocol server (20 tools, 4 resources, 3 prompts)
├── examples/
│   ├── datasets/               # Held-out review and dealflow test splits
│   ├── proposals.json          # Benchmark evaluation candidates
│   └── mcp_config.json         # Claude Desktop & Cursor connection template
├── CITATION.cff                # Citation File Format (Zenodo DOI & SSRN: 7161458)
├── Dockerfile                  # Production container definition
├── fly.toml                    # Serverless Cloud deployment config
├── DATASETS.md                 # Fixture provenance and third-party data guidance
├── LICENSE                     # GNU Affero General Public License v3.0 text
├── LICENSING.md                # AGPL/commercial licensing overview
└── README.md
```

---

## Quickstart & CLI Usage

### 1. Run the Rust Test Suite & Verification
```bash
cargo test --workspace
cargo clippy --workspace --all-targets -- -D warnings
```

### 2. Run the Multi-Regime Monte Carlo Benchmark
```bash
cargo run -p aetre-cli -- benchmark --replications 500
# Export results to JSON or CSV:
cargo run -p aetre-cli -- benchmark --replications 500 --json
cargo run -p aetre-cli -- benchmark --replications 500 --csv
```

### 3. Run the Venture Capital Pareto Dealflow Benchmark
Simulates asymmetric power-law distributions ($\alpha = 1.25$, $x_m = \$50\text{k}$, 10,000 deals, 60 unicorn targets):
```bash
cargo run -p aetre-cli -- vc-benchmark --deals 10000 --budget 100 --alpha 1.25
```

### 4. Run Backtests on Held-Out Datasets
```bash
# Smoke-test the 8-policy backtest with the included synthetic fixture
cargo run -p aetre-cli -- backtest --file examples/datasets/openreview_heldout_backtest.json --budget 4 --boundary 6.0

# Run Level 4 prospective shadow pilot simulation & 3-arm trial
cargo run -p aetre-cli -- shadow-pilot --mode simulate --budget 50 --audit-rate 0.05

# Validate predictions file against frozen test split
cargo run -p aetre-cli -- validate-predictions --file examples/validation_schema.json --budget 20 --threshold 0.5
```

### 5. Evaluate Theoretical Proposition 1 Bounds
```bash
cargo run -p aetre-cli -- bound --arrivals 5000 --capacity 200 --high-rate 0.067 --csv
```

### 6. Run Kingman Capacity Governor Telemetry
```bash
cargo run -p aetre-cli -- queue --arrival-rate 95 --service-rate 100
```

### 7. Calculate Horvitz-Thompson Exploration Audit ($\hat{H}_D$)
```bash
cargo run -p aetre-cli -- audit --pool 4800 --sample 25 --found 1
```

### 8. Compute Super-Linear Anti-Sybil Staking Requirements
```bash
cargo run -p aetre-cli -- staking --base 100 --exponent 1.5 --submissions 20
```

---

## Model Context Protocol (MCP) Integration

AETRE provides a native, high-speed Model Context Protocol (MCP) server implementing **20 Tools**, **4 Resources**, and **3 Pre-Configured Prompts** for Claude Desktop, Cursor, and other MCP clients.

### Configuration (Claude Desktop / Cursor)

AETRE runs locally as a high-performance native JSON-RPC 2.0 stdio MCP server. Add to your `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "aetre": {
      "command": "cargo",
      "args": ["run", "--release", "--manifest-path", "/PATH/TO/aetre/Cargo.toml", "-p", "aetre-mcp"]
    }
  }
}
```

### Optional local HTTP mode

```bash
cargo run -p aetre-mcp -- --serve --headless
```

HTTP mode binds to `127.0.0.1:8080` by default and does not enable cross-origin
browser access. For container deployment, set `AETRE_BIND_ADDRESS=0.0.0.0` and
set a strong `AETRE_HTTP_SERVER_TOKEN`. Non-loopback startup fails closed when
that token is absent. POST clients must send it in the `X-AETRE-Server-Token`
header. Also place the service behind a TLS reverse proxy. The bundled
Dockerfile supplies the bind address and runs as a non-root user.

### Key MCP Tools Included:
1. `aetre_calculate_voi`: Core Bayesian Value-of-Information expected-utility calculation.
2. `aetre_heavy_tailed_voi`: Pareto power-law venture capital screening ($\alpha \approx 1.25$) for asymmetric bets.
3. `aetre_author_preflight_benchmark`: Pre-flight draft diagnostic evaluating reviewer disagreement and variance risk.
4. `aetre_check_governor`: Kingman queue utilization ($\rho$) delay forecasting and capacity governor actions.
5. `aetre_congestion_matching`: Optimal reviewer-paper bipartite matching under workload constraints.
6. `aetre_sequential_stopping_rule`: Wald sequential likelihood ratio multi-round review termination.
7. `aetre_correlated_posterior_update`: Multi-agent reviewer consensus correlation debiasing.
8. `aetre_exploration_audit`: Horvitz-Thompson unbiased audit estimator ($\hat{H}_D$) on rejected pools.
9. `aetre_quadratic_staking`: Super-linear anti-sybil staking curves to deter spam.
10. `aetre_batch_triage`: Bulk dataset triage and three-stream routing.

---

## Open Engine vs. Enterprise Commercial SaaS

AETRE follows an **Open Engine / Dual-Track Architecture**:

| Feature / Capability | Open Engine (AGPL-3.0) | Enterprise Commercial License |
| :--- | :---: | :---: |
| **Core Mathematical Algorithms (`aetre-core`)** | ✅ Fully Open & Auditable | ✅ Included |
| **Model Context Protocol (MCP) Server** | ✅ 20 Local Stdio Tools | ✅ Dedicated Cloud & Local |
| **Local CLI & Terminal Simulation Harness** | ✅ Included | ✅ Included |
| **Author Pre-Flight Scans** | ✅ Included; local limits are source-configurable | ✅ Supported unlimited deployment |
| **Automated VC Dealflow Webhook (Airtable/Affinity)** | Local Script | ✅ Managed Cloud Sync |
| **Custom Corpus Platt Calibration** | Open Source | ✅ Pre-Trained Institutional Priors |
| **Commercial Exemption (No AGPL copyleft)** | ❌ Bound by AGPL-3.0 | ✅ Full Commercial License |
| **Dedicated SLA & Multi-Tenant Support** | Community | ✅ Priority SLA & Direct Support |

---

## Citation & Academic Reference

If you use AETRE in your research, peer-review systems, or investment analysis, please cite:

```bibtex
@article{gray2026innovation,
  title={The Innovation-Absorption Gap: How Artificial Intelligence Can Accelerate Idea Production Faster Than Complementary Institutions Adapt},
  author={Gray, Clayton},
  journal={SSRN Electronic Journal},
  year={2026},
  doi={10.2139/ssrn.7161458},
  url={https://ssrn.com/abstract=7161458}
}
```

---

## License & Inquiries

This software is distributed under a **Dual-License Model**:
* **Open-source option:** The code is licensed under [AGPL-3.0-or-later](LICENSE), including for commercial use, subject to the AGPL's terms.
* **Commercial option:** Organizations wishing to use AETRE without the AGPL's copyleft obligations may negotiate a separate written commercial license.

All bundled example datasets are synthetic test fixtures, not empirical
validation corpora. See [DATASETS.md](DATASETS.md) before using or
redistributing external data. Evaluation
fingerprints emitted by the engine are deterministic reproducibility
identifiers; they are not signed receipts or proof of external validation.

* **Author & Maintainer:** Clayton Gray
* **Portal & Licensing:** [https://www.lithiumeel.com/aetre](https://www.lithiumeel.com/aetre)
* **Inquiries:** `contact@lithiumeel.com` | `privacy@lithiumeel.com`
