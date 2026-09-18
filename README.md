# AETRE & The Governed Agent: Unified Decision-Theoretic Operating System

[![Zenodo DOI (AETRE software)](https://zenodo.org/badge/1346232534.svg)](https://doi.org/10.5281/zenodo.22098366)
[![Zenodo DOI (replication bundle)](https://zenodo.org/badge/DOI/10.5281/zenodo.22814799.svg)](https://doi.org/10.5281/zenodo.22814799)
[![SSRN: 7161458](https://img.shields.io/badge/SSRN-7161458-blue.svg)](https://ssrn.com/abstract=7161458)
[![License: AGPL v3](https://img.shields.io/badge/License-AGPL%20v3-green.svg)](LICENSE)
[![Rust: 1.75+](https://img.shields.io/badge/Rust-1.75%2B-orange.svg)](https://www.rust-lang.org/)
[![Model Context Protocol](https://img.shields.io/badge/MCP-24%20Tools-purple.svg)](https://modelcontextprotocol.io/)
[![Live Portal](https://img.shields.io/badge/Portal-lithiumeel.com%2Faetre-emerald.svg)](https://www.lithiumeel.com/aetre)

> **"Governing Autonomous Intelligence in the Age of Abundance."**  
> A high-performance, mathematically unified decision-theoretic operating system uniting macroeconomic proposal/portfolio triage (AETRE) and microeconomic execution governance (The Governed Agent).

> **Release status: experimental public alpha.** The software and mathematical
> simulations are testable, but the bundled data are synthetic and do not
> establish prospective effectiveness in a live conference, grant, or
> investment workflow. Use outputs as decision-support diagnostics, not as
> autonomous acceptance, rejection, funding, or investment decisions.

Based on the research series by Clayton Gray (2026):  
1. **The Innovation-Absorption Gap: How Artificial Intelligence Can Accelerate Idea Production Faster Than Complementary Institutions Adapt**  
   *Clayton Gray (2026a)* — [SSRN: 7161458](https://ssrn.com/abstract=7161458)
2. **The Admission Frontier: An Economically Regulated Decision-Theoretic Runtime and Fail-Closed Verification Gate for Autonomous Agents**  
   *Clayton Gray (2026d)* — distributed within the replication bundle *The Implementation Frontier: Capital Allocation, Deliberative Stopping, and Verified Agent Gatekeeping* ([Zenodo: 10.5281/zenodo.22814799](https://doi.org/10.5281/zenodo.22814799))

This software is archived under its own DOI, separate from the papers above:
**AETRE: Adaptive Epistemic Triage & Recall Engine** ([Zenodo: 10.5281/zenodo.22098366](https://doi.org/10.5281/zenodo.22098366)).

---

## The Problem: The Innovation-Absorption Gap

When Artificial Intelligence makes idea and action generation cheap ($c_{\text{gen}} \to 0$), proposal and execution volume ($N$) explodes. However, downstream evaluation, laboratory validation, code review, and human gatekeeper capacity ($K$) remain strictly finite.

This creates three critical pipeline pathologies:
1. **The Kingman Delay Explosion:** When evaluator utilization $\rho = \lambda / \mu$ approaches saturation ($\rho > 0.85$), wait times shoot up non-linearly according to Kingman's Heavy-Traffic equation:
   $$E[W_q] \approx \frac{\rho}{1-\rho} \cdot \frac{c_a^2 + c_s^2}{2} \cdot \frac{1}{\mu}$$
2. **The Asymmetric Payoff Trap:** In heavy-tailed domains like venture capital, breakthrough discovery, and agentic code patches (Pareto index $\alpha \approx 1.25$), consensus-seeking scoring systems penalize high-variance, transformative outliers in favor of safe, incremental proposals.
3. **The Finite-Capacity Recall Ceiling (Proposition 1):** Without active epistemic triage, true breakthrough recall asymptotically decays towards zero as arrival rates surge:
   $$R_N \le \min\left(1, \frac{K_N}{H_N}\right) \to 0 \quad \text{as } N \to \infty$$

---

## Two-Layer Decision-Theoretic Architecture

AETRE unifies **Macroeconomic Pipeline Triage** (managing institutional review bandwidth and portfolio recall) with **Microeconomic Execution Governance** (safeguarding agentic execution runtimes and pull-request verification).

```text
========================================================================================
 MACRO LEVEL: Institutional Proposal & Portfolio Triage (AETRE / Gray 2026a)
========================================================================================
                     Incoming Submissions / Dealflow (N)
                                     │
                                     ▼
        ┌─────────────────────────────────────────────────────────┐
        │ 1. Bayesian Value-of-Information (VOI) Engine           │
        │    - Closed-form normal VOI & Pareto heavy-tailed VOI   │
        │    - Direct-Pass, Fast-Drop, or Deep-Review allocation  │
        └─────────────────────────────────────────────────────────┘
                                     │
                                     ▼
        ┌─────────────────────────────────────────────────────────┐
        │ 2. Kingman Heavy-Traffic Capacity Governor              │
        │    - Dynamic queue throttling when ρ → 1.0              │
        │    - Preserves reviewer quality; deters burnout         │
        └─────────────────────────────────────────────────────────┘
                  │                                     │
                  ▼                                     ▼
        [ Admitted Cohort (K) ]             [ Exploration Audit Pool ]
        Optimal Conviction Allocation       Horvitz-Thompson H_hat_D Unbiased Audit

========================================================================================
 MICRO LEVEL: Autonomous Execution & Verification Gate (Governed Agent / Gray 2026d)
========================================================================================
                     Autonomous PRs / Candidate Actions
                                     │
                                     ▼
        ┌─────────────────────────────────────────────────────────┐
        │ 3. Tripartite Review Boundary & Bellman DP (Road A)     │
        │    - Computes Expected Value of Verification (V*)       │
        │    - Classifies AUTO_EXECUTE, REQUIRE_REVIEW, or REJECT │
        │    - Fail-Closed: Rejects on malformed input/divergence │
        └─────────────────────────────────────────────────────────┘
                                     │
                                     ▼
        ┌─────────────────────────────────────────────────────────┐
        │ 4. Heterogeneous cμ-Rule Knapsack Controller            │
        │    - Value density sorting: ρ_i = Δu_i / c_i            │
        │    - Budget-constrained admission under shadow prices   │
        └─────────────────────────────────────────────────────────┘
                  │                                     │
                  ▼                                     ▼
        [ Auto-Merged / Dispatched ]        [ Escalated to Human Gatekeeper ]
```

---

## Repository Structure

```text
.
├── Cargo.toml                  # Workspace manifest (AGPL-3.0)
├── crates/
│   ├── aetre-core/             # Pure Rust decision engine:
│   │   ├── src/voi.rs          #   - Bayesian VOI & heavy-tailed Pareto
│   │   ├── src/governor.rs     #   - Bellman DP & Tripartite Review Boundary (Road A)
│   │   ├── src/knapsack.rs     #   - Heterogeneous cμ-rule knapsack controller
│   │   ├── src/queues.rs       #   - Kingman heavy-traffic capacity governor
│   │   ├── src/audit.rs        #   - Horvitz-Thompson exploration audit
│   │   └── src/staking.rs      #   - Super-linear anti-sybil staking
│   ├── aetre-cli/              # Native CLI for simulations, bounds & backtests
│   └── aetre-mcp/              # Model Context Protocol server (24 tools, 4 resources, 3 prompts)
├── python/
│   └── governed_agent/         # Python reference runtime for agent execution governance
├── scripts/
│   └── pre-commit-governed-gate.py  # Standalone pre-commit verification gatekeeper
├── tests/                      # Python unit & regression suite (79 tests)
├── benchmarks/                 # Verification benchmarks and institutional queue sweeps
├── examples/
│   ├── datasets/               # Held-out review and dealflow test splits
│   ├── proposals.json          # Benchmark evaluation candidates
│   └── mcp_config.json         # Claude Desktop & Cursor connection template
├── .pre-commit-hooks.yaml      # Pre-commit hook definition for git integration
├── .github/workflows/
│   ├── ci.yml                  # Rust & MCP server automated verification
│   └── governed-gate-template.yml # Reusable GitHub Actions agent PR gating workflow
├── CITATION.cff                # Dual academic citation metadata
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

### 2. Run the Python Reference Governance Suite
```bash
python -m unittest discover -s tests
```

### 3. Run the Standalone Governed Agent Pre-Commit Gate
Fast, fail-closed verification gate for agentic code modifications:
```bash
python scripts/pre-commit-governed-gate.py --all-files
```

### 4. Run the Macro Monte Carlo & Dealflow Benchmarks
```bash
# Multi-regime academic triage simulation (500 replications)
cargo run -p aetre-cli -- benchmark --replications 500

# Venture Capital Pareto dealflow benchmark (10,000 deals, α = 1.25)
cargo run -p aetre-cli -- vc-benchmark --deals 10000 --budget 100 --alpha 1.25

# Theoretical Proposition 1 recall ceiling bound
cargo run -p aetre-cli -- bound --arrivals 5000 --capacity 200 --high-rate 0.067 --csv
```

### 5. Simulate Institutional Heterogeneous Agent Queues
Simulates density-greedy $c\mu$-rule knapsack vs. FIFO across varying institutional review capacities:
```bash
python evaluate_institutional_queues.py
```

---

## Model Context Protocol (MCP) Integration

AETRE provides a high-performance native JSON-RPC 2.0 Model Context Protocol (MCP) server implementing **24 Tools**, **4 Resources**, and **3 Pre-Configured Prompts** for Claude Desktop, Cursor, and autonomous agent sidecars.

### Configuration (Claude Desktop / Cursor)

Install the binary (`cargo install aetre-mcp`),
then add to your `claude_desktop_config.json`. The server speaks stdio by default:

```json
{
  "mcpServers": {
    "aetre": {
      "command": "aetre-mcp"
    }
  }
}
```

### Local HTTP / Container Mode

```bash
cargo run -p aetre-mcp -- --serve --headless
```

HTTP mode binds to `127.0.0.1:8080` by default and does not enable cross-origin browser access. For container deployment, set `AETRE_BIND_ADDRESS=0.0.0.0` and set a strong `AETRE_HTTP_SERVER_TOKEN`. Non-loopback startup fails closed when that token is absent. POST clients must send it in the `X-AETRE-Server-Token` header. Also place the service behind a TLS reverse proxy. The bundled Dockerfile supplies the bind address and runs as a non-root user.

### Comprehensive 24-Tool Catalog

#### Macroeconomic Pipeline & Portfolio Tools (AETRE)
1. `aetre_system_catalog`: Diagnostic catalog of registered tools, resources, and algorithms.
2. `aetre_triage_proposal`: Full three-stream triage routing (`FAST_DROP`, `VOI_QUEUE`, `AUTO_PASS`).
3. `aetre_calculate_voi`: Closed-form Gaussian Value-of-Information ($V^*$) calculation.
4. `aetre_check_governor`: Kingman heavy-traffic queue delay ($E[W_q]$) forecasting and capacity throttling.
5. `aetre_exploration_audit`: Horvitz-Thompson unbiased discovery rate estimator ($\hat{H}_D$) for rejected pools.
6. `aetre_evaluate_staking`: Anti-sybil quadratic staking schedule for incoming proposals.
7. `aetre_proposition_1_bound`: Asymptotic recall bound ($R_N \le \min(1, K_N / H_N)$) verification.
8. `aetre_correlated_posterior_update`: Multi-agent reviewer consensus correlation debiasing.
9. `aetre_heavy_tailed_voi`: Pareto power-law ($\alpha \approx 1.25$) expected value of information for extreme outcomes.
10. `aetre_quadratic_staking`: Continuous super-linear deposit curves to eliminate volume spam.
11. `aetre_heterogeneous_queues`: Heterogeneous task duration and multi-server queue delay modeling.
12. `aetre_author_preflight_benchmark`: Pre-flight variance and review risk diagnostic for manuscript drafts.
13. `aetre_simulate_benchmark`: End-to-end multi-policy institutional pipeline Monte Carlo simulation.
14. `aetre_batch_triage`: Bulk dataset triage for high-volume portfolio operations.
15. `aetre_recall_scaling_curve`: Empirical recall scaling curves across arrival volumes.
16. `aetre_congestion_matching`: Bipartite reviewer-candidate matching under capacity constraints.
17. `aetre_sequential_stopping_rule`: Wald sequential likelihood ratio multi-round review termination.
18. `aetre_platt_calibrate`: Empirical score recalibration via Platt sigmoid transformations.
19. `aetre_empirical_bootstrap`: Non-parametric bootstrap confidence intervals for triage policies.
20. `aetre_frontier_sweep`: Multi-dimensional ROC and cost-utility frontier sweep.

#### Microeconomic Agent Runtime Governance Tools (Governed Agent)
21. `governed_bellman_triage`: Evaluates agent action triage via Bellman dynamic programming and computes Expected Value of Verification ($V^*$).
22. `governed_review_boundary`: Computes the tripartite decision boundary ($\Delta u = u_{\text{auto}} - u_{\text{review}}$) classifying candidates into `AUTO_EXECUTE`, `REQUIRE_REVIEW`, or `REJECT`.
23. `governed_knapsack_admit`: Density-greedy $c\mu$-rule knapsack controller packing candidates by value density ($\rho_i = \Delta u_i / c_i$) under budget $K$.
24. `governed_gate_pr`: Road A fail-closed pull request gatekeeper evaluating verification, costs, blast radius, and test regressions.

---

## Autonomous Agent Pre-Commit & CI Integration

The Governed Agent gatekeeper can be integrated into any autonomous coding agent pipeline (Claude Code, Cursor, GitHub Actions, pre-commit):

### Pre-Commit Integration (`.pre-commit-config.yaml`)
```yaml
repos:
  - repo: local
    hooks:
      - id: governed-gate
        name: Governed Agent Verification Gate (Road A)
        entry: python scripts/pre-commit-governed-gate.py
        language: python
        types: [python]
```

### GitHub Actions Pull Request Gate
See [`.github/workflows/governed-gate-template.yml`](.github/workflows/governed-gate-template.yml) for a complete workflow that fails closed on any unverified autonomous pull request.

---

## Open Engine vs. Commercial License

AETRE follows an **Open Engine / Dual-Track Architecture**:

| Feature / Capability | Open Engine (AGPL-3.0) | Enterprise Commercial License |
| :--- | :---: | :---: |
| **Core Mathematical Algorithms (`aetre-core`)** | ✅ Fully Open & Auditable | ✅ Included |
| **Model Context Protocol (MCP) Server** | ✅ 24 local native tools | ✅ Same engine, no copyleft obligation |
| **Micro Execution Governance (Governed Agent)** | ✅ Included (Road A Gate) | ✅ Enterprise Policy Enforcement |
| **Local CLI & Terminal Simulation Harness** | ✅ Included | ✅ Included |
| **Author Pre-Flight Scans** | ✅ Source-configurable | ✅ Supported Unlimited Deployment |
| **Automated VC Dealflow Webhook (Airtable/Affinity)** | Local script | ✅ Local script, commercially licensed |
| **Custom Institutional Priors Calibration** | Open Source | ✅ Pre-Trained Enterprise Priors |
| **Commercial Exemption (No AGPL copyleft)** | ❌ Bound by AGPL-3.0 | ✅ Full Commercial License |
| **Support** | Community | ✅ Direct channel to the author |

---

## Citation & Academic Reference

If you use AETRE or the Governed Agent runtime in your research or production systems, please cite:

```bibtex
@article{gray2026innovation,
  title={The Innovation-Absorption Gap: How Artificial Intelligence Can Accelerate Idea Production Faster Than Complementary Institutions Adapt},
  author={Gray, Clayton},
  journal={SSRN Electronic Journal},
  year={2026},
  doi={10.2139/ssrn.7161458},
  url={https://ssrn.com/abstract=7161458}
}

@software{gray2026admission,
  title={The Admission Frontier: An Economically Regulated Decision-Theoretic Runtime and Fail-Closed Verification Gate for Autonomous Agents},
  author={Gray, Clayton},
  year={2026},
  publisher={Zenodo},
  doi={10.5281/zenodo.22814799},
  url={https://doi.org/10.5281/zenodo.22814799}
}
```

---

## License & Inquiries

This software is distributed under a **Dual-License Model**:
* **Open-source option:** The code is licensed under [AGPL-3.0-or-later](LICENSE), including for commercial use, subject to the AGPL's terms.
* **Commercial option:** Organizations wishing to use AETRE or the Governed Agent without the AGPL's copyleft obligations may negotiate a separate written commercial license. See [LICENSING.md](LICENSING.md).

All bundled example datasets are synthetic test fixtures, not empirical validation corpora. See [DATASETS.md](DATASETS.md) before using or redistributing external data. Evaluation fingerprints emitted by the engine are deterministic reproducibility identifiers; they are not signed receipts or proof of external validation.

* **Author & Maintainer:** Clayton Gray
* **Portal & Licensing:** [https://www.lithiumeel.com/aetre](https://www.lithiumeel.com/aetre)
* **Inquiries:** `contact@lithiumeel.com` | `privacy@lithiumeel.com`
