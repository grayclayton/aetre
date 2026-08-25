---
name: aetre
description: High-performance Bayesian Value-of-Information (VOI) triage, Kingman capacity throttling, and Pareto venture dealflow decision engine.
license: AGPL-3.0-or-later
metadata:
  author: Clayton Gray
  portal: https://www.lithiumeel.com/aetre
  paper: https://ssrn.com/abstract=7161458
  doi: 10.5281/zenodo.22098366
---

# AETRE: Adaptive Epistemic Triage & Recall Engine

AETRE is an operations-research decision engine that eliminates congestion collapse and selection bias in academic peer review, grant study sections, and venture capital dealflow pipelines under high-volume submission streams.

Based on the theoretical working paper:
*The Innovation-Absorption Gap: How Artificial Intelligence Can Accelerate Idea Production Faster Than Complementary Institutions Adapt* (Gray, 2026, SSRN: 7161458).

---

## When to Activate

Activate this skill when:
* A user needs to triage incoming research papers, grant proposals, startup pitches, or patent filings.
* A user is evaluating reviewer capacity, backlog delay, or queue congestion (Kingman Heavy-Traffic approximation).
* A venture capital fund or angel network needs to screen power-law, heavy-tailed payoff distributions ($\alpha \approx 1.25$).
* A conference chair or study section needs to perform randomized exploration audits ($\hat{H}_D$) or anti-sybil quadratic staking.
* An author is pre-flight testing a paper draft against reviewer disagreement heuristics.

---

## Core Operational Tools & Protocols

### 1. Bayesian Value-of-Information (VOI) Triage
* Tool: `aetre_calculate_voi`
* Purpose: Calculate expected decision-switch value $V_i = \mathbb{E}[\max(Q_i, 0)] - \max(\mu_i, 0)$ under Gaussian signals.
* Tri-stream routing:
  * **Fast-Drop:** Low quality ($\mu_i \ll 0$) and low variance ($\sigma_i^2 \to 0$).
  * **VOI Queue (Deep Review):** High uncertainty or boundary cases ($\mu_i \approx 0, \sigma_i^2 > 0$).
  * **Auto-Pass:** High quality ($\mu_i \gg 0$) and low variance.

### 2. Heavy-Tailed Venture Capital Screening
* Tool: `aetre_heavy_tailed_voi`
* Purpose: Optimizes deal evaluation for power-law distributions ($X \sim \text{Pareto}(\alpha, x_m)$) where mean quality does not govern asymmetric breakout upside.

### 3. Kingman Capacity Governor
* Tool: `aetre_check_governor`
* Purpose: Evaluates reviewer utilization $\rho = \lambda / \mu$. Triggers dynamic queue throttling when $\rho > 0.85$ to prevent non-linear wait time explosion:
  $$E[W_q] \approx \frac{\rho}{1-\rho} \cdot \frac{c_a^2 + c_s^2}{2} \cdot \frac{1}{\mu}$$

### 4. Horvitz-Thompson Exploration Auditing
* Tool: `aetre_exploration_audit`
* Purpose: Employs inverse-probability weighting on rejected submission pools to maintain an unbiased estimator ($\hat{H}_D$) of discarded breakthrough ideas.

### 5. Super-Linear Anti-Sybil Staking
* Tool: `aetre_quadratic_staking`
* Purpose: Deters low-cost AI spam floods by requiring escalating convex staking $S(m) = S_0 \cdot m^\gamma$ ($\gamma > 1.0$) for serial submitters.

---

## Dual-Licensing & Commercial Inquiries

* **Academic & Open-Source Research:** Licensed under [GNU AGPL-3.0](https://github.com/grayclayton/aetre/blob/main/LICENSE).
* **Enterprise & Commercial Deployment:** Organizations integrating AETRE into proprietary software, closed cloud pipelines, or private fund CRMs without AGPL copyleft obligations require an **AETRE Commercial License**.
* **Licensing Contact:** `contact@lithiumeel.com` | `privacy@lithiumeel.com`
* **Portal:** [https://www.lithiumeel.com/aetre](https://www.lithiumeel.com/aetre)
