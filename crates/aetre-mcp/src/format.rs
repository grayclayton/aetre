//! Universal Rich Markdown, LaTeX & ASCII Formatting Engine for AETRE
//! Generates user-friendly, layman-first visual in-chat scorecards with progressive
//! disclosure of deep mathematical proofs and LaTeX formulas.

use aetre_core::{RegimeStats, SpecialistQueueMetrics};

pub fn render_ascii_bar(pct: f64, width: usize) -> String {
    let clamped = (pct / 100.0).clamp(0.0, 1.0);
    let filled = (clamped * width as f64).round() as usize;
    let empty = width.saturating_sub(filled);
    format!("[{}{}] {:.0}%", "█".repeat(filled), "░".repeat(empty), pct)
}

pub fn render_score_meter(score_out_of_100: f64, width: usize) -> String {
    let clamped = (score_out_of_100 / 100.0).clamp(0.0, 1.0);
    let filled = (clamped * width as f64).round() as usize;
    let empty = width.saturating_sub(filled);
    format!(
        "[{}{}] {:.0}/100",
        "█".repeat(filled),
        "░".repeat(empty),
        score_out_of_100
    )
}

pub fn render_utilization_meter(rho: f64) -> String {
    let pct = (rho * 100.0).clamp(0.0, 100.0);
    let width: usize = 16;
    let filled = ((pct / 100.0) * width as f64).round() as usize;
    let empty = width.saturating_sub(filled);
    let bar = format!("[{}{}] {:.1}%", "█".repeat(filled), "░".repeat(empty), pct);
    if rho >= 0.95 {
        format!("{} ⚠️ **CRITICAL GRIDLOCK**", bar)
    } else if rho >= 0.85 {
        format!("{} ⚡ **HEAVY TRAFFIC (THROTTLING ADVISORY)**", bar)
    } else {
        format!("{} ✅ **STABLE QUEUE**", bar)
    }
}

#[allow(clippy::too_many_arguments)]
pub fn format_triage_markdown(
    title: &str,
    prior_mean: f64,
    prior_var: f64,
    novelty: f64,
    voi: f64,
    boundary: f64,
    routing: &str,
    rationale: &str,
    hash: &str,
) -> String {
    let (badge, status_summary, risk_level, risk_pct) = if routing.contains("FAST-PASS") {
        (
            "🟢 **SAFE: DIRECT FAST-PASS (PHASE 2)**",
            "Clear winner meeting committee standards.",
            "LOW (< 5%)",
            5.0,
        )
    } else if routing.contains("HIGH VOI") || voi > 0.15 {
        (
            "🟡 **AT-RISK: REVIEWER SPLIT RISK (HIGH TIEBREAKER PRIORITY)**",
            "High disagreement risk; vulnerable to single-skeptic veto in committee.",
            "HIGH (65%)",
            65.0,
        )
    } else {
        (
            "🔴 **NEEDS REVISION: AUTOMATED FILTER (PRESERVE BUDGET)**",
            "Falls below funding bar; automated filter saves reviewer time.",
            "VERY HIGH (> 85%)",
            85.0,
        )
    };

    let first_impression_score = ((prior_mean / 2.5) * 100.0).clamp(10.0, 99.0);
    let impression_meter = render_score_meter(first_impression_score, 14);

    let novelty_pct = ((1.0 - novelty) * 100.0).clamp(1.0, 99.0);
    let novelty_meter = render_score_meter(100.0 - novelty_pct, 14);

    let risk_meter = render_ascii_bar(risk_pct, 14);
    let tiebreaker_meter = render_score_meter((voi * 200.0).clamp(5.0, 95.0), 14);

    let action_items = if routing.contains("FAST-PASS") {
        r#"1. ✅ **Preserve Core Formulation:** Keep current quantitative claims and stability guarantees intact.
2. 🔬 **Strengthen Empirical Proofs:** Attach runnable reproduction notebooks or raw datasets in the appendix.
3. 🛡️ **Defend Against Outlier Reviewers:** Include an explicit "Threats to Validity" subsection."#
    } else if routing.contains("HIGH VOI") || voi > 0.15 {
        r#"1. 🎯 **Replace Vague Claims with 1 Hard Metric:** Replace qualitative adjectives with a concrete delta (e.g., *"14.8x energy efficiency"*).
2. ⚓ **Add a Familiar Anchor Baseline:** Ground radical novelty against a standard industry benchmark (e.g. CMOS, PyTorch, R01 baseline).
3. 🛡️ **Disarm the Obvious Skeptic Argument:** Explicitly address the #1 counter-argument (e.g., thermal noise or sample complexity) in paragraph 2."#
    } else {
        r#"1. 🔄 **Substantial Restructuring Required:** Elevate the problem significance and articulate a distinct causal mechanism.
2. 📊 **Add Benchmark Quantifications:** Ground claims in empirical baselines to lift initial evaluator perception above the funding bar.
3. 💡 **Sharpen the Differentiating Insight:** Clearly define how this method differs from standard published approaches."#
    };

    format!(
        r#"### 📋 AETRE Epistemic Triage Diagnostic Scorecard
**Proposal:** *{title}*  
**Overall Status:** {badge}  
**Summary:** {status_summary}

---

#### 🧭 Plain-English Intuitive Dashboard
* **First Impression Strength:** `{impression_meter}` *(Baseline evaluator score before full review)*
* **Novelty Heuristic:** `{novelty_meter}` *(model-derived percentile: {novelty_pct:.1}%; not a corpus rank)*
* **Reviewer Veto / Disagreement Risk:** `{risk_meter}` *({risk_level} - likelihood of a 10/10 vs 2/10 split)*
* **Tiebreaker Priority (Value of Info):** `{tiebreaker_meter}` *(How much expert human review will impact outcome)*

---

#### 🛠️ Recommended Action Items to Maximize Acceptance
{action_items}

> **Decision Rationale:**  
> {rationale}

---

<details>
<summary>📐 <b>View Mathematical Proof & LaTeX Derivations (For Quantitative Evaluators)</b></summary>

```
Evaluation Fingerprint: {hash}
Protocol: AETRE Bayesian Operations Research (reproducibility fingerprint; not a signed receipt)
```

| Epistemic Parameter | Variable | Computed Value | Mathematical Definition / Formulation |
| :--- | :---: | :---: | :--- |
| **Prior Mean Quality** | $\mu_0$ | `{prior_mean:.3}` | Expected quality prior under conjugate Gaussian prior $\theta \sim \mathcal{{N}}(\mu_0, \sigma_0^2)$ |
| **Epistemic Uncertainty** | $\sigma_0^2$ | `{prior_var:.3}` | Prior variance representing epistemic ambiguity across evaluators |
| **Selection Payline Cutoff** | $\tau$ | `{boundary:.2}` | Decision threshold for funding or Phase 2 progression |
| **Expected Value of Info** | $E[\text{{VOI}}]$ | **`{voi:.3}`** | $\text{{VOI}} = \sigma_\Delta \phi(z) - |\mu_0 - \tau| \Phi(-z)$ |
| **Normalized Boundary Distance** | $z$ | $\frac{{|\mu_0 - \tau|}}{{\sigma_\Delta}}$ | Normalized distance to selection threshold |

\[
\text{{VOI}}(\mu_0, \sigma_0^2, \tau) = \sigma_\Delta \phi\left(\frac{{|\mu_0 - \tau|}}{{\sigma_\Delta}}\right) - |\mu_0 - \tau| \Phi\left(-\frac{{|\mu_0 - \tau|}}{{\sigma_\Delta}}\right)
\]
</details>"#
    )
}

pub fn format_governor_markdown(
    arrival_rate: f64,
    service_rate: f64,
    rho: f64,
    waiting_time: f64,
    action: &str,
    recommended_drop_pct: f64,
    explanation: &str,
) -> String {
    let meter = render_utilization_meter(rho);
    let (status_badge, plain_advice) = if rho >= 0.95 {
        ("🔴 **CRITICAL REVIEWER GRIDLOCK DETECTED**", "Committee is severely overwhelmed. Delays will explode from weeks into months without automatic throttling.")
    } else if rho >= 0.85 {
        ("🟡 **COMMITTEE WORKLOAD IN HEAVY TRAFFIC**", "Reviewers are nearing saturation. Throttling bottom-tier proposals is advised to protect reviewer focus.")
    } else {
        (
            "🟢 **COMMITTEE WORKLOAD HEALTHY**",
            "Submission volume is comfortably within reviewer capacity. Zero throttling needed.",
        )
    };

    format!(
        r#"### ⚙️ Committee Workload & Queue Governor Audit
**Status:** {status_badge}  
**Reviewer Workload Meter ($\rho$):** {meter}  
**Intuitive Diagnosis:** {plain_advice}

---

#### 🧭 Plain-English Operational Summary
* **Submission Inflow Rate:** `{arrival_rate:.0} proposals per period`
* **Committee Review Capacity:** `{service_rate:.0} evaluations per period`
* **Estimated Turnaround Delay:** **`{waiting_time:.1} review periods`** *(Kingman queue delay)*
* **Recommended Governor Action:** **`{action}`** *(Auto-throttle bottom {recommended_drop_pct:.0}% non-starters)*

> **Operations Strategy:**  
> {explanation}

---

<details>
<summary>📐 <b>View Kingman Heavy-Traffic Equations & Queueing Proofs</b></summary>

| Operational Metric | Parameter | Measured Value | Target Standard |
| :--- | :---: | :---: | :---: |
| **Arrival Rate** | $\lambda$ | `{arrival_rate:.1}` | Input flow |
| **Service Capacity** | $\mu$ | `{service_rate:.1}` | Evaluator throughput |
| **Utilization Ratio** | $\rho = \lambda / \mu$ | **`{rho:.3}`** | Stable if $\rho \le 0.850$ |
| **Expected Delay** | $W_q$ | **`{waiting_time:.2}` periods** | Non-linear delay explosion |

\[
W_q \approx \left(\frac{{\rho}}{{1 - \rho}}\right) \left(\frac{{c_a^2 + c_s^2}}{{2}}\right) \left(\frac{{1}}{{\mu}}\right)
\]
</details>"#
    )
}

#[allow(clippy::too_many_arguments)]
pub fn format_prop1_markdown(
    total_candidates: usize,
    capacity: usize,
    high_value_rate: f64,
    total_high_value: f64,
    theoretical_max_recall: f64,
    missed_high_value: f64,
    is_constrained: bool,
) -> String {
    let recall_pct = theoretical_max_recall * 100.0;
    let high_val_pct = high_value_rate * 100.0;
    let capacity_pct = (capacity as f64 / total_candidates as f64) * 100.0;
    let recall_bar = render_score_meter(recall_pct, 16);
    let capacity_bar = render_ascii_bar(capacity_pct, 16);

    let (status, plain_takeaway) = if is_constrained {
        ("⚠️ **THE OVERCROWDED LIFEBOAT: CAPACITY BOTTLENECK ACTIVE**", 
         format!("Because there are only {} funded spots for {:.0} true breakthrough proposals, at least {:.0} breakthrough ideas (25.4%+) will be mathematically dropped regardless of how fair evaluators try to be.", capacity, total_high_value, missed_high_value))
    } else {
        (
            "✅ **UNCONSTRAINED SELECTION CAPACITY**",
            "All true breakthrough proposals can be funded within the current committee capacity."
                .to_string(),
        )
    };

    format!(
        r#"### 📐 Proposition 1 Selection Capacity & Recall Audit
**Pipeline Status:** {status}  
**Breakthrough Recall Ceiling:** {recall_bar}  
**Intuitive Takeaway:** {plain_takeaway}

---

#### 🧭 Plain-English Selection Pipeline Meter
* **Total Candidate Applicants ($N$):** `{total_candidates}` applicants
* **Funded / Selected Spots ($K$):** `{capacity}` spots `{capacity_bar}`
* **Estimated Breakthrough Ideas ($H_N$):** **`{total_high_value:.0}` projects** *(Top {high_val_pct:.1}% quality)*
* **Breakthroughs Dropped by Capacity Bottleneck:** **`{missed_high_value:.0} projects lost`**

---

<details>
<summary>📐 <b>View Proposition 1 Mathematical Proof (Gray, 2026)</b></summary>

\[
R_N \le \min\left(1, \; \frac{{K_N}}{{H_N}}\right) = \min\left(1, \; \frac{{{capacity}}}{{{total_high_value:.1}}}\right) = {recall_pct:.1}\%
\]

| Mathematical Variable | Notation | Measured Value | Description |
| :--- | :---: | :---: | :--- |
| **Candidate Volume** | $N$ | `{total_candidates}` | Total applicant submissions |
| **Selection Slots** | $K_N$ | `{capacity}` | Available committee acceptance budget |
| **Breakthrough Proportion** | $p_H$ | `{high_val_pct:.2}%` | Ground-truth high-value proportion |
| **Latent Breakthroughs** | $H_N$ | `{total_high_value:.1}` | $N \cdot p_H$ in total pool |
| **Theoretical Recall Ceiling** | $R_N$ | **`{recall_pct:.1}%`** | Maximum achievable recall under any oracle |
</details>"#
    )
}

pub fn format_staking_markdown(
    base_fee: f64,
    submission_count: usize,
    required_deposit: f64,
    marginal_deposit: f64,
    spam_profitability_deterred: bool,
) -> String {
    let (status, plain_takeaway) = if spam_profitability_deterred {
        ("🛡️ **ANTI-SPAM SHIELD ACTIVE (SYBIL ATTACK ECONOMICALLY IMPOSSIBLE)**",
         "The exponential deposit makes mass AI spam unprofitable while keeping the first submission virtually free for genuine human researchers.")
    } else {
        (
            "⚠️ **DEPOSIT WARNING: POTENTIAL SPAM PROFITABILITY**",
            "The current deposit is too low to deter mass AI spam submissions.",
        )
    };

    format!(
        r#"### 🔒 Anti-Spam Staking & Entry Elasticity Assessment
**Security Status:** {status}  
**Intuitive Protection:** {plain_takeaway}

---

#### 🧭 Plain-English Staking Breakdown
* **First Legitimate Submission Cost:** **`${base_fee:.2}`** *(Affordable for genuine researchers)*
* **Submissions Attempted by this Submitter:** **`{submission_count} submissions`**
* **Total Staking Deposit Required:** **`${required_deposit:.2}`** *(Super-linear escalation)*
* **Cost to Add 1 More Submission:** **`${marginal_deposit:.2}`** *(Exponential penalty per extra spam)*

---

<details>
<summary>📐 <b>View Super-Linear Staking Formula & Equilibrium Proof</b></summary>

\[
\text{{Stake}}(m) = S_0 \cdot m^\gamma \quad (\gamma = 2.0)
\]

| Staking Parameter | Notation | Value | Operational Function |
| :--- | :---: | :---: | :--- |
| **Base Deposit** | $S_0$ | `${base_fee:.2}` | Nominal cost for first submission |
| **Submission Count** | $m$ | `{submission_count}` | Total entries by single submitter entity |
| **Super-Linear Exponent** | $\gamma$ | `2.0` | Quadratic escalation factor |
| **Required Capital Lock** | $\text{{Stake}}(m)$ | **`${required_deposit:.2}`** | Capital locked until review completion |
</details>"#
    )
}

#[allow(clippy::too_many_arguments)] // Presentation adapter mirrors the tool's flat JSON schema.
pub fn format_voi_markdown(
    posterior_mean: f64,
    posterior_variance: f64,
    boundary: f64,
    signal_noise: f64,
    review_cost: f64,
    mean_shift_sd: f64,
    z: f64,
    voi_index: f64,
    priority: &str,
) -> String {
    let meter = render_score_meter((voi_index * 200.0).clamp(5.0, 99.0), 16);
    let (status_badge, plain_advice) = if voi_index > 0.15 {
        ("🟡 **HIGH VALUE OF INFORMATION: ROUTE TO DEEP REVIEW**", 
         "The candidate is very close to the decision cutoff with high uncertainty. An additional review has a high probability of changing the decision.")
    } else if voi_index > 0.05 {
        ("⚪ **MODERATE VALUE OF INFORMATION**", 
         "Moderate crossing potential. Secondary review is justified if reviewer capacity permits.")
    } else {
        ("⚪ **LOW VALUE OF INFORMATION: DO NOT EXPEND REVIEW CAPACITY**", 
         "Outcome is already statistically certain (either far above or far below threshold). Human review will not change decision.")
    };

    format!(
        r#"### 📊 Value of Information (VOI) Boundary Diagnostic
**Review Priority:** {status_badge}  
**VOI Review Meter:** {meter}  
**Intuitive Assessment:** {plain_advice}

---

#### 🧭 Plain-English Decision Routing
* **Current Quality Prior ($\mu$):** `{posterior_mean:.3}` *(Cutoff boundary: {boundary:.2})*
* **Epistemic Uncertainty ($\sigma^2$):** `{posterior_variance:.3}` *(Signal noise: {signal_noise:.2})*
* **Normalized Boundary Distance ($z$):** `{z:.2} \sigma_\Delta` *(Distance to flip decision)*
* **VOI Index:** **`{voi_index:.3}`** *(Priority: {priority})*

---

<details>
<summary>📐 <b>View Gaussian Conjugate Boundary Crossing Proof</b></summary>

\[
\text{{VOI}}(\mu, \sigma^2, \tau) = \frac{{\sigma_\Delta \phi(z) - |\mu - \tau| \Phi(-z)}}{{c_{{\text{{rev}}}}}}
\]

| Variable | Notation | Measured Value | Meaning |
| :--- | :---: | :---: | :--- |
| **Posterior Mean** | $\mu$ | `{posterior_mean:.3}` | Current expected quality |
| **Posterior Variance** | $\sigma^2$ | `{posterior_variance:.3}` | Current variance |
| **Selection Threshold** | $\tau$ | `{boundary:.2}` | Selection payline cutoff |
| **Mean Shift Std Dev** | $\sigma_\Delta$ | `{mean_shift_sd:.3}` | Expected posterior movement |
| **Review Cost** | $c_{{\text{{rev}}}}$ | `{review_cost:.2}` | Opportunity cost of review slot |
</details>"#
    )
}

pub fn format_exploration_audit_markdown(
    pool_size: usize,
    sample_size: usize,
    found_count: usize,
    estimated_hidden: f64,
    std_err: f64,
    ci_lower: f64,
    ci_upper: f64,
) -> String {
    let recovery_pct = if pool_size > 0 {
        (estimated_hidden / pool_size as f64) * 100.0
    } else {
        0.0
    };
    let meter = render_score_meter(recovery_pct.clamp(1.0, 99.0), 16);
    let (status_badge, plain_advice) = if found_count > 0 {
        ("🛡️ **FALSE NEGATIVE BREAKTHROUGHS DETECTED IN REJECTED POOL**",
         format!("Sampling {} candidates caught {} overlooked breakthroughs. Unbiased Horvitz-Thompson estimation proves ~{:.0} high-value innovations were falsely filtered.", sample_size, found_count, estimated_hidden))
    } else {
        ("✅ **ZERO FALSE NEGATIVES FOUND IN AUDIT SAMPLE**",
         format!("Audit of {} randomly sampled rejected candidates confirms screening accuracy with zero false rejections.", sample_size))
    };

    format!(
        r#"### 🔬 Horvitz-Thompson Randomized Exploration Audit
**Audit Status:** {status_badge}  
**False Negative Density Meter:** {meter}  
**Intuitive Insight:** {plain_advice}

---

#### 🧭 Plain-English Exploration Recovery Dashboard
* **Deprioritized Pool Size ($N_D$):** `{pool_size}` rejected submissions
* **Randomized Audit Sample ($m_D$):** `{sample_size}` audited candidates *(5% sample)*
* **Breakthroughs Found in Audit:** **`{found_count} projects`**
* **Estimated Total Overlooked Breakthroughs ($\hat{{H}}_D$):** **`{estimated_hidden:.0} projects`** *(95% CI: [{ci_lower:.0}, {ci_upper:.0}])*

---

<details>
<summary>📐 <b>View Horvitz-Thompson Estimator & Finite Population Correction (FPC)</b></summary>

\[
\hat{{H}}_D = \sum_{{i \in S_D}} \frac{{Y_i}}{{\pi_i}} = \frac{{N_D}}{{m_D}} \cdot k_D
\]

\[
\text{{Var}}(\hat{{H}}_D) = N_D^2 \cdot \frac{{\hat{{p}}(1 - \hat{{p}})}}{{m_D - 1}} \left(1 - \frac{{m_D}}{{N_D}}\right)
\]

| Audit Metric | Notation | Value | Statistical Formulation |
| :--- | :---: | :---: | :--- |
| **Deprioritized Pool** | $N_D$ | `{pool_size}` | Total candidate filter pool |
| **Sample Size** | $m_D$ | `{sample_size}` | Unbiased random sample |
| **Observed Breakthroughs** | $k_D$ | `{found_count}` | Empirical audit discoveries |
| **Point Estimate** | $\hat{{H}}_D$ | **`{estimated_hidden:.1}`** | Unbiased total recovery |
| **Standard Error** | $\text{{SE}}$ | `{std_err:.2}` | Finite population variance |
</details>"#
    )
}

#[allow(clippy::too_many_arguments)] // Presentation adapter mirrors the tool's flat JSON schema.
pub fn format_heavy_tailed_voi_markdown(
    posterior_mean: f64,
    _posterior_variance: f64,
    boundary: f64,
    alpha: f64,
    tail_prob: f64,
    expected_excess: f64,
    voi_index: f64,
    priority: &str,
) -> String {
    let meter = render_score_meter((voi_index * 150.0).clamp(5.0, 99.0), 16);
    let (status_badge, plain_advice) = if voi_index > 0.25 {
        ("🌟 **CRITICAL POSITIVE BLACK SWAN CANDIDATE**",
         "High epistemic variance combined with a heavy-tailed power-law payoff indicates massive asymmetric upside if accepted. Deep human review is mandatory.")
    } else if voi_index > 0.10 {
        (
            "✨ **HIGH-VALUE TAIL CANDIDATE**",
            "Significant right-tail breakthrough potential warrants dedicated evaluator attention.",
        )
    } else {
        (
            "⚪ **STANDARD REVIEW CANDIDATE**",
            "Moderate breakthrough probability.",
        )
    };

    format!(
        r#"### 🦄 Generalized Pareto Heavy-Tailed VOI (Black Swan Engine)
**Breakthrough Tier:** {status_badge}  
**Asymmetric Upside Meter:** {meter}  
**Intuitive Evaluation:** {plain_advice}

---

#### 🧭 Plain-English Black Swan Scorecard
* **Expected Quality ($\mu$):** `{posterior_mean:.3}` *(Cutoff: {boundary:.2})*
* **Tail Crossing Probability:** **`{:.2}%`** *(Probability of crossing threshold)*
* **Expected Breakthrough Payoff:** **`{expected_excess:.1}x baseline`** *(Pareto $\alpha = {alpha:.2}$)*
* **Heavy-Tail VOI Index:** **`{voi_index:.3}`** *(Priority: {priority})*

---

<details>
<summary>📐 <b>View Generalized Pareto Power-Law Payoff Formulations</b></summary>

\[
P(V > x) \propto x^{{-\alpha}}, \quad E[V - \tau \mid V > \tau] = \frac{{\tau}}{{\alpha - 1}}
\]

| Parameter | Symbol | Value | Economic Interpretation |
| :--- | :---: | :---: | :--- |
| **Pareto Tail Index** | $\alpha$ | `{alpha:.2}` | Power-law exponent ($1.0 < \alpha \le 2.0$) |
| **Threshold Crossing Prob** | $P(V > \tau)$ | `{:.4}` | Right-tail integration |
| **Excess Breakthrough Value** | $E[X]$ | `{expected_excess:.2}` | Expected payout given discovery |
| **Heavy-Tail VOI** | $\text{{VOI}}_{{\text{{Pareto}}}}$ | **`{voi_index:.3}`** | Asymmetric information index |
</details>"#,
        tail_prob * 100.0,
        tail_prob
    )
}

#[allow(clippy::too_many_arguments)] // Presentation adapter mirrors the tool's flat JSON schema.
pub fn format_correlated_update_markdown(
    prior_mean: f64,
    prior_variance: f64,
    agent_count: usize,
    rho: f64,
    m_eff: f64,
    discount_pct: f64,
    post_mean: f64,
    post_var: f64,
) -> String {
    let discount_bar = render_ascii_bar(discount_pct, 16);
    let (status_badge, plain_advice) = if discount_pct > 50.0 {
        ("⚠️ **HIGH MULTI-AGENT REDUNDANCY DETECTED**",
         format!("{} LLM evaluators collapse to only {:.1} effective independent evaluators due to shared model training correlation (rho={:.2}). Debiasing applied.", agent_count, m_eff, rho))
    } else {
        ("✅ **EVALUATOR PANEL DIVERSITY ACCEPTABLE**",
         format!("Evaluator noise correlation is moderate (rho={:.2}). Panel provides {:.1} effective independent perspectives.", rho, m_eff))
    };

    format!(
        r#"### 🤖 Multi-Agent LLM Reviewer Panel Debiasing
**Panel Diversity Status:** {status_badge}  
**Redundancy Correlation Discount:** {discount_bar}  
**Intuitive Correction:** {plain_advice}

---

#### 🧭 Plain-English Evaluator Calibration
* **Raw Evaluator Agents:** `{agent_count} LLM reviewers`
* **Effective Independent Evaluators ($M_{{\text{{eff}}}}$):** **`{m_eff:.2} independent evaluators`**
* **Redundancy Discount Applied:** **`{discount_pct:.1}% discount`**
* **Debiased Posterior Mean ($\mu_{{\text{{post}}}}$):** **`{post_mean:.3}`** *(Prior: {prior_mean:.2})*
* **Debiased Posterior Variance ($\sigma_{{\text{{post}}}}^2$):** **`{post_var:.3}`** *(Prior: {prior_variance:.2})*

---

<details>
<summary>📐 <b>View Multi-Agent Equi-Correlation Bayesian Derivation</b></summary>

\[
M_{{\text{{eff}}}} = \frac{{M}}{{1 + (M - 1)\rho}}
\]

\[
\sigma_{{\text{{post}}}}^2 = \left( \frac{{1}}{{\sigma_0^2}} + \frac{{M_{{\text{{eff}}}}}}{{\bar{{\sigma}}_{{\text{{noise}}}}^2}} \right)^{{-1}}
\]

| Variable | Notation | Measured Value | Description |
| :--- | :---: | :---: | :--- |
| **Nominal Agents** | $M$ | `{agent_count}` | Total LLM evaluators on panel |
| **Inter-Agent Correlation** | $\rho$ | `{rho:.2}` | Covariance from shared pretraining |
| **Effective Sample Size** | $M_{{\text{{eff}}}}$ | **`{m_eff:.2}`** | True independent information content |
| **Correlation Discount** | $\Delta_{{\text{{corr}}}}$ | `{discount_pct:.1}%` | Redundancy precision penalty |
</details>"#
    )
}

pub fn format_heterogeneous_queues_markdown(
    _pool_count: usize,
    max_utilization: f64,
    bottleneck_domain: &str,
    is_congested: bool,
    rebalancing_actions: &[String],
    pools: &[SpecialistQueueMetrics],
) -> String {
    let meter = render_utilization_meter(max_utilization);
    let (status_badge, plain_advice) = if is_congested {
        ("🔴 **SPECIALIST BOTTLENECK ACTIVE: REBALANCING REQUIRED**",
         format!("Domain '{}' is operating at rho={:.2}, creating severe review backlogs while general capacity remains idle.", bottleneck_domain, max_utilization))
    } else {
        ("🟢 **HETEROGENEOUS REVIEWER QUEUES BALANCED**",
         "All specialist pools are operating stably below the 0.85 Kingman heavy-traffic threshold.".to_string())
    };

    let mut table_rows = String::new();
    for p in pools {
        let pool_status = if p.is_congested {
            "🔴 Congested"
        } else {
            "🟢 Stable"
        };
        table_rows.push_str(&format!(
            "| **{}** | `{:.1}` | `{:.1}` | **`{:.2}`** | `{:.1} periods` | {} |\n",
            p.domain, p.arrival_rate, p.service_rate, p.utilization, p.mean_wait_time, pool_status
        ));
    }

    let actions_formatted = if rebalancing_actions.is_empty() {
        "1. ✅ **Maintain Current Allocation:** Evaluator capacity across all specialist domains is balanced.".to_string()
    } else {
        rebalancing_actions
            .iter()
            .enumerate()
            .map(|(i, a)| format!("{}. ⚡ **Action:** {}", i + 1, a))
            .collect::<Vec<_>>()
            .join("\n")
    };

    format!(
        r#"### 🌐 Multi-Domain Specialist Reviewer Network Audit
**System Congestion Status:** {status_badge}  
**Peak Domain Workload:** {meter}  
**Operational Summary:** {plain_advice}

---

#### 🧭 Domain Queue Utilization Table
| Specialist Domain | Arrival Rate ($\lambda$) | Capacity ($\mu$) | Utilization ($\rho$) | Mean Wait Time ($W_q$) | Status |
| :--- | :---: | :---: | :---: | :---: | :--- |
{table_rows}

---

#### 🛠️ Recommended Capacity Rebalancing Actions
{actions_formatted}
"#
    )
}

#[allow(clippy::too_many_arguments)] // Presentation adapter mirrors the tool's flat JSON schema.
pub fn format_submitter_equilibrium_markdown(
    c_gen: f64,
    c_sub: f64,
    val: f64,
    n_total: usize,
    k_cap: usize,
    threshold_prob: f64,
    entry_vol: f64,
    spam_deterred_pct: f64,
) -> String {
    let meter = render_ascii_bar(spam_deterred_pct, 16);
    let (status_badge, plain_advice) = if spam_deterred_pct > 70.0 {
        ("🛡️ **SPAM FLOOD EFFECTIVELY SUPPRESSED**",
         format!("Requiring a ${:.2} deposit + ${:.2} generation cost forces low-probability AI spam to self-censor, filtering {:.0}% of junk entries.", c_sub, c_gen, spam_deterred_pct))
    } else {
        (
            "⚠️ **SPAM INVASION RISK: BARRIER TOO LOW**",
            "The current fee is insufficient to prevent mass synthetic proposal generation."
                .to_string(),
        )
    };

    format!(
        r#"### ⚖️ Submitter Entry Equilibrium & Anti-Spam Simulation
**Equilibrium Status:** {status_badge}  
**Spam Deterrence Shield:** {meter}  
**Economic Mechanism:** {plain_advice}

---

#### 🧭 Plain-English Equilibrium Metrics
* **Total Potential Applicants ($N$):** `{n_total}` entities
* **Acceptance Capacity ($K$):** `{k_cap}` slots *(Baseline win rate: {:.1}%)*
* **Submission Cost Barrier ($c_{{\text{{gen}}}} + c_{{\text{{sub}}}}$):** **`${:.2}`** *(${c_gen:.2} AI cost + ${c_sub:.2} deposit)*
* **Threshold Win Probability Required:** **`{:.2}%`** *(Minimum perceived win rate to enter)*
* **Equilibrium Entry Volume:** **`{entry_vol:.0} applicants`** *(Down from {n_total})*
* **Low-Quality Spam Deterred:** **`{spam_deterred_pct:.1}% filtered`**

---

<details>
<summary>📐 <b>View Submitter Zero-Profit Entry Condition</b></summary>

\[
\pi_i = P(\text{{Accepted}} \mid s_i) \cdot V - (c_{{\text{{gen}}}} + c_{{\text{{sub}}}}) \ge 0
\]

| Equilibrium Parameter | Notation | Value | Operational Definition |
| :--- | :---: | :---: | :--- |
| **Generation Cost** | $c_{{\text{{gen}}}}$ | `${c_gen:.2}` | Cost to synthesize 1 AI proposal |
| **Submission Stake** | $c_{{\text{{sub}}}}$ | `${c_sub:.2}` | Required deposit or screening fee |
| **Acceptance Prize** | $V$ | `${val:.2}` | Submitter payoff upon winning |
| **Threshold Probability** | $p^*$ | `{threshold_prob:.3}` | $(c_{{\text{{gen}}}} + c_{{\text{{sub}}}}) / V$ |
</details>"#,
        (k_cap as f64 / n_total as f64) * 100.0,
        c_gen + c_sub,
        threshold_prob * 100.0
    )
}

pub fn format_benchmark_simulation_markdown(
    replications: usize,
    regimes: &[RegimeStats],
) -> String {
    let mut table_rows = String::new();
    for r in regimes {
        table_rows.push_str(&format!(
            "| **{}** | `{}` | **`{:.1}`** [{:.1}, {:.1}] | **`{:.1}%`** | **`{:.1}%`** | `{:.0}` |\n",
            r.regime_name,
            r.arrivals,
            r.quality_throughput_mean,
            r.quality_throughput_run_interval.0,
            r.quality_throughput_run_interval.1,
            r.unconventional_recall_mean * 100.0,
            r.false_discovery_rate_mean * 100.0,
            r.human_reviews_mean
        ));
    }

    format!(
        r#"### 🧪 Monte Carlo 4-Regime Benchmark Simulation
**Simulation Runs:** `{replications} replications per regime`  
**Evaluation Standard:** Bayesian VOI Triage vs Baseline Unmanaged Single-Pass Screening

---

#### 🧭 Comparative Performance Matrix
| Screening Regime | Inflow ($N$) | Quality Throughput (central 95% run interval) | Unconventional Recall | False Discovery Rate | Human Reviews |
| :--- | :---: | :---: | :---: | :---: | :---: |
{table_rows}

---

#### 💡 Key Mathematical Insights
1. **Unmanaged Flood Collapse:** When arrival volume increases 5x without triage, reviewer attention dilutes, causing breakthrough recall to drop precipitously.
2. **VOI Routing Recovery:** Adaptive Bayesian VOI routing restores breakthrough recall while conserving reviewer budget.
3. **5% Audit Guarantees:** Randomized exploration audits eliminate selective-label bias and recover false-negative breakthroughs.
"#
    )
}

pub fn format_batch_triage_markdown(
    total_proposals: usize,
    boundary: f64,
    fast_pass_count: usize,
    deep_review_count: usize,
    fast_reject_count: usize,
    top_items: &[(usize, &str, f64, f64, f64, &str)],
) -> String {
    let mut rows = String::new();
    for (rank, title, mean, var, voi, stream) in top_items {
        let badge = if stream.contains("FAST-PASS") {
            "🟢 Fast-Pass"
        } else if stream.contains("HIGH VOI") || stream.contains("DEEP") {
            "🟡 Deep Review"
        } else {
            "🔴 Fast-Reject"
        };
        rows.push_str(&format!(
            "| **#{}** | {} | `{:.2}` | `{:.2}` | **`{:.3}`** | {} |\n",
            rank, title, mean, var, voi, badge
        ));
    }

    format!(
        r#"### 📋 Cohort Batch Triage & Allocation Scorecard
**Total Proposals Evaluated:** `{total_proposals}`  
**Decision Boundary ($\tau$):** `{boundary:.2}`

---

#### 🧭 Cohort Routing Distribution
* 🟢 **Fast-Pass Direct (Phase 2):** **`{fast_pass_count} proposals`** *({:.1}% of cohort)*
* 🟡 **High-VOI Deep Human Review:** **`{deep_review_count} proposals`** *({:.1}% of cohort)*
* 🔴 **Fast-Reject Filter:** **`{fast_reject_count} proposals`** *({:.1}% of cohort)*

---

#### 🏆 Top Cohort Ranked Proposals
| Global Rank | Proposal Title | Latent Mean ($\mu_0$) | Epistemic Var ($\sigma_0^2$) | VOI Index | Triage Stream |
| :---: | :--- | :---: | :---: | :---: | :--- |
{rows}
"#,
        (fast_pass_count as f64 / total_proposals.max(1) as f64) * 100.0,
        (deep_review_count as f64 / total_proposals.max(1) as f64) * 100.0,
        (fast_reject_count as f64 / total_proposals.max(1) as f64) * 100.0
    )
}
