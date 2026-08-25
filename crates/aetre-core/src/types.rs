use serde::{Deserialize, Serialize};

/// System configuration parameters for the AETRE engine
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Parameters {
    pub baseline_arrivals: usize,
    pub ai_arrival_multiplier: f64,
    pub evaluation_budget: f64,
    pub acceptance_capacity: usize,
    pub high_value_threshold: f64,
    pub unconventional_share: f64,
    pub novelty_penalty: f64,
    pub unmanaged_baseline_noise: f64,
    pub initial_screen_cost: f64,
    pub fast_review_cost: f64,
    pub deep_review_cost: f64,
    pub initial_screen_noise: f64,
    pub fast_review_noise: f64,
    pub deep_review_noise: f64,
    pub fast_review_budget_share: f64,
    pub randomized_audit_budget_share: f64,
}

impl Default for Parameters {
    fn default() -> Self {
        Self {
            baseline_arrivals: 1_000,
            ai_arrival_multiplier: 5.0,
            evaluation_budget: 1_000.0,
            acceptance_capacity: 200,
            high_value_threshold: 1.5,
            unconventional_share: 0.10,
            novelty_penalty: 0.0,
            unmanaged_baseline_noise: 0.50,
            initial_screen_cost: 0.05,
            fast_review_cost: 0.50,
            deep_review_cost: 2.00,
            initial_screen_noise: 1.50,
            fast_review_noise: 0.80,
            deep_review_noise: 0.35,
            fast_review_budget_share: 0.40,
            randomized_audit_budget_share: 0.05,
        }
    }
}

/// A candidate proposal submitted to the innovation pipeline
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Candidate {
    pub id: usize,
    pub latent_quality: f64,
    pub is_unconventional: bool,
    pub posterior_mean: f64,
    pub posterior_variance: f64,
    pub voi_index: f64,
    pub status: CandidateStatus,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CandidateStatus {
    Pending,
    FastRejected,
    DeepReviewed,
    RandomlyAudited,
    Accepted,
    Rejected,
}

/// Selection and performance summary metrics for a screening run
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SelectionSummary {
    pub arrivals: usize,
    pub accepted: usize,
    pub human_reviews: usize,
    pub quality_throughput: f64,
    pub mean_accepted_quality: f64,
    pub false_discovery_rate: f64,
    pub unconventional_high_value_recall: f64,
    pub final_signal_noise: f64,
    pub evaluation_cost: f64,
    pub estimated_hidden_unconventional: Option<f64>,
}

/// Statistics aggregated across Monte Carlo replications
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegimeStats {
    pub regime_name: String,
    pub arrivals: usize,
    pub acceptances: usize,
    pub quality_throughput_mean: f64,
    /// Central 95% interval of outcomes across simulation replications.
    /// This is a run-to-run distribution interval, not a confidence interval for the mean.
    #[serde(alias = "quality_throughput_ci")]
    pub quality_throughput_run_interval: (f64, f64),
    pub mean_accepted_quality_mean: f64,
    #[serde(alias = "mean_accepted_quality_ci")]
    pub mean_accepted_quality_run_interval: (f64, f64),
    pub unconventional_recall_mean: f64,
    #[serde(alias = "unconventional_recall_ci")]
    pub unconventional_recall_run_interval: (f64, f64),
    pub false_discovery_rate_mean: f64,
    #[serde(alias = "false_discovery_rate_ci")]
    pub false_discovery_rate_run_interval: (f64, f64),
    pub human_reviews_mean: f64,
    pub estimated_hidden_unconventional_mean: Option<f64>,
}

/// An individual evaluator agent observation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentEvaluation {
    pub agent_id: String,
    pub score: f64,
    pub noise_sd: f64,
}

/// Result of a Bayesian conjugate update with correlated agent error noise
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CorrelatedUpdateResult {
    pub posterior_mean: f64,
    pub posterior_variance: f64,
    pub effective_evaluator_count: f64,
    pub correlation_discount: f64,
}

/// Result of a Generalized Pareto / Heavy-Tailed Value-of-Information estimation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HeavyTailVoiResult {
    pub voi_index: f64,
    pub tail_probability: f64,
    pub expected_excess_payoff: f64,
    pub tail_index: f64,
}

/// Result of quadratic anti-sybil staking fee evaluation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuadraticStakingResult {
    pub base_fee: f64,
    pub escalation_exponent: f64,
    pub submission_count: usize,
    pub total_stake_required: f64,
    pub marginal_stake_for_next: f64,
    pub spam_deterrence_pct: f64,
}

/// Single queue pool in a heterogeneous multi-specialist reviewer network
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpecialistQueueMetrics {
    pub domain: String,
    pub arrival_rate: f64,
    pub service_rate: f64,
    pub utilization: f64,
    pub mean_wait_time: f64,
    pub is_congested: bool,
}

/// System-wide metrics for heterogeneous multi-specialist queueing network
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HeterogeneousSystemMetrics {
    pub pools: Vec<SpecialistQueueMetrics>,
    pub max_utilization: f64,
    pub bottleneck_domain: String,
    pub is_system_congested: bool,
    pub rebalancing_actions: Vec<String>,
}

/// Comprehensive pre-submission diagnostic scorecard for authors and researchers
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuthorPreflightReport {
    pub title: String,
    pub prior_mean: f64,
    pub epistemic_variance: f64,
    pub novelty_score: f64,
    pub crowd_novelty_percentile: f64,
    pub reviewer_disagreement_risk: String,
    pub predicted_triage_stream: String,
    pub voi_index: f64,
    pub prescriptive_action_plan: Vec<String>,
    pub variance_reduction_target: f64,
    pub evaluation_fingerprint: String,
    pub markdown_badge: String,
}

/// A single evaluation dimension in multi-attribute epistemic triage
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MultiAttributeDimension {
    pub name: String,
    pub prior_mean: f64,
    pub prior_variance: f64,
    pub weight: f64,
    #[serde(default)]
    pub threshold: Option<f64>,
    #[serde(default = "default_noise_sd")]
    pub review_noise_sd: f64,
}

/// A candidate with multi-attribute evaluation dimensions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MultiAttributeCandidate {
    pub id: String,
    pub title: String,
    pub dimensions: Vec<MultiAttributeDimension>,
}

fn default_noise_sd() -> f64 {
    0.80
}

/// Contribution breakdown for a single dimension in multi-attribute VOI
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DimensionVoiContribution {
    pub dimension: String,
    pub weight: f64,
    pub marginal_voi: f64,
    pub variance_share: f64,
}

/// Result of multi-attribute Bayesian VOI epistemic triage
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MultiAttributeVoiResult {
    pub composite_prior_mean: f64,
    pub composite_prior_variance: f64,
    pub composite_threshold: f64,
    pub composite_voi: f64,
    pub dimension_contributions: Vec<DimensionVoiContribution>,
    pub recommended_review_dimensions: Vec<String>,
    pub suggested_routing: String,
}

/// Profile of a reviewer for congestion-aware matching
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReviewerProfile {
    pub id: String,
    pub name: String,
    pub domain: String,
    pub capacity: usize,
    pub current_load: usize,
    pub service_rate: f64,
    pub arrival_rate: f64,
    pub expertise_tags: Vec<String>,
}

/// Proposal requirement for congestion-aware reviewer matching
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProposalRequirement {
    pub id: String,
    pub title: String,
    pub domain: String,
    pub voi_index: f64,
    pub required_reviews: usize,
    pub keywords: Vec<String>,
}

/// Single assignment of a proposal to a reviewer
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MatchAssignment {
    pub proposal_id: String,
    pub reviewer_id: String,
    pub affinity_score: f64,
    pub priority_rank: usize,
}

/// Reviewer utilization post-assignment
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReviewerUtilizationReport {
    pub reviewer_id: String,
    pub domain: String,
    pub assigned_count: usize,
    pub capacity: usize,
    pub pre_utilization: f64,
    pub post_utilization: f64,
    pub is_over_capacity: bool,
}

/// Result of congestion-aware reviewer matching optimization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CongestionMatchingResult {
    pub assignments: Vec<MatchAssignment>,
    pub unassigned_proposals: Vec<String>,
    pub reviewer_utilizations: Vec<ReviewerUtilizationReport>,
    pub bottleneck_warnings: Vec<String>,
    pub global_objective_score: f64,
}

/// Single evaluation step in a sequential review process
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SequentialReviewStep {
    pub step: usize,
    pub reviewer_id: String,
    pub score: f64,
    pub noise_sd: f64,
    pub cost: f64,
}

/// Recommended action from sequential stopping rule
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum SequentialDecision {
    Accept,
    Reject,
    SolicitMoreReviews {
        recommended_next_evaluations: usize,
        expected_cost: f64,
    },
}

/// Result of dynamic sequential stopping boundary evaluation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SequentialStoppingResult {
    pub current_step: usize,
    pub posterior_mean: f64,
    pub posterior_variance: f64,
    pub decision: SequentialDecision,
    pub decision_confidence: f64,
    pub current_voi: f64,
    pub boundary_distance: f64,
    pub total_accumulated_cost: f64,
    pub stopping_rationale: String,
}

/// A venture capital or equity investment opportunity candidate
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VentureDealCandidate {
    pub deal_id: String,
    pub company_name: String,
    pub sector: String,
    pub preliminary_score: f64,
    pub epistemic_variance: f64,
    pub true_payoff_multiplier: f64,
    pub is_commodity_wrapper: bool,
    pub is_unicorn_outlier: bool,
    pub heavy_tailed_voi: f64,
}

/// Summary metrics for a venture deal screening cohort
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VentureCohortSummary {
    pub strategy_name: String,
    pub deals_evaluated: usize,
    pub diligence_deals_selected: usize,
    pub portfolio_moic: f64,
    pub portfolio_irr_approx: f64,
    pub unicorns_captured: usize,
    pub total_unicorns: usize,
    pub outlier_recall: f64,
    pub diligence_hours_per_unicorn: f64,
    pub wrappers_avoided_count: usize,
    pub wrappers_invested_count: usize,
}

/// Comparison between status quo screening and AETRE heavy-tailed triage
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VentureBenchmarkComparison {
    pub total_dealflow_universe: usize,
    pub diligence_budget: usize,
    pub tail_index_alpha: f64,
    pub status_quo: VentureCohortSummary,
    pub aetre_heavy_tailed: VentureCohortSummary,
    pub moic_multiplier_gain: f64,
    pub irr_percentage_point_uplift: f64,
    pub partner_hours_saved_pct: f64,
    pub outlier_capture_improvement_ratio: f64,
}
