//! # AETRE: Adaptive Epistemic Triage & Recall Engine
//!
//! A high-performance mathematical and operations research engine for
//! resource-constrained innovation screening, Bayesian Value-of-Information (VOI) triage,
//! Kingman heavy-traffic capacity regulation, Proposition 1 recall bounds, and
//! randomized exploration audits.

pub mod audit;
pub mod calibration;
pub mod investment;
pub mod matching;
pub mod multi_attribute;
pub mod queue;
pub mod recall;
pub mod sequential;
pub mod simulation;
pub mod staking;
pub mod types;
pub mod voi;

// Re-export key interfaces
pub use audit::{calculate_exploration_audit, AuditResult};
pub use calibration::{
    calculate_brier_score, calculate_expected_calibration_error, evaluate_text_robustness,
    PerturbationRobustnessReport, PlattCalibrator,
};
pub use investment::{evaluate_venture_benchmark, generate_synthetic_venture_dealflow};
pub use matching::optimize_congestion_matching;
pub use multi_attribute::evaluate_multi_attribute_voi;
pub use queue::{
    calculate_governor_action, evaluate_heterogeneous_queues, evaluate_stage_queue,
    kingman_waiting_time, GovernorAction, StageQueueMetrics,
};
pub use recall::{
    calculate_proposition_1_bound, generate_recall_scaling_curve, RecallScalingPoint,
    ThroughputRecallBound,
};
pub use sequential::evaluate_sequential_stopping;
pub use simulation::{
    generate_candidates, run_benchmark_replications, run_unmanaged_screening, run_voi_screening,
};
pub use staking::{
    evaluate_quadratic_staking, evaluate_submitter_equilibrium, generate_staking_curve,
    StakingCurvePoint, SubmitterEquilibrium,
};
pub use types::{
    AgentEvaluation, AuthorPreflightReport, Candidate, CandidateStatus, CongestionMatchingResult,
    CorrelatedUpdateResult, DimensionVoiContribution, HeavyTailVoiResult,
    HeterogeneousSystemMetrics, MatchAssignment, MultiAttributeCandidate, MultiAttributeDimension,
    MultiAttributeVoiResult, Parameters, ProposalRequirement, QuadraticStakingResult, RegimeStats,
    ReviewerProfile, ReviewerUtilizationReport, SelectionSummary, SequentialDecision,
    SequentialReviewStep, SequentialStoppingResult, SpecialistQueueMetrics,
    VentureBenchmarkComparison, VentureCohortSummary, VentureDealCandidate,
};
pub use voi::{
    calculate_boundary_voi, calculate_heavy_tailed_voi, correlated_posterior_update,
    evaluate_author_preflight, normal_cdf, normal_pdf, posterior_update,
};
