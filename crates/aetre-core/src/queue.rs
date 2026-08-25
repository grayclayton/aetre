use serde::{Deserialize, Serialize};

/// Queueing metrics for a single stage in the innovation pipeline
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StageQueueMetrics {
    pub arrival_rate: f64,
    pub service_rate: f64,
    pub utilization: f64,
    pub mean_wait_time: f64,
    pub mean_items_in_queue: f64,
    pub is_congested: bool,
}

/// Computes the Kingman heavy-traffic waiting time approximation
/// E[W_q] = (rho / (1 - rho)) * ((c_a^2 + c_s^2) / 2) * (1 / mu)
pub fn kingman_waiting_time(
    arrival_rate: f64,
    service_rate: f64,
    cv_arrivals: f64,
    cv_service: f64,
) -> f64 {
    if service_rate <= 0.0 {
        return f64::INFINITY;
    }
    let rho = arrival_rate / service_rate;
    if rho >= 1.0 {
        return f64::INFINITY;
    }
    let var_term = (cv_arrivals * cv_arrivals + cv_service * cv_service) / 2.0;
    let mean_service_time = 1.0 / service_rate;
    (rho / (1.0 - rho)) * var_term * mean_service_time
}

/// Evaluates a stage's queueing state and Little's Law backlog
pub fn evaluate_stage_queue(
    arrival_rate: f64,
    service_rate: f64,
    cv_arrivals: f64,
    cv_service: f64,
) -> StageQueueMetrics {
    let utilization = if service_rate > 0.0 {
        arrival_rate / service_rate
    } else {
        1.0
    };

    let mean_wait_time = if utilization < 0.999 {
        kingman_waiting_time(arrival_rate, service_rate, cv_arrivals, cv_service)
    } else {
        // Near-saturation cap for numerical stability
        kingman_waiting_time(service_rate * 0.999, service_rate, cv_arrivals, cv_service)
    };

    let mean_items_in_queue = arrival_rate * mean_wait_time;
    let is_congested = utilization >= 0.85;

    StageQueueMetrics {
        arrival_rate,
        service_rate,
        utilization,
        mean_wait_time,
        mean_items_in_queue,
        is_congested,
    }
}

/// Dynamic Capacity Governor: calculates recommended throttling or automated routing
/// to keep downstream human review utilization rho <= target_utilization (e.g. 0.80)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GovernorAction {
    pub target_utilization: f64,
    pub max_human_throughput: f64,
    pub current_arrival_rate: f64,
    pub excess_arrival_rate: f64,
    pub recommend_automated_triage: bool,
}

pub fn calculate_governor_action(
    arrival_rate: f64,
    human_service_rate: f64,
    target_utilization: f64,
) -> GovernorAction {
    let max_sustainable_arrivals = human_service_rate * target_utilization;
    let excess = (arrival_rate - max_sustainable_arrivals).max(0.0);
    let recommend_automated_triage = arrival_rate > max_sustainable_arrivals;

    GovernorAction {
        target_utilization,
        max_human_throughput: max_sustainable_arrivals,
        current_arrival_rate: arrival_rate,
        excess_arrival_rate: excess,
        recommend_automated_triage,
    }
}

use crate::types::{HeterogeneousSystemMetrics, SpecialistQueueMetrics};

/// Evaluates a heterogeneous network of specialist and generalist reviewer queues
pub fn evaluate_heterogeneous_queues(
    pools: Vec<(String, f64, f64, f64, f64)>, // (domain, arrival_rate, service_rate, cv_a, cv_s)
) -> HeterogeneousSystemMetrics {
    let mut pool_metrics = Vec::new();
    let mut max_utilization: f64 = 0.0;
    let mut bottleneck_domain = String::from("None");
    let mut rebalancing_actions = Vec::new();

    for (domain, arrival_rate, service_rate, cv_a, cv_s) in pools {
        let q = evaluate_stage_queue(arrival_rate, service_rate, cv_a, cv_s);
        if q.utilization > max_utilization {
            max_utilization = q.utilization;
            bottleneck_domain = domain.clone();
        }

        if q.is_congested {
            let sustainable_rate = service_rate * 0.80;
            let excess = (arrival_rate - sustainable_rate).max(0.0);
            rebalancing_actions.push(format!(
                "Pool '{}' is congested (rho={:.2}). Divert {:.1} arrivals/time-unit to automated VOI screening.",
                domain, q.utilization, excess
            ));
        }

        pool_metrics.push(SpecialistQueueMetrics {
            domain,
            arrival_rate,
            service_rate,
            utilization: q.utilization,
            mean_wait_time: q.mean_wait_time,
            is_congested: q.is_congested,
        });
    }

    let is_system_congested = max_utilization >= 0.85;

    HeterogeneousSystemMetrics {
        pools: pool_metrics,
        max_utilization,
        bottleneck_domain,
        is_system_congested,
        rebalancing_actions,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kingman_delay_nonlinearity() {
        let cv_a = 1.0;
        let cv_s = 1.0;
        let mu = 100.0;

        let wait_50 = kingman_waiting_time(50.0, mu, cv_a, cv_s); // rho = 0.5 -> 1.0 * (1/100) = 0.01
        let wait_90 = kingman_waiting_time(90.0, mu, cv_a, cv_s); // rho = 0.9 -> 9.0 * (1/100) = 0.09
        let wait_95 = kingman_waiting_time(95.0, mu, cv_a, cv_s); // rho = 0.95 -> 19.0 * (1/100) = 0.19

        assert!(wait_90 > 8.0 * wait_50);
        assert!(wait_95 > 2.0 * wait_90);
    }

    #[test]
    fn test_governor_action_recommendation() {
        // Below target utilization: no triage needed
        let gov_under = calculate_governor_action(70.0, 100.0, 0.80);
        assert!(!gov_under.recommend_automated_triage);
        assert_eq!(gov_under.excess_arrival_rate, 0.0);

        // Above target utilization: automated triage recommended
        let gov_over = calculate_governor_action(95.0, 100.0, 0.80);
        assert!(gov_over.recommend_automated_triage);
        assert_eq!(gov_over.excess_arrival_rate, 15.0);
    }

    #[test]
    fn test_queue_saturation_guard() {
        let q_sat = evaluate_stage_queue(120.0, 100.0, 1.0, 1.0);
        assert!(q_sat.is_congested);
        assert!(q_sat.mean_wait_time.is_finite());
    }

    #[test]
    fn test_heterogeneous_queues() {
        let pools = vec![
            ("Genomics".into(), 45.0, 50.0, 1.0, 1.0), // rho = 0.90 (congested)
            ("Computer_Science".into(), 30.0, 50.0, 1.0, 1.0), // rho = 0.60
        ];
        let sys = evaluate_heterogeneous_queues(pools);
        assert_eq!(sys.bottleneck_domain, "Genomics");
        assert!(sys.is_system_congested);
        assert_eq!(sys.rebalancing_actions.len(), 1);
    }
}
