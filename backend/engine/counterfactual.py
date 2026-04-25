"""
Counterfactual Simulation Engine
Forks the cluster state at a decision point and replays with a naive action
to project the alternate outcome for the "Dead Timeline" panel.
"""

import copy

SLA_WINDOW_SECONDS = 600
TICK_INTERVAL_SECONDS = 1


def simulate_counterfactual(state_snapshot: dict, naive_action: str) -> dict:
    """
    Forks the cluster state at the first decision point and replays
    with a naive action to project the alternate outcome.

    Args:
        state_snapshot: Deep copy of the engine state at decision point
        naive_action: The naive strategy to simulate (e.g., "restart_loop")

    Returns:
        dict with projected cost, SLA status, time, and outcome
    """
    forked_state = copy.deepcopy(state_snapshot["state"])
    elapsed_time = state_snapshot.get("elapsed_seconds", 0)
    cost_accrued = state_snapshot.get("cost_accrued", 0.0)
    sla_breached = False

    # Replay the naive action through a simplified physics simulation
    while not sla_breached and elapsed_time < SLA_WINDOW_SECONDS:
        # Simulate one tick
        hourly_burn = forked_state.get("hourly_burn_usd", 2.50)
        cost_accrued += hourly_burn / 3600
        elapsed_time += TICK_INTERVAL_SECONDS

        if naive_action == "restart_loop":
            # Naive restart: OOM returns after each restart, cost escalates rapidly
            # due to cascading failure across the load balancers triggering auto-scale 
            forked_state["vram_gb"] = 11.8   # OOM persists
            forked_state["container_status"] = "critical"
            forked_state["cluster_status"] = "degraded"
            hourly_burn += 1.05  # Compounding scaling failure cost ($47.00 total)
            forked_state["hourly_burn_usd"] = hourly_burn

        elif naive_action == "ignore":
            # Ignoring the issue: cluster degrades over time
            forked_state["cluster_status"] = "degraded"

        elif naive_action == "scale_up":
            # Brute force scaling: expensive but slow to provision
            cost_accrued += 5.0 / 3600  # Extra provisioning cost per tick
            if elapsed_time > 300:  # Takes 5 min to provision
                forked_state["cluster_status"] = "healthy"
                break

        # Check SLA breach
        if elapsed_time >= SLA_WINDOW_SECONDS:
            sla_breached = True

    return {
        "projected_cost_usd": round(cost_accrued, 2),
        "sla_breached": sla_breached,
        "time_elapsed_seconds": elapsed_time,
        "time_elapsed_formatted": f"{int(elapsed_time // 60)}m {int(elapsed_time % 60):02d}s",
        "outcome": "CLUSTER_DOWN" if sla_breached else "DEGRADED",
        "naive_action": naive_action,
    }
