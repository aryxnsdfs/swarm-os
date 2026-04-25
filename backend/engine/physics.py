"""
Butterfly Effect Physics Engine
Implements the causal escalation system where every fix creates a new bottleneck.
Manages cluster state, telemetry ticks, budget burn, and SLA tracking.
"""

import copy
import logging
import time

logger = logging.getLogger("swarm-os.physics")

# ── Constants ──
SLA_WINDOW_SECONDS = 600       # 10 minutes
TICK_INTERVAL_SECONDS = 1
INITIAL_BUDGET = 50.00
HOURLY_BURN_RATE = 2.50


class PhysicsEngine:
    def __init__(self):
        self.reset()

    def reset(self):
        """Reset the physics engine to initial state."""
        self.state = {
            "ram_mb": 320,
            "vram_gb": 0.0,
            "network_pct": 25,
            "cpu_pct": 30,
            "container_status": "idle",
            "cluster_status": "healthy",
            "active_nodes": 4,
            "hourly_burn_usd": HOURLY_BURN_RATE,
        }
        self.start_time = time.time()
        self.elapsed_seconds = 0
        self.budget_cap = INITIAL_BUDGET
        self.budget_remaining = INITIAL_BUDGET
        self.sla_remaining = SLA_WINDOW_SECONDS
        self.cost_accrued = 0.0
        self.escalation_history = []
        logger.info("Physics engine reset. Budget=$%.2f, SLA=%ds", INITIAL_BUDGET, SLA_WINDOW_SECONDS)

    def step(self, action: str = None) -> dict:
        """
        Advance the physics engine by one tick.
        If an action is provided, apply its causal effects.
        """
        self.elapsed_seconds += TICK_INTERVAL_SECONDS
        self.sla_remaining = max(0, SLA_WINDOW_SECONDS - self.elapsed_seconds)

        # Burn budget
        tick_cost = self.state["hourly_burn_usd"] / 3600
        self.cost_accrued += tick_cost
        self.budget_remaining = max(0.0, self.budget_cap - self.cost_accrued)

        # Keep the gauges visibly alive during real runs so the dashboard reflects
        # ongoing work even between major state transitions.
        self._apply_tick_pulse()

        # Apply causal effects from actions
        if action:
            self._apply_causal_effect(action)

        return {
            **self.state,
            "elapsed_seconds": self.elapsed_seconds,
            "sla_remaining": self.sla_remaining,
            "budget_remaining": round(self.budget_remaining, 2),
            "cost_accrued": round(self.cost_accrued, 2),
            "hourly_burn_usd": self.state["hourly_burn_usd"],
            "cluster_status": self.state["cluster_status"],
        }

    def _apply_tick_pulse(self):
        pulse_index = self.elapsed_seconds % 4
        ram_deltas = (4, 9, -5, 6)
        vram_deltas = (0.05, 0.08, -0.04, 0.03)
        network_deltas = (3, -2, 4, -1)
        cpu_deltas = (2, 5, -3, 1)

        self.state["ram_mb"] = max(240, min(860, self.state["ram_mb"] + ram_deltas[pulse_index]))
        self.state["vram_gb"] = round(max(0.2, min(11.9, self.state["vram_gb"] + vram_deltas[pulse_index])), 2)
        self.state["network_pct"] = max(8, min(99, self.state["network_pct"] + network_deltas[pulse_index]))
        self.state["cpu_pct"] = max(10, min(98, self.state["cpu_pct"] + cpu_deltas[pulse_index]))

    def _apply_causal_effect(self, action: str):
        """
        Apply the butterfly effect: every fix creates a downstream bottleneck.

        Causal chain:
        - FSDP fix → reduces VRAM, BUT spikes network (all-reduce traffic)
        - gradient_checkpointing → reduces network, BUT increases CPU
        - restart_loop → wastes time, costs money, does not fix root cause
        """
        if action == "fsdp_sharding":
            # FSDP reduces VRAM dramatically but spikes network
            self.state["vram_gb"] = 0.45
            self.state["network_pct"] = 95  # Butterfly effect: all-reduce traffic
            self.state["container_status"] = "running"
            self.state["cluster_status"] = "degraded"  # Network is now the bottleneck
            self.escalation_history.append({
                "cause": "fsdp_sharding",
                "effect": "network_spike_95pct",
                "detail": "FSDP all-reduce traffic saturated inter-node bandwidth",
            })

        elif action == "gradient_checkpointing":
            # Gradient checkpointing reduces VRAM dramatically but increases CPU
            self.state["vram_gb"] = 0.29
            self.state["network_pct"] = 52
            self.state["cpu_pct"] = 55
            self.state["container_status"] = "stable"
            self.state["cluster_status"] = "healthy"
            self.escalation_history.append({
                "cause": "gradient_checkpointing",
                "effect": "vram_reduction_and_cpu_increase",
                "detail": "Recomputing activations saves VRAM (now 0.29GB) but uses more CPU",
            })

        elif action == "restart_loop":
            # Naive restart — wastes time and money, does not fix the issue
            self.state["vram_gb"] = 11.8  # OOM returns immediately
            self.state["container_status"] = "critical"
            self.state["cluster_status"] = "degraded"
            self.state["hourly_burn_usd"] += 1.0  # Cost escalates with each restart

        elif action == "schema_drift":
            # Schema changes break ingestion pipelines
            self.state["cluster_status"] = "degraded"
            self.escalation_history.append({
                "cause": "schema_drift",
                "effect": "ingestion_failure",
                "detail": "JSON schema mutated from flat to nested format",
            })

    def get_telemetry(self) -> dict:
        """Get current telemetry snapshot for AI agent context window."""
        return {
            "active_compute_nodes": self.state["active_nodes"],
            "hourly_burn_usd": self.state["hourly_burn_usd"],
            "sla_breach_penalty": 0 if self.sla_remaining > 0 else 100.0,
            "ram_mb": self.state["ram_mb"],
            "vram_gb": self.state["vram_gb"],
            "network_pct": self.state["network_pct"],
            "cpu_pct": self.state["cpu_pct"],
            "container_status": self.state["container_status"],
            "sla_remaining_seconds": self.sla_remaining,
            "budget_remaining_usd": round(self.budget_remaining, 3),
            "budget_cap_usd": round(self.budget_cap, 3),
            "cost_accrued_usd": round(self.cost_accrued, 3),
        }

    def get_state_snapshot(self) -> dict:
        """Get a deep copy of the current state for counterfactual forking."""
        return copy.deepcopy({
            "state": self.state,
            "elapsed_seconds": self.elapsed_seconds,
            "cost_accrued": self.cost_accrued,
            "sla_remaining": self.sla_remaining,
            "budget_remaining": self.budget_remaining,
            "budget_cap": self.budget_cap,
        })
