"""
Snorkel AI Auto-Labeler (Snorkel AI Bonus)
Every Docker execution result is automatically written to a JSONL file.
Creates a self-generating, programmatically labeled training dataset
that Snorkel's pipeline can consume directly -- zero human annotation.

The metadata field enables Snorkel labeling functions to operate on
structured reasons, not just binary pass/fail signals.
"""

import logging
import jsonlines

logger = logging.getLogger("swarm-os.snorkel")


def log_execution_result(scenario_id: str, agent_action: dict,
                          result: dict, reward: float):
    """
    Log a single execution result to the Snorkel-compatible JSONL dataset.

    Each record carries full causal context so that Snorkel labeling functions
    can write rules like:
        if metadata.error_type == "OOM"
        and metadata.fix_strategy == "FSDP"
        and label == "positive"
        → high_confidence_positive

    Args:
        scenario_id: ID of the current scenario run
        agent_action: Dict with role, strategy, and code details
        result: Dict with status, vram, error info, causal triggers
        reward: Computed reward value for this action
    """
    record = {
        "scenario_id": scenario_id,
        "agent_action": agent_action,
        "vram_peak_gb": result["vram_peak_gb"],
        "sandbox_outcome": result["status"],   # "PASS" | "OOMKilled" | "SYNTAX_ERR"
        "reward": reward,
        "label": "positive" if reward > 0 else "negative",
        "metadata": {
            "error_type": result.get("error_type"),        # e.g. "OOM" | "TCP_TIMEOUT" | "SCHEMA_DRIFT"
            "fix_strategy": agent_action.get("strategy"),  # e.g. "FSDP" | "gradient_checkpointing"
            "causal_trigger": result.get("causal_trigger"),# e.g. "network_spike_post_fsdp"
            "sla_status": result.get("sla_status"),        # e.g. "SAFE" | "BREACHED"
            "agent_role": agent_action.get("role"),        # e.g. "Builder" | "DB_Admin"
            "episode": result.get("episode_id")
        }
    }
    with jsonlines.open("swarm_dataset.jsonl", mode="a") as writer:
        writer.write(record)
    logger.info("SNORKEL | scenario=%s | agent=%s | outcome=%s | label=%s | reward=%+.2f",
                scenario_id, agent_action.get("role"), result["status"],
                record["label"], reward)
