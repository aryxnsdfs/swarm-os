from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


Difficulty = Literal["easy", "medium", "hard"]


@dataclass(frozen=True)
class TaskSpec:
    task_id: str
    difficulty: Difficulty
    title: str
    objective: str
    incident_summary: str
    artifacts: dict[str, str]
    required_artifacts: tuple[str, ...]
    required_ticket_keywords: tuple[str, ...]
    required_fix_keywords: tuple[str, ...]
    required_resolution_keywords: tuple[str, ...]
    required_status_keywords: tuple[str, ...] = ()
    max_steps: int = 6
    success_threshold: float = 0.8


TASKS: dict[str, TaskSpec] = {
    "task_easy_gpu_oom": TaskSpec(
        task_id="task_easy_gpu_oom",
        difficulty="easy",
        title="Single-GPU OOM Triage",
        objective=(
            "Triage a training outage on a single GPU, file the incident, and "
            "propose a safe memory-saving remediation."
        ),
        incident_summary=(
            "A nightly fine-tuning job failed with CUDA OOM on a single A10 GPU. "
            "The team needs an audit-friendly remediation plan."
        ),
        artifacts={
            "telemetry": (
                "GPU peak memory reached 11.8GB / 12GB. Batch size 32. "
                "Single GPU runner only."
            ),
            "traceback": (
                "torch.cuda.OutOfMemoryError while allocating forward activations "
                "during layer 24."
            ),
            "runbook": (
                "Approved mitigations: mixed precision/autocast, gradient "
                "checkpointing, smaller batch size. Not approved: FSDP or DDP "
                "because this workload is single-GPU."
            ),
        },
        required_artifacts=("telemetry", "runbook"),
        required_ticket_keywords=("p1", "oom", "single gpu"),
        required_fix_keywords=("autocast", "checkpoint"),
        required_resolution_keywords=("memory", "single gpu", "monitor"),
        max_steps=6,
    ),
    "task_medium_schema_drift": TaskSpec(
        task_id="task_medium_schema_drift",
        difficulty="medium",
        title="Analytics Schema Drift",
        objective=(
            "Restore a broken customer analytics pipeline by identifying schema "
            "drift, communicating impact, and proposing a safe compatibility fix."
        ),
        incident_summary=(
            "Customer health dashboards stopped updating after an upstream API "
            "change. Downstream analysts need a remediation plan."
        ),
        artifacts={
            "schema_diff": (
                "Upstream payload renamed field 'status' to 'state'. "
                "Nullability is unchanged."
            ),
            "job_log": (
                "dbt model failed: column status does not exist in events_staging."
            ),
            "runbook": (
                "Preferred remediation: add backward-compatible mapping, backfill "
                "the missing partition, and validate the downstream model before "
                "closing the incident."
            ),
            "stakeholder_note": (
                "Customer success needs an ETA because dashboard numbers are stale."
            ),
        },
        required_artifacts=("schema_diff", "job_log", "runbook"),
        required_ticket_keywords=("p2", "schema drift", "dashboard"),
        required_status_keywords=("eta", "dashboard", "stale"),
        required_fix_keywords=("mapping", "backfill", "status"),
        required_resolution_keywords=("validation", "backfill", "compatible"),
        max_steps=15,
    ),
    "task_hard_canary_regression": TaskSpec(
        task_id="task_hard_canary_regression",
        difficulty="hard",
        title="Canary Rollout Regression",
        objective=(
            "Mitigate a canary deployment that caused customer-visible latency, "
            "communicate status, and close the loop with a safe rollback plan."
        ),
        incident_summary=(
            "A checkout-service canary increased p95 latency and error rates in one "
            "region. The on-call needs a safe mitigation and stakeholder update."
        ),
        artifacts={
            "latency_chart": (
                "p95 latency climbed from 220ms to 1400ms immediately after canary "
                "deploy. Error rate rose to 6.4%."
            ),
            "deploy_diff": (
                "Canary enabled a new recommendation enrichment call guarded by "
                "feature flag enrich_checkout_recs=true."
            ),
            "customer_tickets": (
                "Merchants report intermittent checkout timeouts in eu-west."
            ),
            "runbook": (
                "Safe response: rollback the canary or disable the feature flag, "
                "post a status update, watch metrics for recovery, and only then "
                "close the incident."
            ),
        },
        required_artifacts=("latency_chart", "deploy_diff", "runbook"),
        required_ticket_keywords=("p1", "latency", "checkout"),
        required_status_keywords=("rollback", "impact", "eu-west"),
        required_fix_keywords=("rollback", "feature flag", "monitor"),
        required_resolution_keywords=("recovered", "monitoring", "rollback"),
        max_steps=15,
    ),
}


def get_task(task_id: str) -> TaskSpec:
    try:
        return TASKS[task_id]
    except KeyError as exc:
        raise KeyError(f"Unknown task_id: {task_id}") from exc


def list_task_ids() -> list[str]:
    return list(TASKS.keys())
