from __future__ import annotations

import json
from pathlib import Path

from swarm_openenv_env.environment import IncidentResponseEnv
from swarm_openenv_env.models import IncidentAction
from swarm_openenv_env.tasks import list_task_ids


SMOKE_ACTIONS = {
    "task_easy_gpu_oom": [
        IncidentAction(operation="inspect_artifact", target="telemetry", content=""),
        IncidentAction(operation="inspect_artifact", target="runbook", content=""),
        IncidentAction(
            operation="open_ticket",
            target="incident_ticket",
            content="P1 OOM on single GPU with business impact.",
        ),
        IncidentAction(
            operation="propose_fix",
            target="remediation_plan",
            content="Use autocast and checkpoint to reduce memory.",
        ),
        IncidentAction(
            operation="resolve_incident",
            target="closure_note",
            content="memory stable on single gpu, continue to monitor",
        ),
    ],
    "task_medium_schema_drift": [
        IncidentAction(operation="inspect_artifact", target="schema_diff", content=""),
        IncidentAction(operation="inspect_artifact", target="job_log", content=""),
        IncidentAction(operation="inspect_artifact", target="runbook", content=""),
        IncidentAction(
            operation="open_ticket",
            target="incident_ticket",
            content="P2 schema drift causing stale dashboard data.",
        ),
        IncidentAction(
            operation="send_status_update",
            target="stakeholders",
            content="Dashboard is stale, ETA 45 minutes.",
        ),
        IncidentAction(
            operation="propose_fix",
            target="remediation_plan",
            content="Add mapping for state to status and backfill data.",
        ),
        IncidentAction(
            operation="resolve_incident",
            target="closure_note",
            content="validation passed after backfill and compatible mapping",
        ),
    ],
    "task_hard_canary_regression": [
        IncidentAction(operation="inspect_artifact", target="latency_chart", content=""),
        IncidentAction(operation="inspect_artifact", target="deploy_diff", content=""),
        IncidentAction(operation="inspect_artifact", target="runbook", content=""),
        IncidentAction(
            operation="open_ticket",
            target="incident_ticket",
            content="P1 checkout latency affecting customers in eu-west.",
        ),
        IncidentAction(
            operation="send_status_update",
            target="stakeholders",
            content="Rollback in progress for eu-west impact.",
        ),
        IncidentAction(
            operation="propose_fix",
            target="remediation_plan",
            content="Rollback canary, disable feature flag, and monitor recovery.",
        ),
        IncidentAction(
            operation="resolve_incident",
            target="closure_note",
            content="rollback recovered service and monitoring confirms stability",
        ),
    ],
}


def main() -> None:
    report = {"tasks": [], "passed": True}
    env = IncidentResponseEnv()

    for task_id in list_task_ids():
        obs = env.reset(task_id=task_id)
        for action in SMOKE_ACTIONS[task_id]:
            obs = env.step(action)
            if obs.done:
                break
        score = float(env.state.current_score)
        passed = 0.0 <= score <= 1.0
        report["tasks"].append({"task_id": task_id, "score": score, "passed": passed})
        report["passed"] = report["passed"] and passed

    Path("outputs").mkdir(exist_ok=True)
    Path("outputs/validation_report.json").write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
