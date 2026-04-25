from __future__ import annotations

import os
from uuid import uuid4

import gradio as gr

from swarm_openenv_env.environment import IncidentResponseEnv
from swarm_openenv_env.models import IncidentAction
from swarm_openenv_env.tasks import get_task, list_task_ids


TASK_IDS = list_task_ids()
SESSIONS: dict[str, dict] = {}


TASK_TEMPLATES = {
    "task_easy_gpu_oom": {
        "open_ticket": (
            "incident_ticket",
            "P1 OOM on single GPU training job. Business impact: nightly fine-tune missed SLA and single GPU memory is exhausted.",
        ),
        "propose_fix": (
            "remediation_plan",
            "Enable torch.autocast mixed precision, add gradient checkpoint support, reduce peak memory on the single GPU, and monitor VRAM after rollout.",
        ),
        "resolve_incident": (
            "closure_note",
            "Memory usage is reduced for the single GPU workload. Continue to monitor VRAM and batch-size stability after the change.",
        ),
    },
    "task_medium_schema_drift": {
        "open_ticket": (
            "incident_ticket",
            "P2 schema drift broke the dashboard pipeline. Business impact: customer dashboard numbers are stale.",
        ),
        "send_status_update": (
            "stakeholders",
            "Dashboards are stale. ETA 45 minutes while we apply a compatible fix and validate downstream models.",
        ),
        "propose_fix": (
            "remediation_plan",
            "Add a compatibility mapping from state back to status, backfill the missing partition, and keep downstream reads compatible.",
        ),
        "resolve_incident": (
            "closure_note",
            "Backfill completed, downstream validation passed, and the compatible mapping is in place.",
        ),
    },
    "task_hard_canary_regression": {
        "open_ticket": (
            "incident_ticket",
            "P1 checkout latency regression. Business impact: checkout timeouts for customers in eu-west.",
        ),
        "send_status_update": (
            "stakeholders",
            "Customer impact confirmed in eu-west. Rolling back the canary now; next ETA update in 15 minutes.",
        ),
        "propose_fix": (
            "remediation_plan",
            "Rollback the canary, disable the feature flag, and monitor latency and error-rate recovery before re-enabling anything.",
        ),
        "resolve_incident": (
            "closure_note",
            "Rollback recovered the service. Monitoring remains active to verify latency and error rates stay healthy.",
        ),
    },
}


def _get_session(session_id: str, task_id: str) -> dict:
    session = SESSIONS.get(session_id)
    if session is None:
        session = {
            "env": IncidentResponseEnv(default_task_id=task_id),
            "history": [],
        }
        SESSIONS[session_id] = session
    return session


def _observation_payload(observation) -> dict:
    payload = observation.model_dump()
    payload["reward"] = float(observation.reward or 0.0)
    payload["done"] = bool(observation.done)
    return payload


def _render_history(history: list[dict]) -> str:
    if not history:
        return "_No trajectory yet._"

    lines = []
    for item in history:
        obs = item["observation"]
        sandbox = obs.get("sandbox_result") or {}
        sandbox_bits = ""
        if sandbox:
            label = sandbox.get("validation_label") or "validator"
            sandbox_bits = f" | {label}={sandbox.get('status')}"
            if sandbox.get("gpu_metrics_applicable"):
                sandbox_bits += f" | vram={sandbox.get('vram_peak_mb', 0)}MB"
            else:
                sandbox_bits += " | gpu=n/a"
            checks = sandbox.get("checks_applied") or []
            if checks:
                sandbox_bits += f" | checks={','.join(checks)}"
        lines.append(
            f"{item['step']:02d}. `{item['action']['operation']}` by "
            f"`{obs.get('active_agent', 'UNKNOWN')}` -> {obs.get('last_feedback', '')}"
            f"{sandbox_bits}"
        )
    return "\n".join(lines)


def _render_status(task_id: str, observation: dict, state: dict) -> str:
    telemetry = observation.get("telemetry") or {}
    sandbox = observation.get("sandbox_result") or {}
    health = telemetry.get("sandbox_health") or {}
    validator = telemetry.get("validator_runtime") or {}
    validator_label = validator.get("label", "unknown")
    validator_detail = validator.get("detail", "No validator detail available.")
    validation_scope = validator.get("validation_scope", "Unavailable")
    last_validator_status = sandbox.get("status", "n/a")
    if sandbox and not sandbox.get("gpu_metrics_applicable"):
        last_validator_status = f"{last_validator_status} (GPU N/A)"
    checks_applied = ", ".join(sandbox.get("checks_applied") or []) or "n/a"
    return (
        f"### {observation.get('title', task_id)}\n"
        f"- Task ID: `{task_id}`\n"
        f"- Active Agent: `{observation.get('active_agent', 'COMMANDER')}`\n"
        f"- Assigned Agents: `{', '.join(observation.get('assigned_agents', []))}`\n"
        f"- Score: `{observation.get('reward', 0.0):.3f}`\n"
        f"- Steps Remaining: `{observation.get('steps_remaining', 0)}`\n"
        f"- Done: `{observation.get('done', False)}`\n"
        f"- Validator: `{validator_label}`\n"
        f"- Validator Ready: `{health.get('ready', False)}`\n"
        f"- Validation Scope: `{validation_scope}`\n"
        f"- Container Status: `{telemetry.get('container_status', 'unknown')}`\n"
        f"- VRAM: `{telemetry.get('vram_gb', 0)} GB` | Network: `{telemetry.get('network_pct', 0)}%` | CPU: `{telemetry.get('cpu_pct', 0)}%`\n"
        f"- Last Validator Status: `{last_validator_status}`\n"
        f"- Checks Applied: `{checks_applied}`\n"
        f"- Validator Detail: {validator_detail}"
    )


def _default_target_and_content(task_id: str, operation: str, env: IncidentResponseEnv):
    task = get_task(task_id)
    if operation == "inspect_artifact":
        for artifact in task.required_artifacts:
            if artifact not in env.state.seen_artifacts:
                return artifact, ""
        return next(iter(task.artifacts.keys())), ""

    template = TASK_TEMPLATES.get(task_id, {}).get(operation)
    if template:
        return template

    return "", ""


def _snapshot(session: dict, observation) -> tuple[str, dict, dict, str]:
    obs_payload = _observation_payload(observation)
    state_payload = session["env"].state.model_dump()
    return (
        _render_status(session["env"].state.task_id, obs_payload, state_payload),
        obs_payload,
        state_payload,
        _render_history(session["history"]),
    )


def reset_env(session_id: str, task_id: str):
    session = _get_session(session_id, task_id)
    session["env"] = IncidentResponseEnv(default_task_id=task_id)
    session["history"] = []
    observation = session["env"].reset(task_id=task_id)
    status, obs_payload, state_payload, history_md = _snapshot(session, observation)
    target, content = _default_target_and_content(task_id, "inspect_artifact", session["env"])
    return status, obs_payload, state_payload, history_md, target, content


def autofill_action(session_id: str, task_id: str, operation: str):
    session = _get_session(session_id, task_id)
    if session["env"].state.task_id != task_id:
        session["env"] = IncidentResponseEnv(default_task_id=task_id)
        session["history"] = []
        session["env"].reset(task_id=task_id)
    return _default_target_and_content(task_id, operation, session["env"])


def run_step(session_id: str, task_id: str, operation: str, target: str, content: str):
    session = _get_session(session_id, task_id)
    env = session["env"]
    if env.state.task_id != task_id:
        env = IncidentResponseEnv(default_task_id=task_id)
        session["env"] = env
        session["history"] = []
        env.reset(task_id=task_id)

    action = IncidentAction(operation=operation, target=target, content=content)
    observation = env.step(action)
    session["history"].append(
        {
            "step": env.state.step_count,
            "action": action.model_dump(),
            "observation": _observation_payload(observation),
        }
    )
    status, obs_payload, state_payload, history_md = _snapshot(session, observation)
    next_target, next_content = _default_target_and_content(task_id, operation, env)
    return status, obs_payload, state_payload, history_md, next_target, next_content


def _guided_action(env: IncidentResponseEnv) -> IncidentAction:
    task = get_task(env.state.task_id)
    for artifact in task.required_artifacts:
        if artifact not in env.state.seen_artifacts:
            return IncidentAction(
                operation="inspect_artifact",
                target=artifact,
                content="",
            )

    if not env.state.ticket_opened:
        target, content = TASK_TEMPLATES[task.task_id]["open_ticket"]
        return IncidentAction(operation="open_ticket", target=target, content=content)

    if task.required_status_keywords and not env.state.status_shared:
        target, content = TASK_TEMPLATES[task.task_id]["send_status_update"]
        return IncidentAction(
            operation="send_status_update",
            target=target,
            content=content,
        )

    if not env.state.sandbox_passed:
        target, content = TASK_TEMPLATES[task.task_id]["propose_fix"]
        return IncidentAction(operation="propose_fix", target=target, content=content)

    target, content = TASK_TEMPLATES[task.task_id]["resolve_incident"]
    return IncidentAction(operation="resolve_incident", target=target, content=content)


def run_auto_demo(session_id: str, task_id: str):
    session = _get_session(session_id, task_id)
    session["env"] = IncidentResponseEnv(default_task_id=task_id)
    session["history"] = []
    env = session["env"]
    observation = env.reset(task_id=task_id)

    while not observation.done and env.state.step_count < env.state.max_steps:
        action = _guided_action(env)
        observation = env.step(action)
        session["history"].append(
            {
                "step": env.state.step_count,
                "action": action.model_dump(),
                "observation": _observation_payload(observation),
            }
        )

    status, obs_payload, state_payload, history_md = _snapshot(session, observation)
    target, content = _default_target_and_content(task_id, "inspect_artifact", env)
    return status, obs_payload, state_payload, history_md, target, content


with gr.Blocks(title="Swarm Incident Response OpenEnv", theme=gr.themes.Soft()) as demo:
    session_id = gr.State(str(uuid4()))

    gr.Markdown(
        """
        # Swarm Incident Response OpenEnv
        A compact Space for the OpenEnv submission. The environment automatically assigns
        the Swarm roster and validates fixes with an honest runtime label.

        OOM incidents use the Docker GPU validator with VRAM checks.
        Schema and canary incidents use Docker plain-Python validation, where GPU VRAM
        metrics are not applicable.
        """
    )

    with gr.Row():
        with gr.Column(scale=1):
            task_id = gr.Dropdown(
                choices=TASK_IDS,
                value=TASK_IDS[0],
                label="Task",
            )
            operation = gr.Dropdown(
                choices=[
                    "inspect_artifact",
                    "open_ticket",
                    "send_status_update",
                    "propose_fix",
                    "resolve_incident",
                ],
                value="inspect_artifact",
                label="Operation",
            )
            target = gr.Textbox(label="Target")
            content = gr.Textbox(label="Content", lines=8)

            with gr.Row():
                reset_btn = gr.Button("Reset")
                autofill_btn = gr.Button("Autofill")
                step_btn = gr.Button("Run Step", variant="primary")
            auto_demo_btn = gr.Button("Run Auto Demo")

        with gr.Column(scale=2):
            status_md = gr.Markdown()
            with gr.Row():
                observation_json = gr.JSON(label="Observation")
                state_json = gr.JSON(label="State")
            gr.Markdown("### Trajectory")
            history_md = gr.Markdown()

    reset_btn.click(
        reset_env,
        inputs=[session_id, task_id],
        outputs=[status_md, observation_json, state_json, history_md, target, content],
    )
    autofill_btn.click(
        autofill_action,
        inputs=[session_id, task_id, operation],
        outputs=[target, content],
    )
    step_btn.click(
        run_step,
        inputs=[session_id, task_id, operation, target, content],
        outputs=[status_md, observation_json, state_json, history_md, target, content],
    )
    auto_demo_btn.click(
        run_auto_demo,
        inputs=[session_id, task_id],
        outputs=[status_md, observation_json, state_json, history_md, target, content],
    )
    demo.load(
        reset_env,
        inputs=[session_id, task_id],
        outputs=[status_md, observation_json, state_json, history_md, target, content],
    )


def main() -> None:
    demo.launch(server_name="0.0.0.0", server_port=int(os.getenv("PORT", "7860")))


if __name__ == "__main__":
    main()
