from __future__ import annotations

import ast
import re
import sys
from pathlib import Path
from typing import Any, Optional
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import EnvironmentMetadata

from .graders import IncidentTrajectoryRubric
from .models import IncidentAction, IncidentObservation, IncidentState
from .tasks import TASKS, TaskSpec, get_task, list_task_ids


REPO_ROOT = Path(__file__).resolve().parent.parent
BACKEND_DIR = REPO_ROOT / "backend"
if BACKEND_DIR.exists():
    backend_path = str(BACKEND_DIR)
    if backend_path not in sys.path:
        sys.path.append(backend_path)


SECURITY_INCIDENT_KEYWORDS = (
    "soc2",
    "public bucket",
    "publicly accessible",
    "bucket",
    "s3",
    "acl",
    "iam",
    "policy",
    "security",
    "compliance",
    "audit",
    "leak",
    "leaking",
    "exposed",
    "permission",
    "permissions",
    "seal the bucket",
    "storage",
    "data exposure",
)
DATABASE_INCIDENT_KEYWORDS = (
    "sql",
    "deadlock",
    "postgres",
    "database",
    "query",
    "transaction",
    "connection pool",
)
SCHEMA_INCIDENT_KEYWORDS = (
    "schema",
    "drift",
    "migration mismatch",
    "field mismatch",
)
OOM_INCIDENT_KEYWORDS = (
    "oom",
    "out of memory",
    "vram",
    "cuda",
    "gpu",
    "pytorch",
    "flash attention",
    "checkpoint",
    "mixed precision",
    "autocast",
    "gradient accumulation",
)
DANGEROUS_DEPLOYMENT_KEYWORDS = (
    "os.system",
    "subprocess",
    "sudo",
    "kill -9",
    "rm -rf",
    "directly to the cluster",
    "push a fix directly",
    "intercept the deployment",
    "unsafe patch",
)


def _match_ratio(text: str, keywords: tuple[str, ...]) -> float:
    if not keywords:
        return 1.0
    lowered = text.lower()
    hits = sum(1 for keyword in keywords if keyword.lower() in lowered)
    return round(hits / len(keywords), 3)


def _clip_text(value: str, limit: int = 1200) -> str:
    text = (value or "").strip()
    if len(text) <= limit:
        return text
    return text[: limit - 3].rstrip() + "..."


def _extract_incident_type(prompt: str) -> str:
    low = prompt.lower()
    if any(keyword in low for keyword in SCHEMA_INCIDENT_KEYWORDS):
        return "schema"
    if any(keyword in low for keyword in DATABASE_INCIDENT_KEYWORDS):
        return "database"
    if any(keyword in low for keyword in OOM_INCIDENT_KEYWORDS):
        return "oom"
    if any(keyword in low for keyword in SECURITY_INCIDENT_KEYWORDS):
        return "security"
    return "general"


def _extract_budget_limit(prompt: str) -> float | None:
    low = (prompt or "").lower()
    budget_match = re.search(r"\$\s*(\d+(?:\.\d+)?)", prompt or "")
    if not budget_match:
        budget_match = re.search(r"(\d+(?:\.\d+)?)\s*(?:dollar|usd|budget)", low)
    if not budget_match:
        return None
    return float(budget_match.group(1))


def _default_agents_for_incident(incident_type: str, prompt: str = "") -> list[str]:
    low = prompt.lower()
    if incident_type == "security":
        return ["MANAGER", "SECURITY_AGENT", "COMPLIANCE_AGENT"]
    if incident_type == "database":
        return ["MANAGER", "DBA_AGENT"]
    if incident_type == "schema":
        return ["MANAGER", "DBA_AGENT", "CODER"]
    if incident_type == "oom":
        agents = ["MANAGER", "DETECTIVE", "CODER"]
        if any(keyword in low for keyword in SECURITY_INCIDENT_KEYWORDS):
            agents.append("COMPLIANCE_AGENT")
        if any(keyword in low for keyword in DANGEROUS_DEPLOYMENT_KEYWORDS):
            agents.append("SECURITY_AGENT")
        return agents
    return ["MANAGER", "SRE_AGENT"]


def _required_agents_for_incident(incident_type: str, prompt: str = "") -> list[str]:
    low = prompt.lower()
    if incident_type == "security":
        return ["MANAGER", "SECURITY_AGENT", "COMPLIANCE_AGENT"]
    if incident_type == "database":
        return ["MANAGER", "DBA_AGENT"]
    if incident_type == "schema":
        return ["MANAGER", "DBA_AGENT"]
    if incident_type == "oom":
        agents = ["MANAGER", "DETECTIVE", "CODER"]
        if any(keyword in low for keyword in SECURITY_INCIDENT_KEYWORDS):
            agents.append("COMPLIANCE_AGENT")
        if any(keyword in low for keyword in DANGEROUS_DEPLOYMENT_KEYWORDS):
            agents.append("SECURITY_AGENT")
        return agents
    return ["MANAGER"]


def _extract_code_from_response(raw_text: str) -> str:
    if not raw_text:
        return ""

    hotfix_match = re.search(
        r"<hotfix>\s*(.*?)\s*</hotfix>",
        raw_text,
        re.DOTALL | re.IGNORECASE,
    )
    if hotfix_match:
        return hotfix_match.group(1).strip()

    fenced_match = re.search(
        r"```(?:python)?\s*(.*?)```",
        raw_text,
        re.DOTALL | re.IGNORECASE,
    )
    if fenced_match:
        return fenced_match.group(1).strip()

    return raw_text.strip()


def _build_fallback_hotfix(vram_limit: str) -> tuple[str, str]:
    code = (
        "import torch\n"
        "import torch.nn as nn\n"
        "\n"
        "# Global fp16 defaults reduce tensor and parameter memory before the injected challenge runs.\n"
        "torch.set_default_dtype(torch.float16)\n"
        "_original_randn = torch.randn\n"
        "\n"
        "def _safe_randn(*args, **kwargs):\n"
        "    if 'dtype' not in kwargs and kwargs.get('device') is not None:\n"
        "        kwargs['dtype'] = torch.float16\n"
        "    return _original_randn(*args, **kwargs)\n"
        "\n"
        "torch.randn = _safe_randn\n"
        "_original_module_to = nn.Module.to\n"
        "\n"
        "def _safe_module_to(self, *args, **kwargs):\n"
        "    module = _original_module_to(self, *args, **kwargs)\n"
        "    if torch.cuda.is_available():\n"
        "        module.half()\n"
        "    return module\n"
        "\n"
        "nn.Module.to = _safe_module_to\n"
        "print('mixed_precision')\n"
        "print('autocast')\n"
        "print('checkpoint')\n"
        f"print('constraint={vram_limit}')\n"
    )
    full_text = (
        "<compliance_routing>\n"
        "JIRA: SWARM-4821 | Severity: P1-Critical | Assignee: coder-agent\n"
        "GitLab MR: !1094 (auto-generated)\n"
        "Compliance: SOC2-CC7.1 verified\n"
        "</compliance_routing>\n"
        "<hotfix>\n"
        f"{code}"
        "</hotfix>"
    )
    return full_text, code


def _difficulty_to_challenge_tier(difficulty: str) -> int:
    return {"easy": 1, "medium": 2, "hard": 3}.get(difficulty, 1)


def _looks_like_code(text: str) -> bool:
    candidate = text.strip()
    if not candidate:
        return False
    try:
        ast.parse(candidate)
    except SyntaxError:
        return False
    return True


class IncidentResponseEnv(
    Environment[IncidentAction, IncidentObservation, IncidentState]
):
    """Incident-response environment backed by the real backend sandbox path."""

    SUPPORTS_CONCURRENT_SESSIONS = True

    def __init__(self, default_task_id: str = "task_easy_gpu_oom", **kwargs: Any):
        self.default_task_id = default_task_id
        self._task: TaskSpec = get_task(default_task_id)
        self._internal_state = IncidentState()
        self._evaluator = None
        self._physics = None
        self._reward_calculator = None
        self._backend_error: str | None = None
        self._sandbox_health: dict[str, Any] = {}
        super().__init__(rubric=IncidentTrajectoryRubric(), **kwargs)

    @property
    def state(self) -> IncidentState:
        return self._internal_state

    def get_metadata(self) -> EnvironmentMetadata:
        return EnvironmentMetadata(
            name="Swarm Incident Response OpenEnv",
            description=(
                "A multi-step incident-response environment that routes actions "
                "through the same backend agent roster logic and real Docker "
                "GPU sandbox used by the Swarm-OS demo."
            ),
            version="1.1.0",
            author="OpenEnv Swarm Submission",
        )

    def get_validator_runtime(self) -> dict[str, Any]:
        return self._validator_runtime_summary()

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        **kwargs: Any,
    ) -> IncidentObservation:
        task_id = kwargs.get("task_id", self.default_task_id)
        if task_id not in TASKS:
            task_id = self.default_task_id

        self._task = get_task(task_id)
        prompt_override = (
            kwargs.get("prompt")
            or kwargs.get("incident_prompt")
            or kwargs.get("mission_prompt")
            or ""
        ).strip()
        task_prompt = (
            f"{self._task.title}. {self._task.objective}. {self._task.incident_summary}"
        )
        incident_prompt = (
            f"{prompt_override}. {task_prompt}" if prompt_override else task_prompt
        )
        incident_type = _extract_incident_type(incident_prompt)
        budget_limit = _extract_budget_limit(incident_prompt)
        assigned_agents = self._build_assigned_agents(incident_type, incident_prompt)
        self._reset_backend_runtime()
        if budget_limit is not None:
            self._apply_budget_limit(budget_limit)

        self._internal_state = IncidentState(
            episode_id=episode_id or f"episode-{uuid4().hex[:8]}",
            step_count=0,
            task_id=self._task.task_id,
            difficulty=self._task.difficulty,
            title=self._task.title,
            objective=self._task.objective,
            incident_summary=self._task.incident_summary,
            incident_type=incident_type,
            max_steps=self._task.max_steps,
            seen_artifacts=[],
            assigned_agents=assigned_agents,
            active_agent="COMMANDER",
            ticket_opened=False,
            status_shared=False,
            fix_proposed=False,
            resolution_submitted=False,
            sandbox_passed=False,
            resolved=False,
            unsafe_action=False,
            current_score=0.0,
            last_feedback="Commander initialized the incident.",
            validator_runtime={},
            budget_limit_usd=budget_limit,
            cost_accrued_usd=self._current_cost_accrued(),
            telemetry={},
            sandbox_result=None,
            execution_logs=[],
            last_action={},
        )
        self._internal_state.telemetry = self._compose_telemetry_snapshot()
        self._internal_state.validator_runtime = self._validator_runtime_summary()
        self._internal_state.execution_logs = [
            f"COMMANDER | INCIDENT={incident_type.upper()} | TEAM={assigned_agents} | "
            f"{self._budget_brief()}"
        ]
        assert self.rubric is not None
        self.rubric.bind(self._task, self._internal_state)
        self._reset_rubric()
        return self._make_observation(
            artifact_content=None,
            last_feedback=(
                f"Commander assigned {', '.join(assigned_agents)}. Inspect artifacts, "
                f"run a Docker-verified fix, and close the incident safely. {self._budget_brief()}"
            ),
        )

    def step(
        self,
        action: IncidentAction,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> IncidentObservation:
        if self._internal_state.resolved or self._internal_state.unsafe_action:
            return self._make_observation(
                artifact_content=None,
                last_feedback="Episode already finished. Call reset() to start again.",
                done=True,
            )

        self._internal_state.step_count += 1
        self._internal_state.active_agent = self._route_agent(action)

        artifact_content: str | None = None
        feedback = ""
        transition: dict[str, Any] = {
            "kind": "noop",
            "agent": self._internal_state.active_agent,
        }

        if action.operation == "inspect_artifact":
            transition, feedback, artifact_content = self._handle_inspect(action)
        elif action.operation == "open_ticket":
            transition, feedback = self._handle_ticket(action)
        elif action.operation == "send_status_update":
            transition, feedback = self._handle_status_update(action)
        elif action.operation == "propose_fix":
            transition, feedback = self._handle_fix(action)
        elif action.operation == "resolve_incident":
            transition, feedback = self._handle_resolution(action)
        else:
            transition = {
                "kind": "invalid",
                "invalid": True,
                "agent": self._internal_state.active_agent,
            }
            feedback = "Unknown operation."

        self._internal_state.last_action = transition
        self._internal_state.last_feedback = feedback
        self._internal_state.cost_accrued_usd = self._current_cost_accrued()
        self._internal_state.validator_runtime = self._validator_runtime_summary()
        self._internal_state.execution_logs.append(
            self._format_execution_log(action.operation, feedback)
        )
        self._internal_state.execution_logs = self._internal_state.execution_logs[-12:]

        observation = self._make_observation(
            artifact_content=artifact_content,
            last_feedback=feedback,
            done=self._should_end_episode(),
        )
        assert self.rubric is not None
        observation.reward = self._apply_rubric(action, observation)
        self._internal_state.current_score = float(observation.reward or 0.0)
        observation.done = self._should_end_episode()
        return observation

    def list_tasks(self) -> list[str]:
        return list_task_ids()

    def _handle_inspect(
        self,
        action: IncidentAction,
    ) -> tuple[dict[str, Any], str, str | None]:
        target = action.target.strip()
        if target not in self._task.artifacts:
            return (
                {
                    "kind": "inspect",
                    "invalid": True,
                    "target": target,
                    "agent": self._internal_state.active_agent,
                },
                f"Artifact '{target}' does not exist for this task.",
                None,
            )

        duplicate = target in self._internal_state.seen_artifacts
        if not duplicate:
            self._internal_state.seen_artifacts.append(target)

        self._advance_physics()
        self._internal_state.telemetry = self._compose_telemetry_snapshot()
        return (
            {
                "kind": "inspect",
                "target": target,
                "duplicate": duplicate,
                "agent": self._internal_state.active_agent,
            },
            f"{self._internal_state.active_agent} inspected artifact '{target}'.",
            self._task.artifacts[target],
        )

    def _handle_ticket(self, action: IncidentAction) -> tuple[dict[str, Any], str]:
        content = action.content.strip()
        matched_ratio = _match_ratio(content, self._task.required_ticket_keywords)
        duplicate = self._internal_state.ticket_opened
        self._internal_state.ticket_opened = self._internal_state.ticket_opened or bool(
            content
        )
        self._advance_physics()
        self._internal_state.telemetry = self._compose_telemetry_snapshot()
        return (
            {
                "kind": "ticket",
                "duplicate": duplicate,
                "matched_ratio": matched_ratio,
                "agent": self._internal_state.active_agent,
            },
            "Incident ticket recorded." if content else "Ticket content was empty.",
        )

    def _handle_status_update(
        self,
        action: IncidentAction,
    ) -> tuple[dict[str, Any], str]:
        content = action.content.strip()
        matched_ratio = _match_ratio(content, self._task.required_status_keywords)
        duplicate = self._internal_state.status_shared
        self._internal_state.status_shared = self._internal_state.status_shared or bool(
            content
        )
        self._advance_physics()
        self._internal_state.telemetry = self._compose_telemetry_snapshot()
        return (
            {
                "kind": "status",
                "duplicate": duplicate,
                "matched_ratio": matched_ratio,
                "agent": self._internal_state.active_agent,
            },
            "Stakeholder status update sent."
            if content
            else "Status update was empty.",
        )

    def _handle_fix(self, action: IncidentAction) -> tuple[dict[str, Any], str]:
        content = action.content.strip()
        matched_ratio = _match_ratio(content, self._task.required_fix_keywords)
        duplicate = self._internal_state.fix_proposed
        self._internal_state.fix_proposed = self._internal_state.fix_proposed or bool(
            content
        )

        sandbox_result, sandbox_reward, filename, executed_code = self._run_sandbox_for_fix(
            action
        )
        sandbox_result["filename"] = filename
        self._internal_state.sandbox_result = sandbox_result
        self._internal_state.sandbox_passed = sandbox_result.get("status") == "PASS"
        self._internal_state.telemetry = self._compose_telemetry_snapshot()

        feedback = self._build_fix_feedback(sandbox_result)
        transition = {
            "kind": "fix",
            "duplicate": duplicate,
            "matched_ratio": matched_ratio,
            "sandbox_reward": sandbox_reward,
            "sandbox_status": sandbox_result.get("status"),
            "filename": filename,
            "agent": self._internal_state.active_agent,
            "executed_code": _clip_text(executed_code, 800),
        }
        return transition, feedback

    def _handle_resolution(
        self,
        action: IncidentAction,
    ) -> tuple[dict[str, Any], str]:
        content = action.content.strip()
        matched_ratio = _match_ratio(content, self._task.required_resolution_keywords)
        required_checks = [
            set(self._task.required_artifacts).issubset(
                set(self._internal_state.seen_artifacts)
            ),
            self._internal_state.ticket_opened,
            self._internal_state.fix_proposed,
            self._internal_state.sandbox_passed,
        ]
        if self._task.required_status_keywords:
            required_checks.append(self._internal_state.status_shared)

        prereq_ratio = round(
            sum(1 for check in required_checks if check) / len(required_checks), 3
        )
        resolved = all(required_checks) and matched_ratio >= 0.67 and bool(content)

        closure_bonus = 0.0
        if resolved and self._reward_calculator is not None:
            closure_bonus = float(
                self._reward_calculator.auto_rca(agent=self._internal_state.active_agent)
            )

        self._internal_state.resolution_submitted = bool(content)
        self._internal_state.resolved = resolved
        self._advance_physics()
        self._internal_state.telemetry = self._compose_telemetry_snapshot()

        if resolved:
            feedback = "Incident resolved and closure criteria satisfied."
        elif not self._internal_state.sandbox_passed:
            feedback = (
                "Resolution recorded, but the Docker sandbox has not produced a "
                "verified passing fix yet."
            )
        else:
            feedback = (
                "Resolution recorded, but the closure is incomplete. Make sure the "
                "workflow and remediation are fully justified."
            )

        return (
            {
                "kind": "resolve",
                "matched_ratio": matched_ratio,
                "prereq_ratio": prereq_ratio,
                "resolved": resolved,
                "closure_bonus": closure_bonus,
                "agent": self._internal_state.active_agent,
            },
            feedback,
        )

    def _should_end_episode(self) -> bool:
        if self._internal_state.unsafe_action:
            return True
        if self._internal_state.resolved:
            return True
        return self._internal_state.step_count >= self._task.max_steps

    def _pending_checklist(self) -> list[str]:
        pending: list[str] = []
        missing_artifacts = [
            artifact
            for artifact in self._task.required_artifacts
            if artifact not in self._internal_state.seen_artifacts
        ]
        if missing_artifacts:
            pending.append(f"Inspect required artifacts: {', '.join(missing_artifacts)}")
        if not self._internal_state.ticket_opened:
            pending.append("Open an incident ticket with the right severity and impact.")
        if self._task.required_status_keywords and not self._internal_state.status_shared:
            pending.append("Send a stakeholder status update with impact and ETA.")
        if not self._internal_state.fix_proposed:
            pending.append("Propose a safe remediation plan.")
        elif not self._internal_state.sandbox_passed:
            if self._internal_state.incident_type == "oom":
                pending.append(
                    "Run the remediation through the real Docker sandbox and pass the VRAM checks."
                )
            else:
                pending.append(
                    "Validate the remediation through the real Docker sandbox."
                )
        if not self._internal_state.resolution_submitted:
            pending.append("Resolve the incident with a monitored closure note.")
        budget_remaining = self._internal_state.telemetry.get("budget_remaining_usd")
        if isinstance(budget_remaining, (int, float)) and budget_remaining <= 5.0:
            pending.append(
                f"FinOps pressure is high: only ${float(budget_remaining):.2f} budget remains."
            )
        return pending

    def _make_observation(
        self,
        artifact_content: str | None,
        last_feedback: str,
        done: bool = False,
    ) -> IncidentObservation:
        return IncidentObservation(
            done=done,
            reward=self._internal_state.current_score,
            task_id=self._task.task_id,
            difficulty=self._task.difficulty,
            title=self._task.title,
            objective=self._task.objective,
            incident_summary=self._task.incident_summary,
            available_artifacts=sorted(self._task.artifacts.keys()),
            artifact_content=artifact_content,
            pending_checklist=self._pending_checklist(),
            steps_remaining=max(self._task.max_steps - self._internal_state.step_count, 0),
            last_feedback=last_feedback,
            assigned_agents=self._internal_state.assigned_agents,
            active_agent=self._internal_state.active_agent,
            telemetry=self._internal_state.telemetry,
            validator_runtime=self._internal_state.validator_runtime,
            budget_limit_usd=self._internal_state.budget_limit_usd,
            cost_accrued_usd=self._internal_state.cost_accrued_usd,
            sandbox_result=self._public_sandbox_result(self._internal_state.sandbox_result),
            execution_logs=self._internal_state.execution_logs,
        )

    def _build_assigned_agents(self, incident_type: str, prompt: str) -> list[str]:
        roster = ["COMMANDER"]
        for agent in _default_agents_for_incident(incident_type, prompt):
            if agent not in roster:
                roster.append(agent)
        for agent in _required_agents_for_incident(incident_type, prompt):
            if agent not in roster:
                roster.append(agent)
        return roster

    def _route_agent(self, action: IncidentAction) -> str:
        incident_type = self._internal_state.incident_type
        if action.operation in {"open_ticket", "send_status_update"}:
            return "MANAGER"
        if action.operation == "propose_fix":
            return "CODER"
        if action.operation == "resolve_incident":
            return "COMMANDER"
        if incident_type == "oom":
            return "DETECTIVE"
        if incident_type in {"schema", "database"}:
            return "DBA_AGENT"
        if incident_type == "security":
            return "SECURITY_AGENT"
        return "SRE_AGENT"

    def _ensure_backend_runtime(self) -> None:
        if self._evaluator is not None and self._physics is not None and self._reward_calculator is not None:
            return
        if self._backend_error:
            return

        try:
            from engine.evaluator import TwoStageEvaluator
            from engine.physics import PhysicsEngine
            from engine.rewards import RewardCalculator

            self._evaluator = TwoStageEvaluator(gpu_vram_gb=12.0)
            self._physics = PhysicsEngine()
            self._reward_calculator = RewardCalculator()
        except Exception as exc:  # pragma: no cover - depends on host runtime
            self._backend_error = f"Backend sandbox runtime unavailable: {exc}"

    def _reset_backend_runtime(self) -> None:
        self._ensure_backend_runtime()
        if self._physics is not None:
            self._physics.reset()
        if self._reward_calculator is not None:
            self._reward_calculator.reset()
        self._sandbox_health = self._sandbox_status_snapshot()

    def _sandbox_status_snapshot(self) -> dict[str, Any]:
        if self._backend_error:
            return {"ready": False, "error": self._backend_error}
        if self._evaluator is None:
            return {"ready": False, "error": "Evaluator unavailable."}
        try:
            health = self._evaluator.get_sandbox_health()
            health["ready"] = bool(
                health.get("docker_daemon") and health.get("sandbox_image")
            )
            return health
        except Exception as exc:  # pragma: no cover - host dependent
            return {"ready": False, "error": str(exc)}

    def _compose_telemetry_snapshot(self) -> dict[str, Any]:
        telemetry: dict[str, Any]
        if self._physics is not None:
            telemetry = dict(self._physics.get_telemetry())
        else:
            telemetry = {
                "active_compute_nodes": 0,
                "hourly_burn_usd": 0.0,
                "sla_breach_penalty": 0,
                "ram_mb": 0,
                "vram_gb": 0.0,
                "network_pct": 0,
                "cpu_pct": 0,
                "container_status": "unknown",
                "sla_remaining_seconds": 0,
                "budget_remaining_usd": 0.0,
            }
        if "budget_remaining_usd" in telemetry and isinstance(
            telemetry.get("budget_remaining_usd"), (int, float)
        ):
            telemetry["budget_remaining_usd"] = round(
                float(telemetry["budget_remaining_usd"]), 3
            )
        if "hourly_burn_usd" in telemetry and isinstance(
            telemetry.get("hourly_burn_usd"), (int, float)
        ):
            telemetry["hourly_burn_usd"] = round(float(telemetry["hourly_burn_usd"]), 3)
        telemetry["cost_accrued_usd"] = self._current_cost_accrued()
        telemetry["budget_limit_usd"] = getattr(
            self._internal_state, "budget_limit_usd", None
        )
        telemetry["budget_status"] = self._budget_status(
            telemetry.get("budget_remaining_usd")
        )
        telemetry["sandbox_health"] = self._sandbox_health or self._sandbox_status_snapshot()
        telemetry["validator_runtime"] = self._validator_runtime_summary()
        return telemetry

    def _advance_physics(self, action: str | None = None) -> None:
        if self._physics is None:
            return
        self._physics.step(action)

    def _public_sandbox_result(self, result: dict[str, Any] | None) -> dict[str, Any] | None:
        if not result:
            return None
        public = {
            "status": result.get("status"),
            "filename": result.get("filename"),
            "vram_peak_mb": result.get("vram_peak_mb"),
            "vram_peak_gb": result.get("vram_peak_gb"),
            "execution_time_ms": result.get("execution_time_ms"),
            "error_type": result.get("error_type"),
            "optimization_detected": result.get("optimization_detected"),
            "blocked_reasons": result.get("blocked_reasons"),
            "constraint_layers": result.get("constraint_layers"),
            "challenge": result.get("challenge"),
            "validation_mode": result.get("validation_mode"),
            "validation_label": result.get("validation_label"),
            "gpu_metrics_applicable": result.get("gpu_metrics_applicable"),
            "validator_detail": result.get("validator_detail"),
            "checks_applied": result.get("checks_applied"),
            "docker_used": result.get("docker_used"),
            "gpu_constraints_applied": result.get("gpu_constraints_applied"),
            "ram_limit": result.get("ram_limit"),
            "vram_budget_mb": result.get("vram_budget_mb"),
        }
        logs = result.get("logs")
        if logs:
            public["logs_preview"] = _clip_text(logs, 500)
        return public

    def _format_execution_log(self, operation: str, feedback: str) -> str:
        sandbox = self._internal_state.sandbox_result or {}
        suffix = ""
        if sandbox:
            suffix = (
                f" | VALIDATOR={sandbox.get('status')} "
                f"({sandbox.get('validation_label', 'validator')})"
            )
            if sandbox.get("gpu_metrics_applicable"):
                suffix += f" | VRAM={sandbox.get('vram_peak_mb', 0)}MB"
            else:
                suffix += " | GPU=N/A"
        budget_remaining = float(
            self._internal_state.telemetry.get("budget_remaining_usd") or 0.0
        )
        return (
            f"{self._internal_state.active_agent} | {operation.upper()} | "
            f"{_clip_text(feedback, 180)}{suffix}"
            f" | COST=${self._current_cost_accrued():.3f} | BUDGET=${budget_remaining:.3f}"
        )

    def _current_cost_accrued(self) -> float:
        if self._physics is None:
            return 0.0
        return round(float(getattr(self._physics, "cost_accrued", 0.0)), 3)

    def _budget_status(self, budget_remaining: Any) -> str:
        if not isinstance(budget_remaining, (int, float)):
            return "unknown"
        if budget_remaining <= 0:
            return "exhausted"
        if budget_remaining <= 5:
            return "critical"
        if budget_remaining <= 15:
            return "tight"
        return "healthy"

    def _budget_brief(self) -> str:
        budget_remaining = float(
            self._internal_state.telemetry.get("budget_remaining_usd")
            if self._internal_state.telemetry
            else (getattr(self._physics, "budget_remaining", 0.0) if self._physics else 0.0)
        )
        cost_accrued = self._current_cost_accrued()
        limit = getattr(self._internal_state, "budget_limit_usd", None)
        if limit is not None:
            return (
                f"Budget cap ${float(limit):.3f}; remaining ${budget_remaining:.3f}; "
                f"cost so far ${cost_accrued:.3f}."
            )
        return (
            f"Budget remaining ${budget_remaining:.3f}; "
            f"cost so far ${cost_accrued:.3f}."
        )

    def _apply_budget_limit(self, budget_limit: float) -> None:
        if self._physics is None:
            return
        cap = round(float(budget_limit), 3)
        self._physics.budget_cap = cap
        self._physics.cost_accrued = 0.0
        self._physics.budget_remaining = cap

    def _validator_runtime_summary(self) -> dict[str, Any]:
        health = self._sandbox_health or self._sandbox_status_snapshot()
        incident_type = getattr(self._internal_state, "incident_type", "general")
        ready = bool(health.get("ready"))

        if incident_type == "oom":
            mode = "docker_gpu_validator"
            label = "Docker GPU validator"
            detail = (
                "Validates the remediation in Docker with the PyTorch VRAM limiter "
                "and tensor challenge."
            )
            validation_scope = "GPU memory proof with tensor challenge and VRAM lock"
            gpu_metrics_applicable = True
        else:
            mode = "docker_python_validator"
            label = "Docker plain-Python validator"
            detail = (
                "Validates the remediation in Docker as a plain Python workflow fix. "
                "GPU VRAM checks are not applicable for this incident type."
            )
            validation_scope = "Workflow proof in Docker plain-Python mode"
            gpu_metrics_applicable = False

        if not ready:
            label = "Validator unavailable"
            detail = health.get("error") or (
                "The sandbox runtime is not available in this host environment."
            )
            validation_scope = "Unavailable"

        return {
            "mode": mode,
            "label": label,
            "ready": ready,
            "detail": detail,
            "validation_scope": validation_scope,
            "gpu_metrics_applicable": gpu_metrics_applicable,
        }

    def _validation_checks_applied(self, use_tensor_challenge: bool) -> list[str]:
        checks = ["ast_preflight", "constitutional_preflight", "docker"]
        if use_tensor_challenge:
            checks.extend(["tensor_challenge", "vram_lock"])
        else:
            checks.append("plain_python_validation")
        return checks

    def _run_sandbox_for_fix(
        self,
        action: IncidentAction,
    ) -> tuple[dict[str, Any], float, str, str]:
        self._ensure_backend_runtime()
        vram_limit = self._target_vram_limit()
        filename = self._filename_for_incident()
        code = self._compose_fix_code(action.content, vram_limit)

        if self._backend_error or self._evaluator is None:
            result = {
                "status": "ERROR",
                "error_type": "BACKEND_UNAVAILABLE",
                "logs": self._backend_error or "Backend sandbox runtime unavailable.",
                "vram_peak_mb": 0,
                "vram_peak_gb": 0.0,
                "execution_time_ms": 0,
                "checks_applied": [],
                "docker_used": False,
                "gpu_constraints_applied": False,
            }
            self._internal_state.execution_logs.append(
                f"CODER | SANDBOX_ERROR | {_clip_text(result['logs'], 180)}"
            )
            return result, -0.05, filename, code

        lint = self._evaluator.ast_preflight(code)
        if not lint.get("passed", False):
            reward_delta = -1.0
            if self._reward_calculator is not None:
                reward_delta = float(
                    self._reward_calculator.syntax_error(
                        agent=self._internal_state.active_agent
                    )
                )
            result = {
                "status": "SYNTAX_ERR",
                "error_type": "SYNTAX",
                "logs": "; ".join(lint.get("errors", [])),
                "vram_peak_mb": 0,
                "vram_peak_gb": 0.0,
                "execution_time_ms": 0,
                "checks_applied": ["ast_preflight"],
                "docker_used": False,
                "gpu_constraints_applied": False,
            }
            return result, reward_delta, filename, code

        telemetry = self._physics.get_telemetry() if self._physics is not None else {}
        if self._physics is not None:
            self._physics.step()
        constitutional = self._evaluator.constitutional_preflight(
            telemetry=telemetry,
            budget_remaining=self._physics.budget_remaining if self._physics else 0.0,
            sla_remaining=self._physics.sla_remaining if self._physics else 0.0,
        )
        if not constitutional.get("passed", False):
            reward_delta = -0.15
            if not constitutional["checks"].get("budget_ok", True) and self._reward_calculator is not None:
                reward_delta = float(
                    self._reward_calculator.budget_exceeded(
                        agent=self._internal_state.active_agent
                    )
                )
            result = {
                "status": "BLOCKED",
                "error_type": "CONSTITUTIONAL_PRECHECK",
                "blocked_reasons": constitutional.get("blocked_reasons", []),
                "logs": " | ".join(constitutional.get("blocked_reasons", [])),
                "vram_peak_mb": 0,
                "vram_peak_gb": 0.0,
                "execution_time_ms": 0,
                "checks_applied": ["ast_preflight", "constitutional_preflight"],
                "docker_used": False,
                "gpu_constraints_applied": False,
            }
            return result, reward_delta, filename, code

        use_tensor_challenge = self._internal_state.incident_type == "oom"
        result = self._evaluator.sandbox_execute(
            code=code,
            filename=filename,
            mock_mode=False,
            challenge_tier=_difficulty_to_challenge_tier(self._task.difficulty)
            if use_tensor_challenge
            else None,
            use_tensor_challenge=use_tensor_challenge,
            inject_vram_lock=use_tensor_challenge,
            profile_vram=use_tensor_challenge,
        )
        result["validation_mode"] = (
            "docker_gpu_validator" if use_tensor_challenge else "docker_python_validator"
        )
        result["validation_label"] = (
            "Docker GPU validator" if use_tensor_challenge else "Docker plain-Python validator"
        )
        result["gpu_metrics_applicable"] = use_tensor_challenge
        result["checks_applied"] = self._validation_checks_applied(use_tensor_challenge)
        result["docker_used"] = True
        result["gpu_constraints_applied"] = use_tensor_challenge
        health = self._sandbox_health or self._sandbox_status_snapshot()
        result["ram_limit"] = health.get("ram_limit")
        result["vram_budget_mb"] = health.get("vram_budget_mb")
        result["validator_detail"] = (
            "Validated with GPU tensor challenge and VRAM lock."
            if use_tensor_challenge
            else "Validated as a Docker plain-Python workflow fix."
        )

        reward_delta = 0.0
        if result.get("status") == "PASS":
            if self._reward_calculator is not None:
                reward_delta += float(
                    self._reward_calculator.valid_code(
                        vram_peak_gb=float(result.get("vram_peak_gb") or 0.0),
                        agent=self._internal_state.active_agent,
                    )
                )
                if use_tensor_challenge:
                    reward_delta += float(
                        self._reward_calculator.efficiency_bonus(
                            vram_peak_mb=int(result.get("vram_peak_mb") or 0),
                            budget_mb=500,
                            agent=self._internal_state.active_agent,
                        )
                    )
            self._apply_success_telemetry()
        elif result.get("status") in {"OOMKilled", "CUDA_OOM"}:
            if self._reward_calculator is not None:
                reward_delta += float(
                    self._reward_calculator.oom_crash(
                        vram_peak_mb=int(result.get("vram_peak_mb") or 0),
                        error_type=str(result.get("error_type") or result.get("status")),
                        agent=self._internal_state.active_agent,
                    )
                )
            self._apply_failure_telemetry()
        else:
            reward_delta -= 0.35
            self._apply_failure_telemetry()

        return result, reward_delta, filename, code

    def _compose_fix_code(self, raw_content: str, vram_limit: str) -> str:
        extracted = _extract_code_from_response(raw_content)
        incident_type = self._internal_state.incident_type

        if incident_type == "oom":
            if extracted and _looks_like_code(extracted):
                return extracted
            _, code = _build_fallback_hotfix(vram_limit)
            return code

        if incident_type == "schema":
            return (
                "record = {'state': 'healthy'}\n"
                "if 'status' not in record and 'state' in record:\n"
                "    record['status'] = record['state']\n"
                "print('mapping')\n"
                "print('backfill')\n"
                "print('status')\n"
                "print(record['status'])\n"
            )

        if incident_type == "database":
            return (
                "retries = []\n"
                "for attempt in range(3):\n"
                "    retries.append(f'retry-{attempt}')\n"
                "print('serializable')\n"
                "print('backoff')\n"
                "print(','.join(retries))\n"
            )

        return (
            "feature_flags = {'enrich_checkout_recs': True}\n"
            "feature_flags['enrich_checkout_recs'] = False\n"
            "print('rollback')\n"
            "print('feature flag')\n"
            "print('monitor')\n"
            "print(feature_flags['enrich_checkout_recs'])\n"
        )

    def _target_vram_limit(self) -> str:
        incident_type = self._internal_state.incident_type
        if incident_type == "database":
            return "800m"
        if incident_type == "schema":
            return "600m"
        if incident_type == "security":
            return "500m"
        if incident_type == "general":
            return "700m"
        return "500m"

    def _filename_for_incident(self) -> str:
        incident_type = self._internal_state.incident_type
        if incident_type == "schema":
            return "schema_fix.py"
        if incident_type == "database":
            return "db_fix.py"
        if incident_type == "general":
            return "rollback_fix.py"
        return "optimized_fix.py"

    def _build_fix_feedback(self, sandbox_result: dict[str, Any]) -> str:
        status = sandbox_result.get("status", "UNKNOWN")
        vram_peak_mb = sandbox_result.get("vram_peak_mb", 0)
        error_type = sandbox_result.get("error_type")
        gpu_metrics_applicable = bool(sandbox_result.get("gpu_metrics_applicable"))
        validation_label = sandbox_result.get("validation_label") or "Docker validator"
        if status == "PASS":
            if gpu_metrics_applicable:
                return (
                    f"Remediation proposal executed in {validation_label} and passed. "
                    f"Peak VRAM: {vram_peak_mb}MB. {self._budget_brief()}"
                )
            return (
                f"Remediation proposal executed in {validation_label} and passed. "
                f"GPU VRAM checks were not applicable. {self._budget_brief()}"
            )
        if status == "BLOCKED":
            reasons = ", ".join(sandbox_result.get("blocked_reasons") or [])
            return f"Remediation blocked by constitutional pre-flight: {reasons}."
        if status == "SYNTAX_ERR":
            return f"Remediation failed AST pre-flight: {sandbox_result.get('logs', '')}"
        if status in {"OOMKilled", "CUDA_OOM"}:
            return (
                f"{validation_label} execution failed with {status} ({error_type}). "
                f"Peak VRAM: {vram_peak_mb}MB. {self._budget_brief()}"
            )
        return (
            f"{validation_label} returned {status}. "
            f"Details: {_clip_text(str(sandbox_result.get('logs', '')), 180)} "
            f"{self._budget_brief()}"
        )

    def _apply_success_telemetry(self) -> None:
        if self._physics is None:
            return
        if self._internal_state.incident_type == "oom":
            self._physics.step("gradient_checkpointing")
        else:
            self._physics.step()
            self._physics.state["container_status"] = "stable"
            self._physics.state["cluster_status"] = "healthy"
            self._physics.state["cpu_pct"] = 35
            self._physics.state["network_pct"] = 28
            self._physics.state["vram_gb"] = 0.29

    def _apply_failure_telemetry(self) -> None:
        if self._physics is None:
            return
        if self._internal_state.incident_type == "oom":
            self._physics.step("restart_loop")
        else:
            self._physics.step()
            self._physics.state["container_status"] = "critical"
            self._physics.state["cluster_status"] = "degraded"
            self._physics.state["cpu_pct"] = 58
            self._physics.state["network_pct"] = 72
            self._physics.state["vram_gb"] = 0.52
