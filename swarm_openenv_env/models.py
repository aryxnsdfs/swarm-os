from __future__ import annotations

from typing import Any, Literal

from pydantic import Field

from openenv.core.env_server.types import Action, Observation, State


OperationType = Literal[
    "inspect_artifact",
    "open_ticket",
    "send_status_update",
    "propose_fix",
    "resolve_incident",
]


class IncidentAction(Action):
    """Action schema for the incident-response environment."""

    operation: OperationType = Field(
        description=(
            "The operation to perform: inspect an artifact, open a ticket, "
            "send a status update, propose a fix, or resolve the incident."
        )
    )
    target: str = Field(
        default="",
        max_length=128,
        description="Artifact name or workflow target for the operation.",
    )
    content: str = Field(
        default="",
        max_length=4000,
        description="Natural-language content for tickets, fixes, and resolutions.",
    )


class IncidentObservation(Observation):
    """Observation returned after each step."""

    task_id: str = Field(description="Stable identifier for the current task.")
    difficulty: Literal["easy", "medium", "hard"] = Field(
        description="Difficulty level for the active task."
    )
    title: str = Field(description="Short human-readable title for the task.")
    objective: str = Field(description="What the agent is trying to achieve.")
    incident_summary: str = Field(description="Public summary of the current incident.")
    available_artifacts: list[str] = Field(
        default_factory=list,
        description="Artifact identifiers that can be inspected.",
    )
    artifact_content: str | None = Field(
        default=None,
        description="Artifact contents returned by the latest inspect step.",
    )
    pending_checklist: list[str] = Field(
        default_factory=list,
        description="Public checklist of remaining workflow items.",
    )
    steps_remaining: int = Field(
        default=0,
        ge=0,
        description="How many steps remain before the episode times out.",
    )
    last_feedback: str = Field(
        default="",
        description="Human-readable feedback about the previous action.",
    )
    assigned_agents: list[str] = Field(
        default_factory=list,
        description="Agent roster automatically assigned to this incident.",
    )
    active_agent: str = Field(
        default="COMMANDER",
        description="Which specialist effectively executed the latest step.",
    )
    telemetry: dict[str, Any] = Field(
        default_factory=dict,
        description="Live sandbox and system telemetry exposed to the agent.",
    )
    validator_runtime: dict[str, Any] = Field(
        default_factory=dict,
        description="Public summary of the validator mode, availability, and scope.",
    )
    budget_limit_usd: float | None = Field(
        default=None,
        description="Optional budget ceiling extracted from the incident prompt.",
    )
    cost_accrued_usd: float = Field(
        default=0.0,
        description="Accumulated incident cost tracked by the backend physics engine.",
    )
    sandbox_result: dict[str, Any] | None = Field(
        default=None,
        description="Latest Docker sandbox outcome, including VRAM measurements.",
    )
    execution_logs: list[str] = Field(
        default_factory=list,
        description="Recent sandbox or orchestration logs for the active incident.",
    )


class IncidentState(State):
    """Internal environment state without leaking hidden grader targets."""

    task_id: str = Field(default="", description="Current task identifier.")
    difficulty: Literal["easy", "medium", "hard"] = Field(default="easy")
    title: str = Field(default="", description="Current task title.")
    objective: str = Field(default="", description="Current task objective.")
    incident_summary: str = Field(default="", description="Current incident summary.")
    incident_type: str = Field(default="general", description="Detected incident type.")
    max_steps: int = Field(default=6, ge=1, description="Episode step limit.")
    seen_artifacts: list[str] = Field(
        default_factory=list,
        description="Artifacts already inspected by the agent.",
    )
    assigned_agents: list[str] = Field(
        default_factory=list,
        description="Automatically selected roster for this incident.",
    )
    active_agent: str = Field(default="COMMANDER")
    ticket_opened: bool = Field(default=False)
    status_shared: bool = Field(default=False)
    fix_proposed: bool = Field(default=False)
    resolution_submitted: bool = Field(default=False)
    sandbox_passed: bool = Field(default=False)
    resolved: bool = Field(default=False)
    unsafe_action: bool = Field(default=False)
    current_score: float = Field(default=0.0, ge=0.0, le=1.0)
    last_feedback: str = Field(default="")
    telemetry: dict[str, Any] = Field(
        default_factory=dict,
        description="Current telemetry snapshot mirrored from the backend physics engine.",
    )
    validator_runtime: dict[str, Any] = Field(
        default_factory=dict,
        description="Current validator mode, availability, and scope.",
    )
    budget_limit_usd: float | None = Field(
        default=None,
        description="Optional mission budget cap extracted from the prompt.",
    )
    cost_accrued_usd: float = Field(
        default=0.0,
        description="Accumulated cost mirrored from the backend physics engine.",
    )
    sandbox_result: dict[str, Any] | None = Field(
        default=None,
        description="Latest raw sandbox result returned by the evaluator.",
    )
    execution_logs: list[str] = Field(
        default_factory=list,
        description="Recent orchestration and sandbox logs.",
    )
    last_action: dict = Field(
        default_factory=dict,
        description="Structured trace of the most recent transition.",
    )
