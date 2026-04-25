from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Optional

import uvicorn
from fastapi import Body, FastAPI, HTTPException
from pydantic import BaseModel, Field, ValidationError

from openenv.core.env_server.http_server import (
    HealthResponse,
    JsonRpcErrorCode,
    JsonRpcResponse,
    ResetRequest,
    ResetResponse,
    SchemaResponse,
    StepRequest,
    StepResponse,
)

from swarm_openenv_env.environment import IncidentResponseEnv
from swarm_openenv_env.models import (
    IncidentAction,
    IncidentObservation,
    IncidentState,
)
from swarm_openenv_env.tasks import get_task

from inference import (
    ALLOW_SCRIPTED_BASELINE,
    MODEL_NAME,
    TRAINED_GGUF_PATH,
    LOCAL_OPENAI_BASE_URL,
    available_provider_names,
    clean_error_message,
    compact_action_dict,
    create_clients,
    detect_provider,
    effective_prompt_for_task,
    get_recommended_prompts,
    llm_action,
    select_task_id,
)


app = FastAPI(
    title="Swarm Incident Response OpenEnv",
    version="1.0.0",
    description=(
        "HTTP server for a real-world incident-response OpenEnv benchmark with "
        "reset/step/state endpoints."
    ),
)

_env = IncidentResponseEnv()


def _serialize_observation(observation: IncidentObservation) -> dict:
    return observation.model_dump()


class PromptRunRequest(BaseModel):
    prompt: str = Field(
        default="",
        description="High-level mission brief or incident prompt for the live runner.",
    )
    task_id: Optional[str] = Field(
        default=None,
        description="Optional explicit task override.",
    )


class PromptRunStep(BaseModel):
    step: int
    action: dict[str, Any]
    reward: float
    done: bool
    feedback: str


class PromptRunResponse(BaseModel):
    provider: str
    model: str
    task_id: str
    prompt: str
    success: bool
    score: float
    steps: int
    error: Optional[str] = None
    trajectory: list[PromptRunStep]


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse()


@app.get("/metadata")
def metadata() -> dict:
    metadata = _env.get_metadata().model_dump()
    metadata["recommended_prompts"] = get_recommended_prompts()
    metadata["model_runtime"] = {
        "provider": detect_provider() or "none",
        "provider_chain": available_provider_names(),
        "model": MODEL_NAME,
        "local_openai_base_url": LOCAL_OPENAI_BASE_URL,
        "trained_model_path": str(TRAINED_GGUF_PATH),
        "trained_model_present": TRAINED_GGUF_PATH.exists(),
    }
    metadata["validator_runtime"] = _env.get_validator_runtime()
    readme_path = Path(__file__).resolve().parent.parent / "README.md"
    if readme_path.exists():
        metadata["readme_content"] = readme_path.read_text(encoding="utf-8")
    return metadata


@app.get("/schema", response_model=SchemaResponse)
def schema() -> SchemaResponse:
    return SchemaResponse(
        action=IncidentAction.model_json_schema(),
        observation=IncidentObservation.model_json_schema(),
        state=IncidentState.model_json_schema(),
    )


@app.get("/demo-prompt")
def demo_prompt() -> dict:
    return {
        "provider": detect_provider() or "none",
        "provider_chain": available_provider_names(),
        "model": MODEL_NAME,
        "trained_model_path": str(TRAINED_GGUF_PATH),
        "trained_model_present": TRAINED_GGUF_PATH.exists(),
        "local_openai_base_url": LOCAL_OPENAI_BASE_URL,
        "validator_runtime": _env.get_validator_runtime(),
        "recommended_prompts": get_recommended_prompts(),
    }


@app.get("/state", response_model=IncidentState)
def state() -> IncidentState:
    return _env.state


@app.post("/reset", response_model=ResetResponse)
def reset(request: ResetRequest = Body(default_factory=ResetRequest)) -> ResetResponse:
    payload = request.model_dump(exclude_none=True)
    observation = _env.reset(**payload)
    return ResetResponse(
        observation=_serialize_observation(observation),
        reward=float(observation.reward or 0.0),
        done=bool(observation.done),
    )


@app.post("/step", response_model=StepResponse)
def step(request: StepRequest) -> StepResponse:
    try:
        action = IncidentAction.model_validate(request.action)
    except ValidationError as exc:
        raise HTTPException(status_code=422, detail=exc.errors()) from exc

    observation = _env.step(action, timeout_s=request.timeout_s)
    return StepResponse(
        observation=_serialize_observation(observation),
        reward=float(observation.reward or 0.0),
        done=bool(observation.done),
    )


@app.post("/run", response_model=PromptRunResponse)
def run_prompt(request: PromptRunRequest) -> PromptRunResponse:
    clients = create_clients()
    if not clients and not ALLOW_SCRIPTED_BASELINE:
        raise HTTPException(
            status_code=503,
            detail=(
                "No live provider configured. Start your local OpenAI-compatible runtime "
                "for the trained model, or set OPENAI_API_KEY / GEMINI_API_KEY for a real "
                "prompt-driven run."
            ),
        )

    resolved_task_id = request.task_id or select_task_id(clients, request.prompt)
    if not resolved_task_id:
        resolved_task_id = _env.default_task_id

    mission_prompt = effective_prompt_for_task(resolved_task_id, request.prompt)
    task = get_task(resolved_task_id)
    env = IncidentResponseEnv(default_task_id=resolved_task_id)
    observation = env.reset(task_id=resolved_task_id, prompt=mission_prompt)
    history: list[str] = []
    trajectory: list[PromptRunStep] = []
    runtime_error: Optional[str] = None

    for step_index in range(task.max_steps):
        try:
            action = llm_action(
                clients,
                resolved_task_id,
                observation,
                history,
                step_index,
                mission_prompt,
            )
            action_payload = compact_action_dict(action)
            observation = env.step(action)
            reward = float(observation.reward or 0.0)
            step_no = step_index + 1
            trajectory.append(
                PromptRunStep(
                    step=step_no,
                    action=action_payload,
                    reward=reward,
                    done=bool(observation.done),
                    feedback=observation.last_feedback,
                )
            )
            history.append(
                f"step={step_no} action={action_payload} score={reward:.3f} "
                f"feedback={observation.last_feedback}"
            )
            if observation.done:
                break
        except Exception as exc:
            runtime_error = clean_error_message(exc)
            break

    score = round(float(env.state.current_score), 3)
    success = score >= task.success_threshold and bool(env.state.resolved)
    return PromptRunResponse(
        provider=detect_provider() or "none",
        model=MODEL_NAME,
        task_id=resolved_task_id,
        prompt=mission_prompt,
        success=success,
        score=score,
        steps=len(trajectory),
        error=runtime_error,
        trajectory=trajectory,
    )


@app.post("/mcp")
def mcp() -> dict:
    return JsonRpcResponse.error_response(
        JsonRpcErrorCode.METHOD_NOT_FOUND,
        "This environment exposes the standard OpenEnv HTTP API only.",
    ).model_dump()


def main() -> None:
    port = int(os.getenv("PORT", "7860"))
    uvicorn.run("server.app:app", host="0.0.0.0", port=port, reload=False)


if __name__ == "__main__":
    main()
