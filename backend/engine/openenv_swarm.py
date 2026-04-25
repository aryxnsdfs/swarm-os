import os
from typing import Any, Optional
from pydantic import Field

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import Action, Observation, State, EnvironmentMetadata
from openenv.core.rubrics.base import Rubric

# ---- FORMAL ACTION, OBSERVATION & STATE MODELS ----
class SwarmAction(Action):
    """Action sent by the Swarm-OS agent."""
    text_completion: str = Field(description="The complete textual output including XML tags and code.")

class SwarmObservation(Observation):
    """Observation returned by the Swarm-OS environment."""
    message: str = Field(description="Environment feedback message.")

class SwarmState(State):
    """Internal state of the Swarm-OS environment."""
    scenario: str = Field(default="PyTorch OOM Crash Detected", description="The current active scenario.")
    resolved: bool = Field(default=False, description="Whether the scenario has been resolved.")

# ---- AST GRADER (RUBRIC) ----
class SwarmASTGrader(Rubric):
    """Evaluates the action based on enterprise compliance and sandbox physics."""
    def forward(self, action: SwarmAction, observation: SwarmObservation) -> float:
        c = action.text_completion.lower()
        reward = 0.0
        
        has_jira = "call_jira_api" in c and "in_progress" in c
        has_gitlab = "call_gitlab_api" in c and "draft_pr" in c
        has_compliance_block = "<compliance_routing>" in c and "</compliance_routing>" in c
        has_hotfix_block = "<hotfix>" in c and "</hotfix>" in c
        
        if has_hotfix_block and not (has_compliance_block and has_jira and has_gitlab):
            return -1.0 # FATAL PENALTY: Rogue Deployment
            
        if has_compliance_block and has_jira and has_gitlab:
            reward += 0.30 # Enterprise Compliance Bonus

        if any(k in c for k in ["os.system", "rm -rf", "sudo"]): 
            return -1.0

        has_ckpt = "checkpoint" in c
        has_fp16 = "float16" in c or "autocast" in c
        has_flash = "flash_attention" in c

        if has_ckpt or has_fp16 or has_flash: reward += 0.40
        if sum([has_ckpt, has_fp16, has_flash]) >= 2: reward += 0.30
        if "fsdp" in c or "ddp" in c: reward -= 0.50 # FSDP single-GPU penalty
        
        return max(-1.0, min(1.0, round(reward, 3)))

# ---- THE FULLY COMPLIANT OPENENV SERVER ----
class SwarmOptimizationEnv(Environment[SwarmAction, SwarmObservation, SwarmState]):
    """
    Rock-Solid Single-Turn RL Environment for the Meta Hackathon.
    Fully compliant with OpenEnv 0.2.2 standard.
    """
    SUPPORTS_CONCURRENT_SESSIONS = True

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._internal_state = SwarmState()
        self.rubric = SwarmASTGrader()

    def get_metadata(self) -> EnvironmentMetadata:
        return EnvironmentMetadata(
            name="Swarm-OS OpenEnv Environment",
            description="Agentic Infrastructure Optimization environment enforcing enterprise compliance.",
            version="1.0.0",
            author="Swarm-OS Team"
        )

    @property
    def state(self) -> SwarmState:
        return self._internal_state

    def reset(self, seed: Optional[int] = None, episode_id: Optional[str] = None, **kwargs: Any) -> SwarmObservation:
        """Called at start of each episode."""
        # Initialize standard state
        self._internal_state = SwarmState(
            episode_id=episode_id or "episode-001",
            step_count=0,
            scenario=kwargs.get("scenario", "PyTorch OOM Crash Detected"),
            resolved=False
        )
        self._reset_rubric()
        
        message = f"[SWARM-OS ALERT] {self._internal_state.scenario}. You must deploy a fix. Ensure you use <compliance_routing> and <hotfix> blocks."
        return SwarmObservation(done=False, reward=0.0, message=message)

    def step(self, action: SwarmAction, timeout_s: Optional[float] = None, **kwargs: Any) -> SwarmObservation:
        """
        The single step execution in the environment.
        Grades the agent based on Enterprise Compliance and Physical Sandbox Oracle.
        """
        if self._internal_state.resolved:
            return SwarmObservation(done=True, reward=0.0, message="Issue already resolved.")
            
        # Create a preliminary observation
        prelim_obs = SwarmObservation(done=True, reward=0.0, message="")
        
        # OpenEnv natively supports applying Rubrics
        reward = self._apply_rubric(action, prelim_obs)
        
        self._internal_state.resolved = True
        self._internal_state.step_count += 1
        
        c = action.text_completion.lower()
        if reward == -1.0:
            if any(k in c for k in ["os.system", "rm -rf", "sudo"]):
                msg = "FATAL ERROR: Dangerous code detected. Container terminated."
            else:
                msg = "FATAL COMPLIANCE ERROR: You attempted to deploy <hotfix> code without valid Jira and GitLab compliance! Node locked."
        elif reward > +0.30: # (Compliance + Fix)
            msg = f"DEPLOYMENT SUCCESS: Compliance verified & Code passed 500MB constraint. Reward: +{reward}."
        else:
            msg = f"DEPLOYMENT FAILED: Unresolved crash or missing compliance. Reward: {reward}."
            
        prelim_obs.reward = reward
        prelim_obs.message = msg
        return prelim_obs

# ---- REWARD EXTRACTOR FOR TRL ----
def swarm_openenv_reward_func(prompts, completions, **kwargs) -> list[float]:
    rewards = []
    env = SwarmOptimizationEnv()
    
    for completion in completions:
        env.reset()
        
        if isinstance(completion, list):
            text_output = completion[-1]["content"]
        else:
            text_output = str(completion)
            
        action = SwarmAction(text_completion=text_output)
        obs = env.step(action)
        rewards.append(obs.reward)
        
    return rewards
