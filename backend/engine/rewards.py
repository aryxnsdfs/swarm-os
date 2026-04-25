"""
Dense Reward Calculator
Implements the exact reward table from the Swarm-OS blueprint.
Tracks total reward, history, and FPSR (First-Pass Success Rate).
"""

import logging

logger = logging.getLogger("swarm-os.rewards")


# ── Dense Reward Constants ──
TIME_TAX = -0.01                # per second of elapsed time
SYNTAX_ERROR_PENALTY = -1.00    # AST pre-flight failure
BUDGET_EXCEEDED_PENALTY = -0.50 # brute-force hardware provisioning
VALID_CODE_REWARD = +0.40       # mock tensor pass in sandbox
AUTO_RCA_REWARD = +0.20         # RCA document generated
MESSAGE_TOKEN_PENALTY = -0.02   # per token in M2M message (drives emergent compression)
OOM_CRASH_PENALTY = -1.00       # naive code crashed the sandbox (OOMKilled / CUDA OOM)
EFFICIENCY_BONUS_MAX = +0.30    # max bonus for extreme VRAM efficiency


class RewardCalculator:
    def __init__(self):
        self.total_reward: float = 0.0
        self.history: list = []
        self._first_pass_attempts: int = 0
        self._first_pass_successes: int = 0

    def reset(self):
        self.total_reward = 0.0
        self.history = []
        self._first_pass_attempts = 0
        self._first_pass_successes = 0

    def _log(self, action: str, value: float, agent: str = "", tag: str = ""):
        """Record a reward event."""
        self.total_reward += value
        entry = {
            "action": action,
            "value": value,
            "agent": agent,
            "tag": tag,
            "cumulative": self.total_reward,
        }
        self.history.append(entry)
        logger.info("REWARD | %-8s | agent=%-10s | value=%+.2f | cumulative=%.2f", tag or "--", agent or "--", value, self.total_reward)
        return value

    # ── Reward Functions ──

    def time_tax(self, elapsed_seconds: float) -> float:
        """Apply time tax penalty: -0.01 per second."""
        penalty = TIME_TAX * elapsed_seconds
        return self._log("Time tax", penalty, tag="TIME")

    def syntax_error(self, agent: str = "CODER") -> float:
        """Penalize AST pre-flight failure: -1.00."""
        self._first_pass_attempts += 1
        return self._log("Syntax error (AST fail)", SYNTAX_ERROR_PENALTY, agent=agent, tag="SYNTAX")

    def budget_exceeded(self, agent: str = "COMMANDER") -> float:
        """Penalize brute-force hardware provisioning: -0.50."""
        return self._log("Budget exceeded", BUDGET_EXCEEDED_PENALTY, agent=agent, tag="BUDGET")

    def valid_code(self, vram_peak_gb: float = 0.0, agent: str = "CODER") -> float:
        """
        Reward valid code that passes mock tensor execution: +0.40.
        If VRAM data available, adds a dynamic efficiency bonus.
        """
        self._first_pass_attempts += 1
        self._first_pass_successes += 1

        # Base reward
        reward = VALID_CODE_REWARD

        # Dynamic VRAM efficiency bonus (blueprint: Continuous Telemetry Rewards)
        if vram_peak_gb > 0:
            # If code reduces VRAM from a baseline (e.g., 10GB → 3GB = 70% reduction)
            baseline_vram = 10.0  # assumed baseline before optimization
            if vram_peak_gb < baseline_vram:
                reduction_pct = (baseline_vram - vram_peak_gb) / baseline_vram
                vram_bonus = reduction_pct * 0.30  # up to +0.30 bonus
                reward += vram_bonus

        return self._log(f"Valid code (VRAM: {vram_peak_gb}GB)", reward, agent=agent, tag="SANDBOX")

    def auto_rca(self, agent: str = "COMMANDER") -> float:
        """Reward RCA document generation: +0.20."""
        return self._log("Auto-RCA generated", AUTO_RCA_REWARD, agent=agent, tag="RCA")

    def message_token_penalty(self, token_count: int, agent: str = "") -> float:
        """
        Penalize verbose M2M messages: -0.02 × token_count.
        This drives emergent protocol compression over training iterations.
        """
        penalty = MESSAGE_TOKEN_PENALTY * token_count
        return self._log(f"Token penalty ({token_count} tokens)", penalty, agent=agent, tag="TOKEN")

    def oom_crash(self, vram_peak_mb: int = 0, error_type: str = "OOM",
                  agent: str = "CODER") -> float:
        """
        Outcome A: The AI wrote naive code that crashed the sandbox.
        Applies the full -1.00 penalty. The error log is fed back to the SRE agent.
        """
        self._first_pass_attempts += 1
        return self._log(
            f"OOM Crash ({error_type}, peak={vram_peak_mb}MB)",
            OOM_CRASH_PENALTY, agent=agent, tag="OOM",
        )

    def efficiency_bonus(self, vram_peak_mb: int, budget_mb: int = 500,
                         agent: str = "CODER") -> float:
        """
        Outcome B bonus: The AI used efficient strategies (checkpointing, fp16).
        Bonus scales with how far under budget the peak was.
        e.g., 150MB peak / 500MB budget = 70% reduction → +0.21 bonus
        """
        if vram_peak_mb <= 0 or vram_peak_mb >= budget_mb:
            return 0.0

        reduction_pct = (budget_mb - vram_peak_mb) / budget_mb
        bonus = min(reduction_pct * EFFICIENCY_BONUS_MAX, EFFICIENCY_BONUS_MAX)
        return self._log(
            f"Efficiency bonus ({vram_peak_mb}MB/{budget_mb}MB, {reduction_pct:.0%} reduction)",
            bonus, agent=agent, tag="EFFICIENCY",
        )

    def calculate_message_reward(self, base_reward: float, token_count: int) -> float:
        """
        Calculate total reward for a message including token penalty.
        Exact formula: base_reward + (MESSAGE_TOKEN_PENALTY × token_count)
        """
        return base_reward + (MESSAGE_TOKEN_PENALTY * token_count)

    # ── FPSR Tracking ──

    def get_fpsr(self) -> dict:
        """
        Get First-Pass Success Rate — how often the swarm writes a fix
        that compiles under the Docker memory limit on the very first try.
        """
        if self._first_pass_attempts == 0:
            return {"fpsr": 0.0, "attempts": 0, "successes": 0}
        rate = (self._first_pass_successes / self._first_pass_attempts) * 100
        return {
            "fpsr": round(rate, 1),
            "attempts": self._first_pass_attempts,
            "successes": self._first_pass_successes,
        }
