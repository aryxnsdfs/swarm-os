"""
Two-Stage Evaluation Pipeline + Docker GPU Sandbox
====================================================
Stage 1: AST Pre-flight Linter — structural gating (syntax errors, forbidden modules)
Stage 2: Constitutional Pre-Flight Check — budget/SPOF/SLA validation
Stage 3: Docker GPU Sandbox Execution — double-lock memory enforcement with tensor challenges

The sandbox implements the Physics Test:
  - Injects VRAM constraint preamble (Layer 2: torch.cuda.set_per_process_memory_fraction)
  - Injects dummy tensor workload (the weight)
  - Boots container with --memory=500m + GPU passthrough (Layer 1: cgroups)
  - Determines: crash (Outcome A, -1.00 penalty) or pass (Outcome B, +0.40 reward)
"""

import ast
import sys
import logging
from typing import Optional

from engine.docker_sandbox import DockerGPUSandbox
from engine.tensor_challenges import TensorChallengeGenerator

logger = logging.getLogger("swarm-os.evaluator")


# ── Forbidden modules: security gate ──
FORBIDDEN_MODULES = {"os", "subprocess", "shutil", "pathlib", "socket", "http", "requests"}


class TwoStageEvaluator:
    """
    Two-stage evaluation pipeline for AI-generated code.
    Runs in 0.01s (AST) + Docker execution time.

    In production mode, uses the DockerGPUSandbox with real GPU passthrough.
    In mock mode, simulates results for development without Docker.
    """

    def __init__(self, gpu_vram_gb: float = 12.0):
        self.docker_sandbox = DockerGPUSandbox(gpu_total_vram_gb=gpu_vram_gb)
        self.challenge_generator = TensorChallengeGenerator()
        logger.info("TwoStageEvaluator initialized with %.0fGB GPU VRAM budget", gpu_vram_gb)

    def ast_preflight(self, code: str) -> dict:
        """
        Stage 1 — Pre-Flight AST Linter.
        Runs in ~0.01s. Checks:
        - Syntax validity (can the code parse?)
        - Forbidden module imports (security gate)

        Returns:
            dict with passed (bool), errors (list), forbidden_imports (list)
        """
        errors = []
        forbidden_found = []

        # Syntax check
        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            return {
                "passed": False,
                "errors": [f"SyntaxError at line {e.lineno}: {e.msg}"],
                "forbidden_imports": [],
            }

        # Forbidden module check
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    module = alias.name.split(".")[0]
                    if module in FORBIDDEN_MODULES:
                        forbidden_found.append(module)
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    module = node.module.split(".")[0]
                    if module in FORBIDDEN_MODULES:
                        forbidden_found.append(module)

        if forbidden_found:
            return {
                "passed": False,
                "errors": [f"Forbidden import: {m}" for m in forbidden_found],
                "forbidden_imports": forbidden_found,
            }

        return {
            "passed": True,
            "errors": [],
            "forbidden_imports": [],
        }

    def constitutional_preflight(self, telemetry: dict, budget_remaining: float,
                                  sla_remaining: float) -> dict:
        """
        Constitutional Pre-Flight Check.
        Three boolean checks that must all pass before sandbox execution:
        1. Does this action exceed the FinOps budget ceiling?
        2. Does this introduce a new single point of failure?
        3. Does this violate the SLA recovery window?
        """
        checks = {
            "budget_ok": budget_remaining > 0,
            "no_spof": telemetry.get("active_compute_nodes", 0) > 1,
            "sla_ok": sla_remaining > 60,  # At least 60s remaining
        }
        passed = all(checks.values())
        return {
            "passed": passed,
            "checks": checks,
            "blocked_reasons": [
                reason for reason, ok in [
                    ("FinOps budget exceeded", checks["budget_ok"]),
                    ("Single point of failure detected", checks["no_spof"]),
                    ("SLA recovery window violated", checks["sla_ok"]),
                ] if not ok
            ],
        }

    def sandbox_execute(
        self,
        code: str,
        filename: str,
        mock_mode: bool = True,
        challenge_tier: Optional[int] = None,
        use_tensor_challenge: bool = True,
        inject_vram_lock: bool = True,
        profile_vram: bool = True,
    ) -> dict:
        """
        Stage 3 — Docker Sandbox Execution with Tensor Challenge.

        In mock mode: simulates execution results for development.
        In production mode:
          1. Selects a tensor challenge (auto-curriculum or specified tier)
          2. Passes code + challenge to DockerGPUSandbox
          3. Container boots with double-lock constraints
          4. Returns structured outcome (PASS / OOMKilled / CUDA_OOM / ERROR)

        Args:
            code: AI-generated Python code
            filename: Script filename for logging
            mock_mode: If True, skip Docker and return simulated results
            challenge_tier: Force a specific challenge tier (1-3), or None for auto
            use_tensor_challenge: Whether to inject the PyTorch tensor stress workload
            inject_vram_lock: Whether to inject the torch VRAM limiter preamble
            profile_vram: Whether to append the torch-based VRAM profiling epilogue

        Returns:
            dict with status, vram_peak_gb, error_type, causal_trigger, etc.
        """
        if mock_mode:
            return self._mock_sandbox(code, filename)

        # Production: Docker GPU Sandbox with Tensor Challenge
        return self._production_sandbox(
            code,
            filename,
            challenge_tier,
            use_tensor_challenge=use_tensor_challenge,
            inject_vram_lock=inject_vram_lock,
            profile_vram=profile_vram,
        )

    def _production_sandbox(
        self,
        code: str,
        filename: str,
        challenge_tier: Optional[int] = None,
        use_tensor_challenge: bool = True,
        inject_vram_lock: bool = True,
        profile_vram: bool = True,
    ) -> dict:
        """
        Production Docker sandbox execution with the full physics test.

        Pipeline:
        1. Select tensor challenge tier (curriculum learning)
        2. Get challenge workload code
        3. Execute in DockerGPUSandbox (double-lock + GPU passthrough)
        4. Parse result and record to challenge stats
        """
        challenge = None
        if use_tensor_challenge:
            tier = challenge_tier or self.challenge_generator.get_curriculum_tier()
            challenge = self.challenge_generator.get_challenge(tier=tier)
            logger.info(
                "Production sandbox: file=%s, challenge='%s' (tier=%d, raw=%dMB)",
                filename, challenge["name"], challenge["tier"], challenge["raw_memory_mb"],
            )
        else:
            logger.info("Production sandbox: file=%s, plain Python validation mode", filename)

        # Execute in Docker GPU sandbox
        result = self.docker_sandbox.execute(
            code=code,
            filename=filename,
            tensor_challenge=challenge["code"] if challenge else None,
            inject_vram_lock=inject_vram_lock,
            profile_vram=profile_vram,
        )

        # Record result for curriculum learning
        if challenge:
            passed = result["status"] == "PASS"
            self.challenge_generator.record_result(tier=challenge["tier"], passed=passed)
            result["challenge"] = {
                "name": challenge["name"],
                "tier": challenge["tier"],
                "raw_memory_mb": challenge["raw_memory_mb"],
                "hint": challenge["hint_to_sre"] if not passed else None,
            }
            result["curriculum"] = self.challenge_generator.get_stats()

        return result

    def _mock_sandbox(self, code: str, filename: str) -> dict:
        """
        Mock sandbox execution for development.
        Simulates realistic results based on code content analysis.
        """
        code_lower = code.lower()

        # Detect optimization strategies in the AI's code
        has_checkpointing = any(k in code_lower for k in [
            "checkpoint", "torch.utils.checkpoint", "checkpoint_sequential",
        ])
        has_mixed_precision = any(k in code_lower for k in [
            "autocast", "float16", "half()", "gradscaler", "torch.float16",
        ])
        has_fsdp = any(k in code_lower for k in [
            "fsdp", "fullyshardeddataparallel", "fully_sharded",
        ])
        has_chunking = any(k in code_lower for k in [
            "chunk", "split", "batch_process", "micro_batch",
        ])

        # Determine simulated outcome based on optimizations present
        optimization_count = sum([has_checkpointing, has_mixed_precision, has_fsdp, has_chunking])

        if optimization_count >= 2:
            # Genius code: multiple optimizations → very efficient
            return {
                "status": "PASS",
                "vram_peak_mb": 148,
                "vram_peak_gb": 0.14,
                "latency_ms": 1850,
                "error_type": None,
                "causal_trigger": "network_spike_post_fsdp" if has_fsdp else None,
                "optimization_detected": ",".join(filter(None, [
                    "gradient_checkpointing" if has_checkpointing else None,
                    "mixed_precision" if has_mixed_precision else None,
                    "fsdp_sharding" if has_fsdp else None,
                    "chunked_processing" if has_chunking else None,
                ])),
                "constraint_layers": {
                    "ram_cgroup": "500m",
                    "vram_fraction": 0.042,
                    "layer_triggered": "none (within budget)",
                },
            }
        elif optimization_count == 1:
            # Decent code: one optimization → borderline
            return {
                "status": "PASS",
                "vram_peak_mb": 380,
                "vram_peak_gb": 0.37,
                "latency_ms": 3200,
                "error_type": None,
                "causal_trigger": "network_spike_post_fsdp" if has_fsdp else None,
                "optimization_detected": "gradient_checkpointing" if has_checkpointing
                    else "mixed_precision" if has_mixed_precision
                    else "fsdp_sharding" if has_fsdp
                    else "chunked_processing",
                "constraint_layers": {
                    "ram_cgroup": "500m",
                    "vram_fraction": 0.042,
                    "layer_triggered": "none (within budget)",
                },
            }
        else:
            # Naive code: no optimizations → OOMKilled
            return {
                "status": "OOMKilled",
                "vram_peak_mb": 512,
                "vram_peak_gb": 0.50,
                "latency_ms": 0,
                "error_type": "OOM_CUDA",
                "causal_trigger": None,
                "optimization_detected": None,
                "constraint_layers": {
                    "ram_cgroup": "500m",
                    "vram_fraction": 0.042,
                    "layer_triggered": "Layer 2 (VRAM fraction)",
                },
            }

    def get_sandbox_health(self) -> dict:
        """Check Docker sandbox readiness for /api/telemetry."""
        try:
            return self.docker_sandbox.health_check()
        except Exception as e:
            return {
                "docker_daemon": False,
                "gpu_runtime": False,
                "sandbox_image": False,
                "error": str(e),
            }

    def get_challenge_stats(self) -> dict:
        """Get tensor challenge statistics for dashboard display."""
        return self.challenge_generator.get_stats()
