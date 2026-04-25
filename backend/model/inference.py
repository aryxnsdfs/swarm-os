"""
Model Inference Abstraction Layer
Provides a unified interface for LLM inference. Reads from config.yaml
for active model selection. Supports mock mode for development and
easy swap between model backends (Unsloth, HuggingFace, API).
"""

import logging
from typing import Optional
import requests
from model.config import ModelConfigManager


logger = logging.getLogger("swarm-os.inference")
LOCAL_LM_STUDIO_URL = "http://localhost:1234/v1"


class InferenceEngine:
    """
    Unified inference interface. In production, this wraps Unsloth 4-bit QLoRA
    inference. In dev mode, returns mock responses for testing.
    """

    def __init__(self, config_manager: ModelConfigManager, mock_mode: bool = True):
        self.config = config_manager
        self.mock_mode = mock_mode
        self._loaded_models: dict = {}  # model_key -> loaded model reference
        self._runtime_logged = False

    def describe_runtime(self) -> dict:
        configured_agents = {}
        for agent_role in ("COMMANDER", "MANAGER", "DETECTIVE", "CODER"):
            configured_agents[agent_role] = self.config.get_model_for_agent(agent_role)

        summary = {
            "mode": "mock" if self.mock_mode else "live",
            "active_model": self.config.active_model,
            "configured_agents": configured_agents,
            "endpoint": LOCAL_LM_STUDIO_URL if not self.mock_mode else None,
            "lm_studio": self._probe_local_runtime() if not self.mock_mode else None,
        }
        return summary

    def log_runtime_summary(self) -> None:
        summary = self.describe_runtime()
        logger.info(
            "Inference runtime: mode=%s active_model=%s endpoint=%s",
            summary["mode"],
            summary["active_model"],
            summary["endpoint"] or "disabled",
        )
        for agent_role, model_info in summary["configured_agents"].items():
            logger.info(
                "Agent model route: %s -> %s (%s)",
                agent_role,
                model_info.get("model_key", "unknown"),
                model_info.get("name", "unnamed"),
            )

        lm_runtime = summary.get("lm_studio")
        if lm_runtime:
            if lm_runtime.get("reachable"):
                logger.info(
                    "LM Studio detected: endpoint=%s loaded_models=%s default_model=%s",
                    lm_runtime.get("endpoint"),
                    lm_runtime.get("models") or [],
                    lm_runtime.get("default_model") or "unknown",
                )
            else:
                logger.warning(
                    "LM Studio probe failed: endpoint=%s error=%s",
                    lm_runtime.get("endpoint"),
                    lm_runtime.get("error"),
                )
        self._runtime_logged = True

    def _probe_local_runtime(self) -> dict:
        try:
            response = requests.get(f"{LOCAL_LM_STUDIO_URL}/models", timeout=(3, 5))
            response.raise_for_status()
            payload = response.json()
            models = payload.get("data") or []
            model_ids = [item.get("id", "unknown") for item in models if isinstance(item, dict)]
            return {
                "reachable": True,
                "endpoint": LOCAL_LM_STUDIO_URL,
                "models": model_ids,
                "default_model": model_ids[0] if model_ids else None,
            }
        except Exception as exc:
            return {
                "reachable": False,
                "endpoint": LOCAL_LM_STUDIO_URL,
                "error": str(exc),
            }

    def generate(
        self,
        prompt: str,
        agent_role: str,
        system_prompt: str = "",
        max_tokens: int = 512,
        temperature: float = 0.7,
    ) -> dict:
        """
        Generate a response from the model assigned to the given agent role.
        Respects per-agent model overrides from config.yaml.

        Returns:
            dict with keys: response, model_used, tokens_generated, think_block
        """
        if not self._runtime_logged:
            self.log_runtime_summary()

        model_info = self.config.get_model_for_agent(agent_role)
        model_key = model_info["model_key"]

        if self.mock_mode:
            return self._mock_generate(prompt, agent_role, model_key)

        # Production path: Unsloth inference
        return self._unsloth_generate(prompt, agent_role, model_key, system_prompt, max_tokens, temperature)

    def _mock_generate(self, prompt: str, agent_role: str, model_key: str) -> dict:
        """Mock inference for development — returns incident-aware placeholder responses."""
        low = prompt.lower()

        if any(keyword in low for keyword in ("soc2", "public bucket", "bucket", "security", "compliance", "audit")):
            incident_type = "security"
        elif any(keyword in low for keyword in ("sql", "deadlock", "database", "schema", "drift")):
            incident_type = "database"
        elif any(keyword in low for keyword in ("oom", "out of memory", "vram", "cuda", "gpu", "pytorch")):
            incident_type = "oom"
        else:
            incident_type = "general"

        mock_responses = {
            "COMMANDER": {
                "security": "Security incident confirmed. Dispatching containment and compliance specialists.\n[\"MANAGER\", \"SECURITY_AGENT\", \"COMPLIANCE_AGENT\"]",
                "database": "Database incident confirmed. Dispatching manager and DBA for diagnosis.\n[\"MANAGER\", \"DBA_AGENT\"]",
                "oom": "OOM incident confirmed. Dispatching manager, detective, and coder.\n[\"MANAGER\", \"DETECTIVE\", \"CODER\"]",
                "general": "Infrastructure incident triaged. Dispatching manager and SRE.\n[\"MANAGER\", \"SRE_AGENT\"]",
            },
            "MANAGER": {
                "security": "Incident intake complete. Coordinating containment and compliance follow-up.",
                "database": "Database fault acknowledged. Handing diagnosis to the DBA specialist.",
                "oom": "Baseline execution failed under the current constraint. Escalating to root-cause analysis.",
                "general": "Incident intake complete. Coordinating the next operator.",
            },
            "SECURITY_AGENT": {
                "security": "Seal the exposed bucket, remove public access, and preserve the audit trail before any broader remediation.",
                "database": "No direct security action required for this database incident.",
                "oom": "No security-specific containment required for this OOM incident.",
                "general": "Containment guidance not required for this incident type.",
            },
            "COMPLIANCE_AGENT": {
                "security": "Compliance review: Jira and GitLab evidence must be checked before declaring the incident closed.",
                "database": "Compliance review: operational changes should still be logged for auditability.",
                "oom": "Compliance review: ensure deployment records exist before rollout.",
                "general": "Compliance review logged.",
            },
            "DBA_AGENT": {
                "security": "No database action required for this security incident.",
                "database": "Deadlock and schema state need targeted DBA review before changes are applied.",
                "oom": "No database action required for this OOM incident.",
                "general": "Database triage not required for this incident type.",
            },
            "DETECTIVE": {
                "security": "No PyTorch detective flow required for this security incident.",
                "database": "No PyTorch detective flow required for this database incident.",
                "oom": "Root cause points to GPU memory pressure. Recommend mixed precision and checkpointing before retry.",
                "general": "General incident diagnosis complete.",
            },
            "CODER": {
                "security": "<compliance_routing>\nJIRA: SWARM-4821 | Severity: P1-Critical | Assignee: coder-agent\nGitLab MR: !1094 (auto-generated)\nCompliance: SOC2 review pending\n</compliance_routing>\n<hotfix>\n# No code patch generated in mock mode for this incident type.\n</hotfix>",
                "database": "BEGIN; SET TRANSACTION ISOLATION LEVEL SERIALIZABLE; -- apply targeted remediation",
                "oom": "<compliance_routing>\nJIRA: SWARM-4821 | Severity: P1-Critical | Assignee: coder-agent\nGitLab MR: !1094 (auto-generated)\nCompliance: SOC2-CC7.1 verified\n</compliance_routing>\n<hotfix>\nimport torch\nfrom torch.cuda.amp import autocast\n\n# placeholder optimization fix\n</hotfix>",
                "general": "No code patch generated in mock mode.",
            },
            "SRE_AGENT": {
                "security": "SLA currently stable; prioritize containment and evidence preservation.",
                "database": "Infrastructure appears healthy; focus remains on the database layer.",
                "oom": "Container pressure is localized to GPU memory. Awaiting optimized retry.",
                "general": "Infrastructure triage complete.",
            },
        }
        response = mock_responses.get(agent_role, {}).get(incident_type, "ACK | PROCESSING")

        return {
            "response": response,
            "model_used": model_key,
            "tokens_generated": len(response.split()),
            "think_block": f"[Mock CoT for {agent_role}]: Analyzing the situation based on the prompt...",
        }

    def _unsloth_generate(
        self, prompt: str, agent_role: str, model_key: str,
        system_prompt: str, max_tokens: int, temperature: float,
    ) -> dict:
        """
        Production inference using the locally hosted GGUF model via LM Studio/Ollama.
        """
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        # Hit the Local GGUF Server (LM Studio default port is 1234)
        # timeout=(connect_s, read_s): fail fast if LM Studio is offline (5s),
        # but give the model up to 180s to finish prefill + generation.
        try:
            response = requests.post(
                f"{LOCAL_LM_STUDIO_URL}/chat/completions",
                json={
                    "messages": messages,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                    "stream": False
                },
                timeout=(5, 180)
            )
            response.raise_for_status()
            data = response.json()
            
            ai_text = data['choices'][0]['message']['content']
            tokens_used = data.get('usage', {}).get('completion_tokens', len(ai_text.split()))
            runtime_model = data.get("model") or model_key
            logger.info(
                "Live generation: agent=%s configured_model=%s runtime_model=%s endpoint=%s tokens=%s",
                agent_role,
                model_key,
                runtime_model,
                LOCAL_LM_STUDIO_URL,
                tokens_used,
            )

            return {
                "response": ai_text,
                "model_used": runtime_model,
                "tokens_generated": tokens_used,
                "think_block": f"[Generated securely on local PC via GGUF]",
            }
        except Exception as e:
            logger.error("Failed to connect to local GGUF server at %s: %s", LOCAL_LM_STUDIO_URL, e)
            # Fall back to deterministic, incident-aware mock output so the app stays usable
            # without silently forcing every failed call into the old OOM storyline.
            fallback = self._mock_generate(prompt, agent_role, model_key)
            fallback["think_block"] = f"[Local GGUF unavailable: {e}] Deterministic fallback response generated."
            return fallback

    def _load_model(self, model_key: str):
        """Load a model into GPU memory. Placeholder for Unsloth integration."""
        # In production: FastLanguageModel.from_pretrained(...)
        self._loaded_models[model_key] = {"loaded": True, "key": model_key}
        print(f"[InferenceEngine] Model '{model_key}' loaded (mock)")

    def unload_model(self, model_key: str):
        """Unload a model from GPU memory to free VRAM."""
        if model_key in self._loaded_models:
            del self._loaded_models[model_key]
            print(f"[InferenceEngine] Model '{model_key}' unloaded")
