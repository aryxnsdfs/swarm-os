"""
Model Configuration Manager
Loads config.yaml, manages model registry, handles per-agent overrides,
and validates VRAM budget for model switching.
"""

import os
import yaml
import logging
from typing import Optional

logger = logging.getLogger("swarm-os.config")


CONFIG_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config.yaml")


class ModelConfigManager:
    def __init__(self):
        self.active_model: str = ""
        self.models: dict = {}
        self.agent_model_overrides: dict = {}
        self.gpu_vram_gb: float = 12.0  # RTX 3060 default

    def load(self, path: str = CONFIG_PATH):
        """Load model configuration from YAML file."""
        with open(path, "r") as f:
            config = yaml.safe_load(f)

        self.active_model = config.get("active_model", "llama-3.1-8b-instruct-4bit")
        self.models = config.get("models", {})
        self.agent_model_overrides = config.get("agent_model_overrides", {})
        logger.info("Config loaded from %s: %d models, active=%s", path, len(self.models), self.active_model)

    def get_model_for_agent(self, agent_role: str) -> dict:
        """
        Resolve which model an agent should use.
        Checks agent_model_overrides first, then falls back to active_model.
        """
        model_key = self.agent_model_overrides.get(agent_role, self.active_model)
        model = self.models.get(model_key, {})
        return {"model_key": model_key, **model}

    def switch_model(self, model_key: str) -> dict:
        """
        Switch the active model. Validates the model exists and checks VRAM budget.
        Returns success/failure with details.
        """
        if model_key not in self.models:
            return {"success": False, "error": f"Model '{model_key}' not found in registry."}

        model = self.models[model_key]
        if model.get("max_vram_gb", 0) > self.gpu_vram_gb:
            return {
                "success": False,
                "error": f"Model requires {model['max_vram_gb']}GB VRAM but GPU only has {self.gpu_vram_gb}GB.",
                "model": model,
            }

        self.active_model = model_key
        return {
            "success": True,
            "active_model": model_key,
            "model": model,
        }

    def list_models(self) -> list:
        """List all available models with VRAM feasibility flags."""
        result = []
        for key, model in self.models.items():
            result.append({
                "key": key,
                "is_active": key == self.active_model,
                "vram_feasible": model.get("max_vram_gb", 0) <= self.gpu_vram_gb,
                **model,
            })
        return result

    def set_agent_override(self, agent_role: str, model_key: str) -> dict:
        """Set a per-agent model override."""
        if model_key not in self.models:
            return {"success": False, "error": f"Model '{model_key}' not found."}

        self.agent_model_overrides[agent_role] = model_key
        return {"success": True, "agent": agent_role, "model": model_key}
