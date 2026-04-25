from __future__ import annotations

from typing import Any

import requests


class FrontierLabsEnvClient:
    """Simple HTTP client mirroring the first-round package structure."""

    def __init__(self, base_url: str = "http://127.0.0.1:7860") -> None:
        self.base_url = base_url.rstrip("/")

    def health(self) -> dict[str, Any]:
        response = requests.get(f"{self.base_url}/health", timeout=30)
        response.raise_for_status()
        return response.json()

    def metadata(self) -> dict[str, Any]:
        response = requests.get(f"{self.base_url}/metadata", timeout=30)
        response.raise_for_status()
        return response.json()

    def reset(self, task_id: str) -> dict[str, Any]:
        response = requests.post(
            f"{self.base_url}/reset",
            json={"task_id": task_id},
            timeout=30,
        )
        response.raise_for_status()
        return response.json()

    def step(self, action: dict[str, Any], timeout_s: float | None = None) -> dict[str, Any]:
        payload: dict[str, Any] = {"action": action}
        if timeout_s is not None:
            payload["timeout_s"] = timeout_s
        response = requests.post(f"{self.base_url}/step", json=payload, timeout=30)
        response.raise_for_status()
        return response.json()

    def state(self) -> dict[str, Any]:
        response = requests.get(f"{self.base_url}/state", timeout=30)
        response.raise_for_status()
        return response.json()

    def run(self, prompt: str, task_id: str | None = None) -> dict[str, Any]:
        payload: dict[str, Any] = {"prompt": prompt}
        if task_id:
            payload["task_id"] = task_id
        response = requests.post(f"{self.base_url}/run", json=payload, timeout=60)
        response.raise_for_status()
        return response.json()
