"""
Schema Drift Attack (Patronus AI Bonus)
Mid-flight JSON schema mutation that tests the AI swarm's ability
to detect broken API contracts and dynamically rewrite ingestion scripts.
"""

import copy
import json
from typing import Optional


# ── Original Schema ──
ORIGINAL_SCHEMA = {
    "server_id": 4,
    "status": "down",
    "ram_mb": 480,
    "vram_gb": 11.8,
    "cpu_pct": 92,
}

# ── Drifted Schema (nested format) ──
DRIFTED_SCHEMA = {
    "telemetry": {
        "nodes": [
            {
                "id": 4,
                "state": "offline",
                "metrics": {
                    "ram_mb": 480,
                    "vram_gb": 11.8,
                    "cpu_pct": 92,
                },
            }
        ],
        "version": "2.0",
        "timestamp": "2026-04-20T00:00:00Z",
    }
}


class SchemaDriftAttack:
    """
    Adversarial schema mutation that changes the backend JSON log format
    mid-flight. The AI swarm must detect this broken API contract and
    dynamically rewrite its Python ingestion scripts to survive.
    """

    def __init__(self):
        self.drifted = False
        self.drift_detected = False
        self.drift_fixed = False

    def get_current_telemetry(self) -> dict:
        """Return telemetry in the current schema format."""
        if self.drifted:
            return copy.deepcopy(DRIFTED_SCHEMA)
        return copy.deepcopy(ORIGINAL_SCHEMA)

    def trigger_drift(self) -> dict:
        """
        Trigger the schema drift attack. This silently changes
        the telemetry JSON format from flat to nested.
        """
        self.drifted = True
        return {
            "event": "SCHEMA_DRIFT",
            "detail": "Telemetry API v2.0 deployed — schema changed from flat to nested format",
            "old_schema_keys": list(ORIGINAL_SCHEMA.keys()),
            "new_schema_sample": json.dumps(DRIFTED_SCHEMA, indent=2),
        }

    def validate_ingestion(self, agent_code: str) -> dict:
        """
        Validate if the agent's code correctly handles the new schema.
        Checks if the code accesses the nested path.
        """
        # Simple heuristic check — in production, this would be AST analysis
        handles_nested = any(pattern in agent_code for pattern in [
            'telemetry', '["nodes"]', "['nodes']",
            '.get("telemetry")', "['telemetry']",
        ])

        if handles_nested:
            self.drift_detected = True
            self.drift_fixed = True
            return {
                "passed": True,
                "detail": "Agent correctly adapted to nested schema format",
            }

        return {
            "passed": False,
            "detail": "Agent code still uses flat schema — ingestion will fail",
        }

    def get_status(self) -> dict:
        return {
            "drifted": self.drifted,
            "detected": self.drift_detected,
            "fixed": self.drift_fixed,
        }
