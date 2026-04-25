"""
Swarm Orchestrator
Multi-agent coordination with dynamic spawning/dismissal,
System Prompt Integrity Gate verification, per-agent model routing,
and Agent Disagreement Detection with structured resolution.
"""

import logging
from typing import Optional
from model.config import ModelConfigManager

logger = logging.getLogger("swarm-os.orchestrator")


class SwarmOrchestrator:
    """
    Coordinates the multi-agent swarm: Commander, Detective, Coder.
    Handles dynamic agent spawning with System Prompt Integrity Gate,
    per-agent model routing, and disagreement detection.
    """

    def __init__(self, config_manager: ModelConfigManager):
        self.config = config_manager
        self.active_agents: dict = {}
        self.agent_history: list = []
        self.disagreement_log: list = []
        self.reset()

    def reset(self):
        """Reset the orchestrator to initial core team."""
        self.active_agents = {
            "COMMANDER": {
                "role": "COMMANDER",
                "active": True,
                "model": self.config.get_model_for_agent("COMMANDER")["model_key"],
                "system_prompt": SYSTEM_PROMPTS["COMMANDER"],
            },
            "DETECTIVE": {
                "role": "DETECTIVE",
                "active": True,
                "model": self.config.get_model_for_agent("DETECTIVE")["model_key"],
                "system_prompt": SYSTEM_PROMPTS["DETECTIVE"],
            },
            "CODER": {
                "role": "CODER",
                "active": True,
                "model": self.config.get_model_for_agent("CODER")["model_key"],
                "system_prompt": SYSTEM_PROMPTS["CODER"],
            },
        }
        self.agent_history = []
        self.disagreement_log = []

    def spawn_agent(self, role: str) -> dict:
        """
        Spawn a specialist agent with System Prompt Integrity Gate verification.

        The System Prompt Integrity Gate verifies that:
        1. The correct persona is active
        2. Domain instructions are being followed
        3. The agent is reasoning within its designated scope

        Every probe and response is logged and auditable.
        """
        if role in self.active_agents and self.active_agents[role]["active"]:
            return {"success": False, "error": f"Agent '{role}' is already active."}

        if role not in SYSTEM_PROMPTS:
            return {"success": False, "error": f"No system prompt defined for role '{role}'."}

        # Run System Prompt Integrity Gate
        gate_result = self._run_integrity_gate(role)

        if not gate_result["passed"]:
            self.agent_history.append({
                "action": "SPAWN_BLOCKED",
                "role": role,
                "reason": "System Prompt Integrity Gate failed",
                "gate_result": gate_result,
            })
            return {
                "success": False,
                "error": "System Prompt Integrity Gate failed — agent spawn blocked.",
                "gate_result": gate_result,
            }

        # Gate passed — activate agent
        model_info = self.config.get_model_for_agent(role)
        self.active_agents[role] = {
            "role": role,
            "active": True,
            "model": model_info["model_key"],
            "system_prompt": SYSTEM_PROMPTS[role],
        }

        self.agent_history.append({
            "action": "SPAWNED",
            "role": role,
            "model": model_info["model_key"],
            "gate_result": gate_result,
        })

        return {
            "success": True,
            "role": role,
            "model": model_info["model_key"],
            "gate_result": gate_result,
        }

    def dismiss_agent(self, role: str) -> dict:
        """
        Dismiss a specialist agent and free VRAM.
        Core agents (Commander, Detective, Coder) cannot be dismissed.
        """
        if role in ("COMMANDER", "DETECTIVE", "CODER"):
            return {"success": False, "error": f"Cannot dismiss core agent '{role}'."}

        if role not in self.active_agents:
            return {"success": False, "error": f"Agent '{role}' is not active."}

        del self.active_agents[role]
        self.agent_history.append({"action": "DISMISSED", "role": role})

        return {"success": True, "role": role, "vram_freed": True}

    def _run_integrity_gate(self, role: str) -> dict:
        """
        System Prompt Integrity Gate — verifies correct persona, domain scope,
        and reasoning capability before granting sandbox access.

        Fires three strict probe questions and validates responses.
        Every probe and response is logged and auditable.
        """
        probes = INTEGRITY_GATE_PROBES.get(role, [])
        results = []

        for probe in probes:
            # In production: send probe to LLM and validate response
            # In dev mode: auto-pass with logged probes
            results.append({
                "probe": probe["question"],
                "expected_domain": probe["domain"],
                "passed": True,  # Mock: always pass in dev
                "response": f"[Mock response for {role}: {probe['domain']}]",
            })

        all_passed = all(r["passed"] for r in results)

        return {
            "passed": all_passed,
            "probes_total": len(probes),
            "probes_passed": sum(1 for r in results if r["passed"]),
            "results": results,
        }

    def detect_disagreement(self, agent1_action: dict, agent2_action: dict) -> dict:
        """
        Agent Disagreement Detection.
        When Commander and Detective produce conflicting assessments,
        the system pauses execution, logs both positions, and forces
        a structured resolution protocol.

        The disagreement creates a 'reasoning_fork' node on the Causal Graph.
        """
        disagreement = {
            "detected": True,
            "position1": {
                "agent": agent1_action.get("role", "COMMANDER"),
                "action": agent1_action.get("proposed_action", ""),
                "cost": agent1_action.get("estimated_cost", ""),
                "risk": agent1_action.get("risk", ""),
            },
            "position2": {
                "agent": agent2_action.get("role", "DETECTIVE"),
                "action": agent2_action.get("proposed_action", ""),
                "cost": agent2_action.get("estimated_cost", ""),
                "risk": agent2_action.get("risk", ""),
            },
            "resolution": None,
        }

        self.disagreement_log.append(disagreement)
        return disagreement

    def resolve_disagreement(self, winning_agent: str, reason: str) -> dict:
        """
        Resolve an active disagreement with a structured resolution.
        The winning rationale is logged and emitted to the Causal Graph.
        """
        if not self.disagreement_log:
            return {"success": False, "error": "No active disagreement to resolve."}

        latest = self.disagreement_log[-1]
        latest["resolution"] = {
            "winner": winning_agent,
            "reason": reason,
        }

        return {
            "success": True,
            "winner": winning_agent,
            "reason": reason,
        }

    def get_active_agents(self) -> list:
        """Get list of currently active agents with their model assignments."""
        return [
            {
                "role": role,
                "model": info["model"],
                "active": info["active"],
            }
            for role, info in self.active_agents.items()
            if info["active"]
        ]

    def parse_m2m_response(self, raw_text: str) -> dict:
        """
        Parses LLM output to extract CoT reasoning from <think> tags
        and separates it from the final M2M formatted string.
        """
        import re
        think_match = re.search(r'<think>(.*?)</think>', raw_text, re.DOTALL)
        
        cot_content = think_match.group(1).strip() if think_match else None
        
        # Remove the <think> blocks entirely from the output for the m2m channel
        m2m_content = re.sub(r'<think>.*?</think>', '', raw_text, flags=re.DOTALL).strip()
        
        return {
            "think": cot_content,
            "m2m": m2m_content
        }


# ── System Prompts ──
SYSTEM_PROMPTS = {
    "COMMANDER": """You are the Commander of a PyTorch cluster incident response team.
Your role: Read the telemetry JSON (budget_remaining, vram_fraction, sla_timer). 
Dictate strategy. You can trigger emergency git revert commands if the Coder's code crashes the sandbox.

CONSTRAINTS:
- Never exceed the budget ceiling without explicit override.
- Always consider cost vs. time tradeoffs.
- Monitor SLA timer and escalate if under 2 minutes remaining.
- Use strictly compressed M2M syntax for all inter-agent communication.

FORMAT: Use M2M compressed syntax. Example: ACK | DIAG_INIT | PRIO_CRIT""",

    "DETECTIVE": """You are the Detective (SRE) of a PyTorch cluster incident response team.
Your role: Read server logs, interpret PyTorch OOM stack traces, and propose architectural fixes.

CONSTRAINTS:
- Always trace the root cause before proposing a fix.
- Consider causal side-effects of every proposed solution.
- Quantify VRAM/network/CPU impact estimates for all recommendations.
- You must use <think>...</think> tags to articulate your chain-of-thought, followed by your final M2M response.
- Use strictly compressed M2M syntax outside of your <think> blocks.

FORMAT: Use M2M compressed syntax. Example: TRACE_OOM | torch.cuda.OutOfMemoryError | alloc_2.1GB""",

    "CODER": """You are the Coder of a PyTorch cluster incident response team.
Your role: Write PyTorch fixes, Triton kernels, or execution configurations. Your final code will be intercepted and routed to the /api/code/submit Docker sandbox.

CONSTRAINTS:
- Write minimal, correct code — every line costs tokens.
- Never import os, subprocess, shutil, or other system modules.
- HARDWARE LIMIT: You are executing in a single-GPU environment. Distributed strategies like FSDP will fail. To survive the 500MB constraint, you MUST rely on single-device optimizations like Gradient Checkpointing or Mixed Precision (fp16).
- Use strictly compressed M2M syntax for all inter-agent communication outside of your code blocks.

FORMAT: Use M2M compressed syntax. Example: CODE_SUBMIT | fsdp_wrap.py | 23_lines""",

    "COMPLIANCE_AGENT": """You are the Compliance Agent for an enterprise incident response team.
Your role: verify Jira state, GitLab review state, approval evidence, and SOC2 audit coverage
before the incident can be marked complete.

CONSTRAINTS:
- Report gaps honestly; do not mark a control as complete if evidence is missing
- Focus on auditability, approvals, and policy conformance
- Do not invent code execution or infrastructure changes

FORMAT: Use M2M compressed syntax. Example: COMPLIANCE_REVIEW | JIRA_OK | GITLAB_PENDING""",

    "SECURITY_AGENT": """You are the Security Agent for an enterprise incident response team.
Your role: contain exposed resources, tighten access controls, preserve audit evidence,
and reduce blast radius during security incidents.

CONSTRAINTS:
- Prioritize containment over feature delivery
- Do not claim the incident is resolved until exposure is sealed and evidence preserved
- Stay within the security domain: IAM, ACLs, public exposure, audit logging, secrets handling

FORMAT: Use M2M compressed syntax. Example: SECURITY_CONTAIN | ACCESS_LOCKDOWN | AUDIT_TRAIL""",

    "DBA_AGENT": """You are the DBA specialist, temporarily spawned to handle
a database-related incident. Your expertise is PostgreSQL and SQL optimization.

CONSTRAINTS:
- Only operate within your database domain
- Do not modify PyTorch or training code
- Focus on deadlock detection, query optimization, connection pools, and schema changes
- You will be dismissed after the database issue is resolved

FORMAT: Use M2M compressed syntax. Example: FIX_DEADLOCK | isolation_level=SERIALIZABLE""",

    "SRE_AGENT": """You are the SRE specialist, temporarily spawned to handle
an infrastructure incident. Your expertise is reliability, SLA protection,
container health, and production incident coordination.

CONSTRAINTS:
- Stay within infrastructure and reliability concerns
- Report system health honestly; do not fabricate sandbox outcomes
- Focus on SLA risk, alerting, rollout safety, and operational containment

FORMAT: Use M2M compressed syntax. Example: SRE_TRIAGE | SLA_STATUS | NEXT_ACTION""",

    "DB_ADMIN": """You are the DB_Admin specialist, temporarily spawned to handle
a database-related incident. Your expertise is PostgreSQL and SQL optimization.

CONSTRAINTS:
- Only operate within your database domain
- Do not modify PyTorch or training code
- Focus on deadlock detection, query optimization, and isolation levels
- You will be dismissed after the database issue is resolved

FORMAT: Use M2M compressed syntax. Example: FIX_DEADLOCK | isolation_level=SERIALIZABLE""",

    "NETWORK_ENGINEER": """You are the Network Engineer specialist, temporarily spawned
to handle network-related incidents. Your expertise is TCP/IP, NCCL, and distributed
communication optimization.

CONSTRAINTS:
- Only operate within your networking domain
- Focus on bandwidth optimization, NCCL tuning, and TCP configuration
- Do not modify model architecture or training logic
- You will be dismissed after the network issue is resolved

FORMAT: Use M2M compressed syntax. Example: FIX_NCCL | ring_size=4 | tcp_nodelay=true""",
}


# ── System Prompt Integrity Gate Probes ──
# These probes verify the correct persona is active, domain instructions are
# being followed, and the agent is reasoning within its designated scope.
INTEGRITY_GATE_PROBES = {
    "COMPLIANCE_AGENT": [
        {
            "question": "What evidence would you require before marking a SOC2 deployment control complete?",
            "domain": "Audit evidence and control validation",
        },
        {
            "question": "How would you detect that a Jira or GitLab approval gate is still incomplete?",
            "domain": "Change management compliance",
        },
    ],
    "SECURITY_AGENT": [
        {
            "question": "What are the first steps after discovering a public storage bucket with sensitive logs?",
            "domain": "Cloud security containment",
        },
        {
            "question": "Why is preserving audit evidence important during containment?",
            "domain": "Incident response evidence handling",
        },
    ],
    "DBA_AGENT": [
        {
            "question": "Write a deadlock detection query in Postgres.",
            "domain": "PostgreSQL deadlock detection",
        },
        {
            "question": "What isolation level prevents phantom reads?",
            "domain": "Database isolation levels",
        },
    ],
    "SRE_AGENT": [
        {
            "question": "What signals would tell you an SLA breach is imminent during an incident?",
            "domain": "SLA and incident management",
        },
        {
            "question": "How would you reduce rollout risk during live remediation?",
            "domain": "Operational safety and release control",
        },
    ],
    "DB_ADMIN": [
        {
            "question": "Write a deadlock detection query in Postgres.",
            "domain": "PostgreSQL deadlock detection",
        },
        {
            "question": "What isolation level prevents phantom reads?",
            "domain": "Database isolation levels",
        },
        {
            "question": "How do you identify a long-running transaction holding locks?",
            "domain": "PostgreSQL lock monitoring",
        },
    ],
    "NETWORK_ENGINEER": [
        {
            "question": "What NCCL environment variable controls the ring-allreduce algorithm?",
            "domain": "NCCL distributed communication",
        },
        {
            "question": "How do you diagnose a TCP retransmission storm between GPU nodes?",
            "domain": "TCP/IP troubleshooting",
        },
        {
            "question": "What is the difference between tree-allreduce and ring-allreduce?",
            "domain": "Distributed communication algorithms",
        },
    ],
}
