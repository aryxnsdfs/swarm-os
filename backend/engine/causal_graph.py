"""
Causal Graph Engine
Builds and traverses a Directed Acyclic Graph (DAG) mapping the causal chain
of events. Every agent action writes a node to the DAG. Used for root cause
analysis generation and live visualization.
"""

import logging
import re
from typing import Optional

logger = logging.getLogger("swarm-os.causal")


class CausalGraphEngine:
    def __init__(self):
        self.reset()

    def reset(self):
        """Reset the causal graph to empty state."""
        self.nodes: list = []
        self.edges: list = []
        self._node_map: dict = {}  # id -> node

    def add_node(self, node_id: str, label: str, node_type: str,
                 detail: str = "", parent_id: Optional[str] = None,
                 display_detail: Optional[str] = None) -> dict:
        """
        Add a node to the causal graph.

        Args:
            node_id: Unique identifier for the node
            label: Human-readable label
            node_type: One of 'error', 'fix', 'escalation', 'resolution', 'fork'
            detail: Additional detail text
            parent_id: ID of the parent node (creates an edge)

        Returns:
            The created node dict
        """
        node = {
            "id": node_id,
            "label": label,
            "type": node_type,
            "detail": detail,
            "display_detail": display_detail if display_detail is not None else detail,
        }
        self.nodes.append(node)
        self._node_map[node_id] = node

        edge = None
        if parent_id and parent_id in self._node_map:
            edge = {
                "id": f"{parent_id}-{node_id}",
                "source": parent_id,
                "target": node_id,
                "animated": True,
            }
            self.edges.append(edge)

        return {"node": node, "edge": edge}

    def get_graph(self) -> dict:
        """Get the full graph state for frontend rendering."""
        return {
            "nodes": self.nodes,
            "edges": self.edges,
        }

    def get_chain(self) -> list:
        """
        Get the causal chain as an ordered list from root to leaf.
        Traverses the DAG from nodes with no parent to leaf nodes.
        """
        # Find root nodes (nodes that are not targets of any edge)
        target_ids = {e["target"] for e in self.edges}
        roots = [n for n in self.nodes if n["id"] not in target_ids]

        chain = []
        visited = set()

        def traverse(node_id):
            if node_id in visited:
                return
            visited.add(node_id)
            node = self._node_map.get(node_id)
            if node:
                chain.append(node)
            # Find children
            children = [e["target"] for e in self.edges if e["source"] == node_id]
            for child_id in children:
                traverse(child_id)

        for root in roots:
            traverse(root["id"])

        return chain

    def find_root_cause(self) -> Optional[dict]:
        """Find the root cause node (first error node in the chain)."""
        chain = self.get_chain()
        for node in chain:
            if node["type"] == "error":
                return node
        return None

    def generate_rca(self) -> str:
        """
        Generate a Root Cause Analysis document from the causal chain.
        Returns a formatted Markdown string with clean table layouts.
        """
        chain = self.get_chain()
        if not chain:
            return "No causal data available."

        root_cause = self.find_root_cause()
        fixes = [n for n in chain if n["type"] == "fix"]
        escalations = [n for n in chain if n["type"] == "escalation"]
        resolutions = [n for n in chain if n["type"] == "resolution"]

        def clean_detail(text):
            """Normalize RCA text for markdown rendering."""
            if not text:
                return "—"
            t = re.sub(r"<[^>]+>", " ", text)
            t = re.sub(r"\s+", " ", t.replace("\n", " ")).strip()
            if "import torch" in t or "def " in t or "class " in t:
                t = "Executable remediation content recorded."
            return t[:120].replace("|", "·")

        rca = "# Auto-Generated Root Cause Analysis\n\n"

        if root_cause:
            rca += "## Root Cause\n\n"
            rca += "| Field | Detail |\n"
            rca += "|---|---|\n"
            rca += f"| **Trigger** | {root_cause['label']} |\n"
            rca += f"| **Detail** | {clean_detail(root_cause['detail'])} |\n"
            rca += f"| **Type** | `{root_cause['type']}` |\n\n"

        rca += "## Causal Chain\n\n"
        rca += "| Step | Event | Type | Detail |\n"
        rca += "|---|---|---|---|\n"
        for i, node in enumerate(chain):
            rca += f"| {i+1} | {node['label']} | `{node['type']}` | {clean_detail(node['detail'])} |\n"
        rca += "\n"

        if fixes:
            rca += "## Fixes Applied\n\n"
            rca += "| Fix | Detail |\n"
            rca += "|---|---|\n"
            for fix in fixes:
                rca += f"| {fix['label']} | {clean_detail(fix['detail'])} |\n"
            rca += "\n"

        if escalations:
            rca += "## Escalations\n\n"
            rca += "| Event | Detail |\n"
            rca += "|---|---|\n"
            for esc in escalations:
                rca += f"| {esc['label']} | {clean_detail(esc['detail'])} |\n"
            rca += "\n"

        if resolutions:
            rca += "## Resolution\n\n"
            rca += "| Outcome | Detail |\n"
            rca += "|---|---|\n"
            for res in resolutions:
                rca += f"| {res['label']} | {clean_detail(res['detail'])} |\n"
            rca += "\n"

        return rca
