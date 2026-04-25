import { useEffect, useMemo, useRef } from 'react';
import ReactFlow, { Background, Controls, useNodesState, useEdgesState } from 'reactflow';
import 'reactflow/dist/style.css';
import { useSimulationState } from '../../store/simulationStore';
import CustomNode from './CustomNode';

const nodeTypes = { custom: CustomNode };
const COLUMN_WIDTH = 260;
const ROW_HEIGHT = 140;
const MAX_COLUMNS = 4;

function buildWrappedLayout(nodes, edges) {
  const incoming = new Map();
  const outgoing = new Map();

  nodes.forEach((node) => {
    incoming.set(node.id, 0);
    outgoing.set(node.id, []);
  });

  edges.forEach((edge) => {
    if (!incoming.has(edge.target)) incoming.set(edge.target, 0);
    incoming.set(edge.target, (incoming.get(edge.target) || 0) + 1);
    if (!outgoing.has(edge.source)) outgoing.set(edge.source, []);
    outgoing.get(edge.source).push(edge.target);
  });

  const queue = nodes
    .filter((node) => (incoming.get(node.id) || 0) === 0)
    .map((node) => node.id);
  const orderedIds = [];
  const seen = new Set();

  while (queue.length > 0) {
    const current = queue.shift();
    if (seen.has(current)) continue;
    seen.add(current);
    orderedIds.push(current);
    const targets = outgoing.get(current) || [];
    targets.forEach((target) => {
      incoming.set(target, (incoming.get(target) || 1) - 1);
      if ((incoming.get(target) || 0) <= 0) {
        queue.push(target);
      }
    });
  }

  nodes.forEach((node) => {
    if (!seen.has(node.id)) {
      orderedIds.push(node.id);
    }
  });

  const positionById = new Map();
  orderedIds.forEach((id, index) => {
    const col = index % MAX_COLUMNS;
    const row = Math.floor(index / MAX_COLUMNS);
    positionById.set(id, {
      x: 40 + col * COLUMN_WIDTH,
      y: 40 + row * ROW_HEIGHT,
    });
  });

  return nodes.map((node) => ({
    ...node,
    position: positionById.get(node.id) || { x: 40, y: 40 },
  }));
}

export default function CausalDAG() {
  const { causalNodes, causalEdges } = useSimulationState();
  const [nodes, setNodes, onNodesChange] = useNodesState([]);
  const [edges, setEdges, onEdgesChange] = useEdgesState([]);
  const reactFlowRef = useRef(null);

  const layoutNodes = useMemo(
    () => buildWrappedLayout(causalNodes, causalEdges),
    [causalNodes, causalEdges]
  );

  useEffect(() => {
    setNodes(layoutNodes);
    setEdges(causalEdges);
  }, [layoutNodes, causalEdges, setNodes, setEdges]);

  useEffect(() => {
    if (!reactFlowRef.current || layoutNodes.length === 0) return;
    const timeout = setTimeout(() => {
      reactFlowRef.current.fitView({
        padding: 0.2,
        duration: 300,
        minZoom: 0.35,
        maxZoom: 1.2,
      });
    }, 50);
    return () => clearTimeout(timeout);
  }, [layoutNodes, causalEdges]);

  return (
    <div className="panel-card h-full flex flex-col overflow-hidden">
      <div className="flex items-center justify-between px-3 py-2 border-b border-zinc-800 shrink-0">
        <div className="flex items-center gap-1.5">
          <svg className="w-3.5 h-3.5 text-zinc-400" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.5">
            <circle cx="4" cy="4" r="2" />
            <circle cx="12" cy="12" r="2" />
            <circle cx="12" cy="4" r="2" />
            <line x1="6" y1="4" x2="10" y2="4" />
            <line x1="12" y1="6" x2="12" y2="10" />
          </svg>
          <span className="text-[10px] font-semibold text-zinc-400 uppercase tracking-wider">
            Causal Graph (DAG)
          </span>
        </div>
        <span className="text-[9px] font-mono text-zinc-600">
          {causalNodes.length} nodes / {causalEdges.length} edges
        </span>
      </div>
      <div className="flex-1 min-h-0">
        {causalNodes.length === 0 ? (
          <div className="flex flex-col items-center justify-center h-full text-zinc-600">
            <svg className="w-8 h-8 text-zinc-700 mb-2" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
              <circle cx="5" cy="6" r="3" />
              <circle cx="19" cy="6" r="3" />
              <circle cx="12" cy="18" r="3" />
              <line x1="7.5" y1="7.5" x2="10" y2="15.5" />
              <line x1="16.5" y1="7.5" x2="14" y2="15.5" />
            </svg>
            <p className="text-[10px]">Causal chain builds during scenario...</p>
          </div>
        ) : (
          <ReactFlow
            nodes={nodes}
            edges={edges}
            onNodesChange={onNodesChange}
            onEdgesChange={onEdgesChange}
            onInit={(instance) => { reactFlowRef.current = instance; }}
            nodeTypes={nodeTypes}
            fitView
            fitViewOptions={{ padding: 0.1, minZoom: 0.35, maxZoom: 1.5 }}
            minZoom={0.3}
            maxZoom={2.0}
            nodesDraggable={false}
            panOnDrag
            proOptions={{ hideAttribution: true }}
          >
            <Background color="#27272a" gap={20} size={1} />
            <Controls
              showInteractive={false}
              style={{ backgroundColor: '#27272a', borderColor: '#3f3f46' }}
            />
          </ReactFlow>
        )}
      </div>
    </div>
  );
}
