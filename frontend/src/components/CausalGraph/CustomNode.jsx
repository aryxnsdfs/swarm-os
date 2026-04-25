import { Handle, Position } from 'reactflow';

const TYPE_STYLES = {
  error:      { symbol: 'ERR', bg: '#ef444420', border: '#ef4444' },
  fix:        { symbol: 'FIX', bg: '#10b98120', border: '#10b981' },
  escalation: { symbol: 'ESC', bg: '#f59e0b20', border: '#f59e0b' },
  resolution: { symbol: 'RES', bg: '#3b82f620', border: '#3b82f6' },
  fork:       { symbol: 'FRK', bg: '#a855f720', border: '#a855f7' },
};

export default function CustomNode({ data }) {
  const { label, nodeType, detail, color } = data;
  const style = TYPE_STYLES[nodeType] || { symbol: '---', bg: '#71717a20', border: '#71717a' };

  return (
    <div
      className="px-3 py-2 rounded-lg border min-w-[160px] max-w-[220px]"
      style={{
        backgroundColor: '#18181b',
        borderColor: color + '60',
        boxShadow: `0 0 8px ${color}15`,
      }}
    >
      <Handle type="target" position={Position.Left} className="!bg-zinc-600 !w-2 !h-2 !border-0" />
      <div className="flex items-center gap-1.5 mb-1">
        <span
          className="text-[8px] font-bold font-mono px-1 py-0.5 rounded"
          style={{ backgroundColor: style.bg, color: style.border, border: `1px solid ${style.border}40` }}
        >
          {style.symbol}
        </span>
        <span className="text-[11px] font-semibold text-zinc-200 leading-tight">{label}</span>
        {data.points !== undefined && (
           <span className={`ml-auto font-mono text-[9px] font-bold ${data.points > 0 ? 'text-emerald-400' : 'text-red-400'}`}>
              {data.points > 0 ? '+' : ''}{data.points.toFixed(2)}
           </span>
        )}
      </div>
      {detail && (
        <p className="text-[9px] font-mono leading-tight mt-1" style={{ color: color || '#a1a1aa' }}>
          {detail.replace(/\*\*/g, '').replace(/\*/g, '')}
        </p>
      )}
      <Handle type="source" position={Position.Right} className="!bg-zinc-600 !w-2 !h-2 !border-0" />
    </div>
  );
}
