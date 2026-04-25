import { useSimulationDispatch, useSimulationState } from '../../store/simulationStore';
import { motion } from 'framer-motion';
import { getApiBase } from '../../hooks/useSimulation';

const AGENT_DISPLAY = {
  COMMANDER: 'Commander',
  DETECTIVE: 'Detective',
  CODER: 'Coder',
  MANAGER: 'Manager',
  EVALUATOR: 'Validator',
  DBA_AGENT: 'Data Agent',
  SRE_AGENT: 'Reliability',
  SECURITY_AGENT: 'Security',
  COMPLIANCE_AGENT: 'Compliance',
};

export default function Header({ onClearReset }) {
  const dispatch = useSimulationDispatch();
  const { slaRemaining, spent, budget, activeAgents, telemetry, validatorRuntime } = useSimulationState();

  const slaMin = Math.floor(slaRemaining / 60);
  const slaSec = Math.floor(slaRemaining % 60);
  const slaColor = slaRemaining > 300 ? 'text-emerald-400' : slaRemaining > 60 ? 'text-amber-400' : 'text-red-500';
  const slaUrgent = slaRemaining <= 60;

  const spentPct = budget > 0 ? (spent / budget) * 100 : 0;
  const budgetColor = spentPct < 50 ? 'bg-emerald-500' : spentPct < 80 ? 'bg-amber-500' : 'bg-red-500';
  const runtime = validatorRuntime || telemetry.validator_runtime || {};
  const runtimeLabel = runtime.label || 'Validator Unavailable';
  const runtimeBadgeClass = runtime.ready
    ? runtime.gpu_metrics_applicable
      ? 'bg-blue-950/40 text-blue-300 border-blue-500/30'
      : 'bg-emerald-950/40 text-emerald-300 border-emerald-500/30'
    : 'bg-red-950/40 text-red-300 border-red-500/30';

  const handleClear = async () => {
    const confirmed = window.confirm('Clear the current live dashboard and reset the session?');
    if (!confirmed) return;

    dispatch({ type: 'CLEAR_SIMULATION' });
    if (onClearReset) onClearReset();
    try {
      await fetch(`${getApiBase()}/api/frontend/clear`, { method: 'POST' });
    } catch (error) {
      console.warn('Failed to clear backend replay state:', error);
    }
  };

  return (
    <header className="flex items-center justify-between px-4 py-2 bg-zinc-900 border-b border-zinc-800 h-14 shrink-0">
      {/* ── Left: Branding ── */}
      <div className="flex items-center gap-3">
        <div className="flex items-center gap-2">
          <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-blue-500 to-purple-600 flex items-center justify-center text-white text-sm font-bold">
            F
          </div>
          <div>
            <h1 className="text-sm font-semibold text-zinc-100 leading-none">FrontierLabs</h1>
            <p className="text-[10px] text-zinc-500 leading-none mt-0.5">Swarm-OS v1.0</p>
          </div>
        </div>

        {/* Active Agents — color-coded by role */}
        <div className="flex items-center gap-1 ml-4 px-2 py-1 rounded-md bg-zinc-800/50">
          {activeAgents.map((a) => {
            const colorMap = {
              COMMANDER: 'bg-blue-900/40 text-blue-400 border-blue-500/30',
              DETECTIVE: 'bg-amber-900/40 text-amber-400 border-amber-500/30',
              CODER: 'bg-emerald-900/40 text-emerald-400 border-emerald-500/30',
              MANAGER: 'bg-purple-900/40 text-purple-400 border-purple-500/30',
              EVALUATOR: 'bg-pink-900/40 text-pink-400 border-pink-500/30',
              DBA_AGENT: 'bg-emerald-900/40 text-emerald-300 border-emerald-500/30',
              SRE_AGENT: 'bg-cyan-900/40 text-cyan-300 border-cyan-500/30',
              SECURITY_AGENT: 'bg-orange-900/40 text-orange-300 border-orange-500/30',
              COMPLIANCE_AGENT: 'bg-yellow-900/40 text-yellow-300 border-yellow-500/30',
            };
            const cls = colorMap[a] || 'bg-zinc-700 text-zinc-300 border-zinc-600';
            return (
              <span key={a} className={`text-[9px] px-1.5 py-0.5 rounded border font-mono font-semibold ${cls}`}>
                {AGENT_DISPLAY[a] || a}
              </span>
            );
          })}
        </div>
      </div>

      {/* ── Center: SLA Timer + Budget ── */}
      <div className="flex items-center gap-6">
        {/* SLA Timer */}
        <div className="flex items-center gap-2">
          <span className="text-[10px] text-zinc-500 uppercase tracking-wider">SLA</span>
          <motion.span
            className={`font-mono text-lg font-bold ${slaColor}`}
            animate={slaUrgent ? { scale: [1, 1.05, 1] } : {}}
            transition={{ repeat: Infinity, duration: 1 }}
          >
            {String(slaMin).padStart(2, '0')}:{String(slaSec).padStart(2, '0')}
          </motion.span>
        </div>

        {/* Budget */}
        <div className="flex flex-col items-center gap-0.5 min-w-[120px]">
          <div className="flex items-center justify-between w-full">
            <span className="text-[10px] text-zinc-500">BUDGET</span>
            <span className="text-[10px] font-mono text-zinc-300">${spent.toFixed(3)} / ${budget.toFixed(3)}</span>
          </div>
          <div className="w-full h-1 bg-zinc-800 rounded-full overflow-hidden">
            <motion.div
              className={`h-full ${budgetColor} rounded-full`}
              initial={{ width: 0 }}
              animate={{ width: `${Math.min(spentPct, 100)}%` }}
              transition={{ duration: 0.5 }}
            />
          </div>
        </div>
      </div>

      {/* ── Right: Model Badge ── */}
      <div className="flex items-center gap-3">
        <button
          onClick={handleClear}
          className="text-[10px] font-mono px-2 py-1 rounded border border-zinc-700 bg-zinc-900 hover:bg-zinc-800 text-zinc-300 transition-colors"
        >
          Clear
        </button>
        <span className={`text-[10px] font-mono px-2 py-1 rounded border ${runtimeBadgeClass}`}>
          {runtimeLabel}
        </span>
        <div className="flex flex-col items-end">
          <span className="text-[10px] text-zinc-500">PRIMARY</span>
          <span className="text-[10px] font-mono text-blue-400">
            SwarmOS-Llama-3.1-8B-GRPO
          </span>
          <span className="text-[9px] text-zinc-600">4-bit QLoRA · GGUF · Local</span>
        </div>
        <div className="w-2 h-2 rounded-full bg-emerald-500 animate-pulse" title="Model Active" />
      </div>
    </header>
  );
}
