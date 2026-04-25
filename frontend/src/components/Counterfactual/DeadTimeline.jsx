import { motion } from 'framer-motion';
import { useSimulationState } from '../../store/simulationStore';

function TimelineCard({ title, dotClass, borderClass, bgClass, metrics, progressClass, progressPct }) {
  return (
    <div className={`rounded-lg border ${borderClass} ${bgClass} p-3 flex flex-col gap-3 min-w-0`}>
      <div className="flex items-center gap-1.5">
        <span className={`w-2 h-2 rounded-full ${dotClass}`} />
        <span className="text-[10px] font-semibold uppercase tracking-wider text-zinc-200">{title}</span>
      </div>

      <div className="space-y-2">
        {metrics.map((metric) => (
          <div key={metric.label} className="flex items-start justify-between gap-3">
            <span className="text-[10px] text-zinc-500 shrink-0">{metric.label}</span>
            <span className={`text-[11px] font-mono text-right break-words ${metric.valueClass}`}>
              {metric.value}
            </span>
          </div>
        ))}
      </div>

      <div className="space-y-1">
        <div className="flex items-center justify-between text-[10px] text-zinc-500">
          <span>Execution Track</span>
          <span className="font-mono">{progressPct}%</span>
        </div>
        <div className="h-1.5 rounded-full bg-zinc-800 overflow-hidden">
          <motion.div
            className={`h-full ${progressClass}`}
            initial={{ width: 0 }}
            animate={{ width: `${progressPct}%` }}
            transition={{ duration: 0.5, ease: 'easeOut' }}
          />
        </div>
      </div>
    </div>
  );
}

export default function DeadTimeline() {
  const { counterfactual, spent, budget, elapsedMs, telemetry, scenarioComplete, isRunning } = useSimulationState();
  const liveSeconds = Math.max(1, Math.floor(elapsedMs / 1000));
  const liveSpent = Number(spent || 0);

  const normalizeMetric = (value, fallback) => {
    if (typeof value !== 'string') return value ?? fallback;
    return value.includes('...') ? fallback : value;
  };

  const actual = counterfactual?.actual || {
    time: `${liveSeconds}s`,
    cost: `$${liveSpent.toFixed(3)}`,
    sla: telemetry.sla_remaining_seconds > 0 ? 'SAFE' : 'BREACHED',
    outcome: scenarioComplete ? 'RESOLVED' : 'COMPUTING...',
  };

  const dead = counterfactual?.dead || {
    time: `${Math.max(10, Math.floor(liveSeconds * 2.4) || 18)}s`,
    cost: `$${Math.max(liveSpent * 4.5, 0.25).toFixed(2)}`,
    sla: 'BREACHED',
    outcome: scenarioComplete ? 'MANUAL_ESCALATION' : 'PROJECTING FALLBACK...',
  };

  const showPlaceholder = !isRunning && !scenarioComplete && !counterfactual;

  const actualPct = Math.max(8, Math.min(100, budget > 0 ? Math.round((liveSpent / budget) * 100) : 0));
  const deadPct = Math.max(actualPct + 18, 72);

  return (
    <motion.div
      initial={{ opacity: 0, scale: 0.98 }}
      animate={{ opacity: 1, scale: 1 }}
      className="panel-card p-3 flex flex-col"
    >
      <div className="flex items-center justify-between mb-3">
        <span className="text-[10px] font-bold text-zinc-400 uppercase tracking-widest flex items-center gap-1.5">
          <span className="w-1 h-3 bg-zinc-600 rounded-full" />
          Counterfactual Analysis
        </span>
        <span className="text-[9px] font-mono text-zinc-600">
          {showPlaceholder ? 'waiting' : counterfactual ? 'live comparison ready' : 'tracking live estimate'}
        </span>
      </div>

      {showPlaceholder ? (
        <div className="h-32 flex items-center justify-center border border-zinc-800/50 bg-zinc-900/20 rounded-lg">
          <p className="text-[10px] text-zinc-600 font-mono italic">Awaiting scenario execution...</p>
        </div>
      ) : (
        <div className="grid grid-cols-1 xl:grid-cols-2 gap-2">
          <TimelineCard
          title="Actual Timeline (Swarm OS)"
          dotClass="bg-emerald-500"
          borderClass="border-emerald-500/20 shadow-[0_0_15px_-5px_rgba(16,185,129,0.1)]"
          bgClass="bg-emerald-500/[0.03]"
          progressClass="bg-emerald-500"
          progressPct={actualPct}
          metrics={[
            { label: 'Resolution Cost', value: normalizeMetric(actual.cost, `$${liveSpent.toFixed(3)}`), valueClass: 'text-emerald-400' },
            { label: 'Time', value: normalizeMetric(actual.time, `${liveSeconds}s`), valueClass: 'text-emerald-300' },
            { label: 'SLA Status', value: normalizeMetric(actual.sla, telemetry.sla_remaining_seconds > 0 ? 'SAFE' : 'BREACHED'), valueClass: 'text-emerald-300 font-bold' },
            { label: 'Outcome', value: normalizeMetric(actual.outcome, scenarioComplete ? 'RESOLVED' : 'COMPUTING...'), valueClass: 'text-emerald-300' },
          ]}
        />

        <TimelineCard
          title="Dead Timeline (Human Manual)"
          dotClass="bg-red-500"
          borderClass="border-red-500/20 opacity-60 grayscale-[0.5]"
          bgClass="bg-red-500/[0.03]"
          progressClass="bg-red-500"
          progressPct={Math.min(deadPct, 100)}
          metrics={[
            { label: 'Projected Cost', value: normalizeMetric(dead.cost, `$${Math.max(liveSpent * 4.5, 0.25).toFixed(2)}`), valueClass: 'text-red-400' },
            { label: 'Time', value: normalizeMetric(dead.time, `${Math.max(10, Math.floor(liveSeconds * 2.4))}s`), valueClass: 'text-red-300' },
            { label: 'SLA Status', value: normalizeMetric(dead.sla, 'BREACHED'), valueClass: 'text-red-300 font-bold' },
            { label: 'Outcome', value: normalizeMetric(dead.outcome, scenarioComplete ? 'MANUAL_ESCALATION' : 'PROJECTING FALLBACK...'), valueClass: 'text-red-300' },
          ]}
        />
      </div>
      )}
    </motion.div>
  );
}
