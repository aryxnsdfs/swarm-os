import { useEffect, useMemo, useState } from 'react';
import { motion } from 'framer-motion';
import { useSimulationState } from '../../store/simulationStore';

function toCheckState(value) {
  if (value === true) return 'pass';
  if (value === false) return 'fail';
  return 'pending';
}

function parseStatus(value) {
  const normalized = String(value || '').toUpperCase();
  if (!normalized) return 'pending';
  if (normalized === 'PASS' || normalized === 'STABLE' || normalized === 'RUNNING') return 'pass';
  if (normalized === 'FAIL' || normalized === 'FAILED' || normalized === 'CRITICAL') return 'fail';
  return 'pending';
}

function getRowStyles(state, isFocused) {
  if (state === 'pass') {
    return {
      border: 'border-emerald-500/40',
      bg: 'bg-emerald-500/10',
      text: 'text-emerald-300',
      mark: '✓',
      markClass: 'text-emerald-300',
    };
  }
  if (state === 'fail') {
    return {
      border: 'border-red-500/40',
      bg: 'bg-red-500/10',
      text: 'text-red-300',
      mark: '✗',
      markClass: 'text-red-300',
    };
  }
  return {
    border: isFocused ? 'border-blue-400/50' : 'border-zinc-800',
    bg: isFocused ? 'bg-blue-500/10' : 'bg-zinc-900/60',
    text: isFocused ? 'text-blue-300' : 'text-zinc-400',
    mark: '•',
    markClass: isFocused ? 'text-blue-300' : 'text-zinc-600',
  };
}

export default function FinOpsPreFlightAudit() {
  const {
    rewardFeed,
    telemetry,
    preflight,
    spent,
    budget,
    lastValidatorResult,
    chosenRun,
  } = useSimulationState();

  const [activeStepIdx, setActiveStepIdx] = useState(-1);

  const coderSignal = useMemo(() => {
    for (let i = rewardFeed.length - 1; i >= 0; i -= 1) {
      const agent = String(rewardFeed[i]?.agent || '').toUpperCase();
      if (agent.includes('CODER') || agent.includes('ENGINEER')) {
        return rewardFeed[i].id;
      }
    }
    return null;
  }, [rewardFeed]);

  useEffect(() => {
    if (!coderSignal) return undefined;
    setActiveStepIdx(0);
    let step = 0;
    const timer = setInterval(() => {
      step += 1;
      if (step > 3) {
        clearInterval(timer);
        setActiveStepIdx(-1);
        return;
      }
      setActiveStepIdx(step);
    }, 280);
    return () => clearInterval(timer);
  }, [coderSignal]);

  const validator = lastValidatorResult || chosenRun || {};
  const astState = parseStatus(validator.status || telemetry.last_validator_status);
  const rawBudgetLimit = Number(telemetry.budget_limit_usd);
  const fallbackBudget = Number(budget);
  const budgetLimit = Number.isFinite(rawBudgetLimit) && rawBudgetLimit > 0
    ? rawBudgetLimit
    : Number.isFinite(fallbackBudget) && fallbackBudget > 0
      ? fallbackBudget
      : 50;
  const budgetSpent = Number.isFinite(Number(telemetry.cost_accrued_usd))
    ? Number(telemetry.cost_accrued_usd)
    : Number(spent || 0);
  const budgetState = toCheckState(
    (preflight?.budget !== false) && budgetSpent <= budgetLimit
  );
  const vramPeak = Number(
    validator.vram_peak_mb ??
      telemetry.vram_peak_mb ??
      telemetry.vram ??
      0
  );
  const vramState = vramPeak > 0 ? toCheckState(vramPeak <= 500) : 'pending';
  const dockerState = parseStatus(telemetry.containerStatus);

  const checks = [
    { id: 'ast', label: 'AST Syntax Check', state: astState, detail: astState === 'pending' ? 'Awaiting validator response' : validator.status || telemetry.last_validator_status || 'PASS' },
    { id: 'budget', label: `Budget Constraint ($${budgetLimit.toFixed(0)})`, state: budgetState, detail: `$${budgetSpent.toFixed(3)} / $${budgetLimit.toFixed(3)}` },
    { id: 'vram', label: 'VRAM Simulation (500MB Limit)', state: vramState, detail: vramPeak > 0 ? `${vramPeak.toFixed(0)}MB peak` : 'Awaiting VRAM sample' },
    { id: 'docker', label: 'Docker Execution Status', state: dockerState, detail: String(telemetry.containerStatus || 'idle').toUpperCase() },
  ];

  const failed = checks.some((c) => c.state === 'fail');
  const passed = checks.every((c) => c.state === 'pass');
  const gateLabel = failed ? 'BLOCKED' : passed ? 'CLEARED' : 'PENDING';
  const gateClass = failed ? 'text-red-400 border-red-500/40' : passed ? 'text-emerald-400 border-emerald-500/40' : 'text-zinc-400 border-zinc-700';

  return (
    <div className="panel-card p-3 h-full min-h-0 flex flex-col">
      <div className="flex items-center justify-between mb-3 pb-2 border-b border-zinc-800/60">
        <span className="text-[10px] font-bold text-zinc-400 uppercase tracking-widest">
          FinOps Pre-Flight Audit
        </span>
        <span className={`text-[10px] px-2 py-0.5 border rounded font-mono font-bold ${gateClass}`}>
          {gateLabel}
        </span>
      </div>

      <div className="flex-1 min-h-0 space-y-1.5 overflow-y-auto overflow-x-hidden pr-1 scrollbar-thin scrollbar-thumb-zinc-800 scrollbar-track-transparent">
        {checks.map((check, idx) => {
          const isFocused = activeStepIdx === idx;
          const styles = getRowStyles(check.state, isFocused);
          return (
            <motion.div
              key={check.id}
              animate={isFocused ? { scale: [1, 1.01, 1], opacity: [0.85, 1, 0.9] } : { scale: 1, opacity: 1 }}
              transition={{ duration: 0.45, repeat: isFocused ? Infinity : 0 }}
              className={`rounded-md border ${styles.border} ${styles.bg} px-2.5 py-2`}
            >
              <div className="flex items-center justify-between gap-2">
                <span className={`min-w-0 text-[10px] font-semibold leading-snug break-words ${styles.text}`}>{check.label}</span>
                <span className={`text-[11px] font-mono ${styles.markClass}`}>{styles.mark}</span>
              </div>
              <div className="text-[9px] text-zinc-500 font-mono mt-1 break-words">{check.detail}</div>
            </motion.div>
          );
        })}
      </div>
    </div>
  );
}
