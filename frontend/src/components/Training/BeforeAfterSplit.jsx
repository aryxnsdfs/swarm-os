import { useSimulationState } from '../../store/simulationStore';

function MetricRow({ label, value, valueClass = 'text-zinc-200' }) {
  return (
    <div className="flex items-start justify-between gap-3 py-0.5 min-w-0">
      <span className="text-[11px] text-zinc-500 shrink-0">{label}</span>
      <span className={`min-w-0 text-[11px] font-mono text-right leading-snug break-words ${valueClass}`}>{value}</span>
    </div>
  );
}

function Panel({ accent, title, children, badge }) {
  return (
    <div className={`panel-card p-3 border ${accent} flex flex-col gap-2.5 h-full min-h-0`}>
      <div className="flex items-center gap-2 shrink-0 pb-1.5 border-b border-zinc-800/60">
        <span className={`w-2 h-2 rounded-full ${badge}`} />
        <span className="text-[10px] font-semibold uppercase tracking-wider text-zinc-200">{title}</span>
      </div>
      <div className="flex-1 min-h-0 pr-0.5 flex flex-col gap-2 overflow-y-auto overflow-x-hidden scrollbar-thin scrollbar-thumb-zinc-800 scrollbar-track-transparent">
        {children}
      </div>
    </div>
  );
}

export default function BeforeAfterSplit() {
  const {
    scenarioContext,
    chosenRun,
    rejectedRun,
    rcaDocument,
    lastValidatorResult,
    telemetry,
    counterfactual,
    scenarioComplete,
    isRunning,
    spent,
  } = useSimulationState();

  const runtime = telemetry.validator_runtime || scenarioContext?.validator_runtime || {};
  const validator = lastValidatorResult || chosenRun || {};
  const gpuMetricsApplicable = validator.gpu_metrics_applicable ?? runtime.gpu_metrics_applicable ?? false;
  const checksApplied = Array.isArray(validator.checks_applied) ? validator.checks_applied : [];

  // Phase 3: if we have a passing fix (chosenRun), the incident IS resolved —
  // don't wait for the counterfactual payload to arrive.
  const isResolved = scenarioComplete || Boolean(chosenRun);
  const closureOutcome =
    counterfactual?.actual?.outcome ||
    (isResolved ? 'RESOLVED' : 'IN_PROGRESS');
  const isIncomplete = closureOutcome === 'INCOMPLETE';
  const outcomeClass = isIncomplete
    ? 'text-red-400 font-bold'
    : isResolved
    ? 'text-emerald-400 font-bold'
    : 'text-zinc-400 font-bold';

  const closureNote = isResolved
    ? 'Success: The incident closed with validator-backed evidence, cost tracking, and replayable execution logs.'
    : 'Pending: Awaiting the final closure note and counterfactual summary.';

  return (
    <div
      className="grid gap-3 h-full min-h-0"
      style={{ gridTemplateColumns: 'repeat(3, minmax(0, 1fr))' }}
    >

      {/* ── Phase 1: Incident Trigger ── */}
      <Panel accent="border-red-500/20 shadow-[0_0_15px_-5px_rgba(239,68,68,0.1)]" badge="bg-red-500" title="Phase 1: Incident Trigger">
        <p className="text-[11px] text-zinc-300 leading-relaxed">
          {scenarioContext?.incident_summary || 'Waiting for the incident brief from the backend...'}
        </p>
        <div className="flex flex-col gap-0.5 pt-2 border-t border-zinc-800/60">
          <MetricRow label="Task Context"      value={scenarioContext?.title     || '—'} valueClass="text-red-400 font-bold" />
          <MetricRow label="Primary Objective" value={scenarioContext?.objective || '—'} valueClass="text-zinc-300" />
          <MetricRow
            label="System State"
            value={scenarioComplete ? 'Incident Reproduced' : 'Live Evidence Trace Active'}
            valueClass="text-red-400"
          />
        </div>
      </Panel>

      {/* ── Phase 2: Validation Proof ── */}
      <Panel accent="border-blue-500/20" badge="bg-blue-500" title="Phase 2: Validation Proof">
        <div className="flex flex-col gap-0.5">
          <MetricRow label="Validator Agent" value={validator.validation_label || runtime.label || '—'} valueClass="text-blue-300" />
          <MetricRow label="Pass Status"     value={validator.status || telemetry.last_validator_status || 'pending'} valueClass="text-emerald-400 font-bold" />
          <MetricRow
            label="Applied Checks"
            value={checksApplied.length ? checksApplied.join(', ') : 'Waiting for validator output...'}
            valueClass="text-zinc-400 italic"
          />
          {gpuMetricsApplicable ? (
            <MetricRow
              label="Peak VRAM"
              value={validator.vram_peak_mb ? `${validator.vram_peak_mb}MB` : 'Pending'}
              valueClass="text-emerald-300"
            />
          ) : (
            <MetricRow label="Sandbox Mode" value="Docker Plain-Python Workflow" valueClass="text-zinc-400" />
          )}
        </div>
        <div className="rounded-md border border-zinc-800 bg-zinc-950/60 px-2.5 py-2 text-[10px] text-zinc-400 leading-relaxed font-mono mt-1">
          {validator.validator_detail || telemetry.validator_detail || runtime.detail || 'Validator proof details will appear here after execution.'}
        </div>
      </Panel>

      {/* ── Phase 3: Incident Outcome ── */}
      <Panel accent="border-emerald-500/20" badge="bg-emerald-500" title="Phase 3: Incident Outcome">
        <div className="flex flex-col gap-0.5">
          <MetricRow
            label="Closure Outcome"
            value={closureOutcome}
            valueClass={outcomeClass}
          />
          <MetricRow
            label="Final Resolution Cost"
            value={counterfactual?.actual?.cost || `$${Number(spent || 0).toFixed(3)}`}
            valueClass="text-emerald-300"
          />
          <MetricRow
            label="Residual SLA Window"
            value={counterfactual?.actual?.sla || (telemetry.sla_remaining_seconds > 0 ? 'SAFE' : 'BREACHED')}
            valueClass={telemetry.sla_remaining_seconds > 0 || counterfactual?.actual?.sla === 'SAFE' ? 'text-emerald-400' : 'text-red-400'}
          />
        </div>
        <div className="rounded-md border border-emerald-500/30 bg-emerald-500/5 px-2.5 py-2 text-[10px] text-zinc-300 leading-relaxed italic mt-1">
          {closureNote}
        </div>
      </Panel>
    </div>
  );
}
