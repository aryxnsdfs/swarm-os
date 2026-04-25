import { useSimulationState } from '../../store/simulationStore';

function EvidenceCard({ label, present, detail, accent }) {
  return (
    <div className={`rounded-lg border p-3 ${present ? accent : 'border-zinc-800 bg-zinc-900/30'}`}>
      <div className="flex items-center justify-between gap-2">
        <span className="text-[10px] text-zinc-400 uppercase tracking-wider">{label}</span>
        <span className={`text-[10px] font-mono font-bold ${present ? 'text-emerald-400' : 'text-zinc-500'}`}>
          {present ? 'CAPTURED' : (label === 'Rejected Baseline' ? 'NOT REQUIRED' : 'PENDING')}
        </span>
      </div>
      <p className="text-[10px] text-zinc-500 mt-2 leading-relaxed">
        {detail}
      </p>
    </div>
  );
}

export default function FPSRChart() {
  const { rejectedRun, chosenRun, rcaDocument } = useSimulationState();
  const totalCaptured = [rejectedRun, chosenRun, rcaDocument].filter(Boolean).length;

  return (
    <div className="panel-card p-3 h-full flex flex-col">
      <div className="flex items-center justify-between mb-2">
        <span className="text-[10px] font-semibold text-zinc-400 uppercase tracking-wider">
          Outcome Snapshot
        </span>
        <span className="text-[10px] font-mono text-emerald-400 font-bold">
          {totalCaptured === 0 ? 'Awaiting Evidence' : `${totalCaptured} sample${totalCaptured === 1 ? '' : 's'} captured`}
        </span>
      </div>

      <div className="flex-1 min-h-0 overflow-y-auto scrollbar-thin scrollbar-thumb-zinc-700 pr-1">
        <div className="flex flex-col gap-3">
          <EvidenceCard
            label="Rejected Baseline"
            present={Boolean(rejectedRun)}
            detail={
              rejectedRun
                ? `Status ${rejectedRun.status}. ${rejectedRun.validator_detail || 'This earlier remediation attempt was rejected before the final passing fix.'}`
                : 'No rejected remediation sample was required for this run.'
            }
            accent="border-red-900/50 bg-red-950/10"
          />
          <EvidenceCard
            label="Passing Fix"
            present={Boolean(chosenRun)}
            detail={
              chosenRun
                ? `Status ${chosenRun.status}. ${chosenRun.validator_detail || 'Passing validator evidence and remediation details were recorded successfully.'}`
                : 'Waiting for a passing validator result to be recorded.'
            }
            accent="border-emerald-900/50 bg-emerald-950/10"
          />
          <EvidenceCard
            label="RCA Summary"
            present={Boolean(rcaDocument)}
            detail={
              rcaDocument
                ? 'A full incident RCA has been generated with the causal chain, remediation proof, and closure notes.'
                : 'The RCA summary will appear after the incident evidence is finalized.'
            }
            accent="border-blue-900/50 bg-blue-950/10"
          />
        </div>
      </div>

      <div className="pt-2 border-t border-zinc-800 flex items-center justify-between">
        <span className="text-[9px] text-zinc-500">Passing fix present</span>
        <span className={`text-[10px] font-mono font-bold ${chosenRun ? 'text-emerald-400' : 'text-zinc-500'}`}>
          {chosenRun ? 'YES' : 'NO'}
        </span>
      </div>
    </div>
  );
}
