import { useState } from 'react';
import { useSimulation } from '../../hooks/useSimulation';
import { useSimulationState } from '../../store/simulationStore';

export default function CommandPrompt() {
  const { isRunning, scenarioComplete, scenarioContext } = useSimulationState();
  const { orchestrate, stop } = useSimulation();
  const [prompt, setPrompt] = useState("");
  const isCliDrivenRun = isRunning && scenarioContext?.source === 'inference_cli';

  const handleSubmit = () => {
    if (prompt.trim() && !isRunning && !isCliDrivenRun) {
      orchestrate(prompt);
      setPrompt("");
    }
  };

  const handleKeyDown = (e) => {
    // Submit on Enter without Shift. Add a new line on Shift+Enter.
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSubmit();
    }
  };

  return (
    <div className="flex flex-col gap-2 p-3 bg-zinc-900 border border-zinc-800 rounded-lg flex-1 min-h-0 overflow-y-auto scrollbar-thin scrollbar-thumb-zinc-700">
      <div className="text-xs font-semibold tracking-wider text-zinc-500 uppercase flex justify-between shrink-0">
        <span>Prompt Sandbox</span>
      </div>

      {!isRunning ? (
        <div className="flex flex-col gap-2 flex-1 min-h-0">
          <div className="rounded-md border border-zinc-800 bg-zinc-950/70 px-3 py-2 text-[10px] text-zinc-400 leading-relaxed shrink-0">
            This sandbox launches backend-driven OpenEnv runs through `POST /api/orchestrate`. If
            `python inference.py` is driving the dashboard, this prompt stays read-only until that CLI run finishes.
          </div>
          {scenarioComplete && (
            <div className="rounded-md border border-emerald-900/40 bg-emerald-950/10 px-3 py-2 text-[10px] text-emerald-300 leading-relaxed shrink-0">
              Previous scenario completed. Start another live run from this prompt box.
            </div>
          )}
          <textarea
            value={prompt}
            onChange={(e) => setPrompt(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="Type an incident prompt here to drive the frontend live run (e.g. 'Customer dashboards are stale after a schema change. Keep it within a $5 budget.')"
            className="w-full flex-1 min-h-[110px] resize-none px-3 py-2 rounded-md bg-zinc-950 border border-zinc-700 text-sm text-zinc-200 placeholder:text-zinc-500 focus:outline-none focus:border-amber-500/50 focus:shadow-[0_0_10px_rgba(245,158,11,0.15)] transition-all scrollbar-thin scrollbar-thumb-zinc-700"
          />
          <button
            onClick={handleSubmit}
            disabled={!prompt.trim()}
            className="w-full py-1.5 rounded-md bg-emerald-600 hover:bg-emerald-500 disabled:opacity-50 disabled:hover:bg-emerald-600 text-white text-xs font-medium transition-colors shrink-0"
          >
            Start Live Run
          </button>
        </div>
      ) : isCliDrivenRun ? (
        <div className="flex flex-col gap-2 flex-1 min-h-0">
          <div className="w-full px-3 py-2 rounded-md bg-zinc-950 border border-zinc-700 text-sm text-zinc-400 text-center flex-1 min-h-[80px] flex items-center justify-center leading-relaxed">
            <p>
              This live run is being driven by <span className="font-mono">python inference.py</span>.
              Stop that terminal run to re-enable the Prompt Sandbox.
            </p>
          </div>
        </div>
      ) : isRunning ? (
        <div className="flex flex-col gap-2 flex-1 min-h-0">
          <div className="w-full px-3 py-2 rounded-md bg-zinc-950 border border-zinc-700 text-sm text-zinc-500 italic text-center flex-1 min-h-[80px] flex items-center justify-center">
            Simulation is running...
          </div>
          <button
            onClick={stop}
            className="w-full py-1.5 rounded-md bg-red-600 hover:bg-red-500 text-white text-xs font-medium transition-colors shrink-0"
          >
            ■ Stop Simulation
          </button>
        </div>
      ) : null}
    </div>
  );
}
