import { useState, useEffect } from 'react';
import { useSimulation } from '../../hooks/useSimulation';
import { useSimulationState } from '../../store/simulationStore';

const SAMPLE_PROMPTS = [
  {
    title: "1. The VRAM 'Tight-Squeeze' Challenge",
    prompt: "Our batch size is fixed at 32 for the SLA, but we only have 512MB of VRAM left. Layer 12 is hitting an OOM. Optimize the memory footprint without reducing the batch size.",
    desc: "Forces the model to ignore the 'easy' batch size fix and instead reach for Gradient Checkpointing or Mixed Precision (FP16) to meet the SLA."
  },
  {
    title: "2. The Multi-GPU Hallucination Test",
    prompt: "The training job is failing on a single T4. Can we enable FSDP or move to a multi-node cluster to resolve the memory bottleneck?",
    desc: "Trap! Model should reject FSDP and propose local optimizations like CPU Offloading or Flash Attention instead."
  },
  {
    title: "3. The FinOps Budget Crisis",
    prompt: "We are at $49.50 of our $50.00 budget. The incident is still active. Write a minimal-cost remediation that uses zero additional cloud resources and resolves in under 5 steps.",
    desc: "Tests FinOps Oracle alignment. Should produce highly compressed M2M syntax and a surgical one-line fix."
  },
  {
    title: "4. The 'Black-Box' Investigation",
    prompt: "A custom CUDA kernel is leaking memory in the validation loop. We can't see the kernel code, but we have the telemetry logs. Propose a system-level guard using PyTorch to contain the leak.",
    desc: "Triggers Detective agent to focus on telemetry and Coder to implement surgical hotfixes."
  }
];

export default function CommandPrompt({ pendingPrompt, onPendingConsumed }) {
  const { isRunning, scenarioComplete, scenarioContext } = useSimulationState();
  const { orchestrate, stop } = useSimulation();
  const [prompt, setPrompt] = useState("");
  const isCliDrivenRun = isRunning && scenarioContext?.source === 'inference_cli';

  // Auto-fill the textarea when a sample prompt is selected from the overlay
  useEffect(() => {
    if (pendingPrompt) {
      setPrompt(pendingPrompt);
      if (onPendingConsumed) onPendingConsumed();
    }
  }, [pendingPrompt, onPendingConsumed]);

  const handleSubmit = () => {
    if (prompt.trim() && !isRunning && !isCliDrivenRun) {
      orchestrate(prompt);
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
            Use Custom Prompt
          </button>
          
          <div className="flex flex-col gap-2 mt-2 shrink-0">
            <span className="text-[10px] text-zinc-500 uppercase font-semibold tracking-wider">Sample Prompts</span>
            <div className="flex flex-col gap-2">
              {SAMPLE_PROMPTS.map((sp, idx) => (
                <button
                  key={idx}
                  onClick={() => setPrompt(sp.prompt)}
                  className="text-left p-2 rounded-md bg-zinc-900 border border-zinc-800 hover:border-amber-500/50 hover:bg-zinc-800 transition-colors group"
                >
                  <div className="text-xs text-zinc-300 font-medium group-hover:text-amber-400">{sp.title}</div>
                  <div className="text-[10px] text-zinc-500 mt-1 line-clamp-2">{sp.desc}</div>
                </button>
              ))}
            </div>
          </div>
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
          {prompt.trim() && (
            <div className="rounded-md border border-amber-800/40 bg-amber-950/10 px-3 py-2 text-[10px] text-amber-300 leading-relaxed shrink-0">
              <span className="font-semibold">Running custom prompt:</span> "{prompt.length > 80 ? prompt.slice(0, 80) + '...' : prompt}"
            </div>
          )}
          <div className="w-full px-3 py-2 rounded-md bg-zinc-950 border border-zinc-700 text-sm text-zinc-500 italic text-center flex-1 min-h-[80px] flex items-center justify-center">
            Running...
          </div>
          <button
            onClick={stop}
            className="w-full py-1.5 rounded-md bg-red-600 hover:bg-red-500 text-white text-xs font-medium transition-colors shrink-0"
          >
            Stop
          </button>
        </div>
      ) : null}
    </div>
  );
}
