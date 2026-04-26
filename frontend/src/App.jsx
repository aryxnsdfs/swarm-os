import { useEffect, useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import Header from "./components/Layout/Header";
import TabBar from "./components/Layout/TabBar";
import EnterpriseChat from "./components/Chat/EnterpriseChat";
import DockerPhysicsMonitor from "./components/Monitor/DockerPhysicsMonitor";
import CommandPrompt from "./components/Chat/CommandPrompt";
import CausalDAG from "./components/CausalGraph/CausalDAG";
import DeadTimeline from "./components/Counterfactual/DeadTimeline";
import GitRCAPanel from "./components/GitPanel/GitRCAPanel";
import RewardCurve from "./components/Training/RewardCurve";
import BeforeAfterSplit from "./components/Training/BeforeAfterSplit";
import RewardMathFeed from "./components/Training/RewardMathFeed";
import FinOpsPreFlightAudit from "./components/Training/FinOpsPreFlightAudit";
import { useSimulationState } from "./store/simulationStore";
import { useSimulation } from "./hooks/useSimulation";

const DEFAULT_PROMPT = "Deploy Llama-3 on a 4-node GPU cluster with a strict 500MB VRAM constraint and a $50 budget ceiling. An OOM crash is active on Node-2.";

const SAMPLE_PROMPT_DATA = [
  {
    title: '1. The VRAM "Tight-Squeeze" Challenge',
    prompt: "Our batch size is fixed at 32 for the SLA, but we only have 512MB of VRAM left. Layer 12 is hitting an OOM. Optimize the memory footprint without reducing the batch size.",
  },
  {
    title: "2. The Multi-GPU Hallucination Test",
    prompt: "The training job is failing on a single T4. Can we enable FSDP or move to a multi-node cluster to resolve the memory bottleneck?",
  },
  {
    title: "3. The FinOps Budget Crisis",
    prompt: "We are at $49.50 of our $50.00 budget. The incident is still active. Write a minimal-cost remediation that uses zero additional cloud resources and resolves in under 5 steps.",
  },
  {
    title: '4. The "Black-Box" Investigation',
    prompt: "A custom CUDA kernel is leaking memory in the validation loop. We can't see the kernel code, but we have the telemetry logs. Propose a system-level guard using PyTorch to contain the leak.",
  },
];

function FinOpsSummaryBar() {
  const { messages, scenarioComplete, spent, budget, taskViews, totalReward, rewardFeed } = useSimulationState();
  if (messages.length === 0) return null;

  const incidentCount = Object.keys(taskViews || {}).length || 1;
  const aiCost = Number(spent || 0);
  const humanCost = incidentCount * 79.50;
  const saved = humanCost - aiCost;
  const budgetLeft = Number(budget || 50) - aiCost;
  const isComplete = scenarioComplete;
  const steps = rewardFeed.length;

  return (
    <div className={`shrink-0 rounded-lg border px-4 py-2.5 ${isComplete ? 'border-emerald-500/30 bg-emerald-500/5' : 'border-zinc-800 bg-zinc-900/50'}`}>
      <div className="flex items-center gap-3 mb-1.5">
        <span className={`w-1.5 h-1.5 rounded-full ${isComplete ? 'bg-emerald-400' : 'bg-zinc-500 animate-pulse'}`} />
        <span className={`text-[10px] font-bold uppercase tracking-widest ${isComplete ? 'text-emerald-300' : 'text-zinc-400'}`}>
          {isComplete ? 'Global FinOps Summary' : 'Live FinOps Tracker'}
        </span>
        {isComplete && <span className="text-[9px] font-mono text-emerald-600">[ SUCCESS ]</span>}
      </div>
      <div className="flex items-center gap-5 text-[10px] font-mono flex-wrap">
        <span className="text-zinc-500">Incidents: <span className={isComplete ? 'text-emerald-300 font-bold' : 'text-zinc-300'}>{incidentCount}</span></span>
        <span className="text-zinc-500">Steps: <span className="text-zinc-300">{steps}</span></span>
        <span className="text-zinc-500">Human Cost: <span className="text-red-400 font-bold">${humanCost.toFixed(2)}</span></span>
        <span className="text-zinc-500">AI Cost: <span className={isComplete ? 'text-emerald-300 font-bold' : 'text-zinc-300'}>${aiCost.toFixed(3)}</span></span>
        <span className="text-zinc-500">Saved: <span className="text-emerald-400 font-bold">${saved.toFixed(2)}</span></span>
        <span className="text-zinc-500">Budget Left: <span className="text-zinc-300">${budgetLeft.toFixed(3)}</span></span>
        <span className="text-zinc-500">Total Reward: <span className={`font-bold ${totalReward >= 0 ? 'text-emerald-400' : 'text-red-400'}`}>{(totalReward >= 0 ? '+' : '')}{totalReward.toFixed(2)}</span></span>
      </div>
    </div>
  );
}

export default function App() {
  const [activeTab, setActiveTab] = useState("live");
  const { isRunning, scenarioComplete } = useSimulationState();
  const { orchestrate } = useSimulation();
  const [overlayDismissed, setOverlayDismissed] = useState(false);
  const [pendingPrompt, setPendingPrompt] = useState("");

  const showOverlay = !isRunning && !scenarioComplete && !overlayDismissed;

  const handleRunInference = () => {
    setOverlayDismissed(true);
    orchestrate(DEFAULT_PROMPT);
  };

  const handleClearReset = () => {
    setOverlayDismissed(false);
  };

  useEffect(() => {
    const resetOverlay = () => setOverlayDismissed(false);
    window.addEventListener("swarm-os:sleep-reset", resetOverlay);
    return () => window.removeEventListener("swarm-os:sleep-reset", resetOverlay);
  }, []);

  // Direct route detection for isolated dashboard views
  const isTrainingPage = window.location.pathname === "/training-proof";

  const TrainingGrid = () => (
    <div className="h-full w-full overflow-y-auto overflow-x-hidden pr-1 scrollbar-thin scrollbar-thumb-zinc-700 scrollbar-track-transparent">
      <div
        className="h-full min-h-[560px] p-2 gap-3 grid bg-zinc-950 text-zinc-300"
        style={{
          gridTemplateColumns: "minmax(0, 1.35fr) minmax(280px, 0.75fr)",
          gridTemplateRows: "minmax(240px, 0.9fr) minmax(260px, 1fr)",
        }}
      >
        {/* Top-left: Reward trace */}
        <div className="min-w-0 min-h-0">
          <RewardCurve />
        </div>

        {/* Top-right: Reward feed */}
        <div className="min-w-0 min-h-0">
          <RewardMathFeed />
        </div>

        {/* Bottom-left: Incident phases */}
        <div className="min-w-0 min-h-0">
          <BeforeAfterSplit />
        </div>

        {/* Bottom-right: FinOps gatekeeper */}
        <div className="min-w-0 min-h-0">
          <FinOpsPreFlightAudit />
        </div>
      </div>
    </div>
  );

  if (isTrainingPage) {
    return (
      <div className="h-screen w-screen overflow-hidden">
        <TrainingGrid />
      </div>
    );
  }

  return (
    <div className="flex flex-col h-screen bg-zinc-950 text-zinc-300 relative">
      <Header onClearReset={handleClearReset} />
      <TabBar activeTab={activeTab} onTabChange={setActiveTab} />

      {/* Full-screen overlay — Start inference.py */}
      <AnimatePresence>
        {showOverlay && (
          <motion.div
            key="start-overlay"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            transition={{ duration: 0.3 }}
            className="absolute inset-0 z-50 flex items-center justify-center"
            style={{ backdropFilter: "blur(12px)", WebkitBackdropFilter: "blur(12px)" }}
          >
            <div className="absolute inset-0 bg-black/60" />
            <motion.div
              initial={{ scale: 0.9, opacity: 0 }}
              animate={{ scale: 1, opacity: 1 }}
              exit={{ scale: 0.9, opacity: 0 }}
              transition={{ duration: 0.3, ease: "easeOut" }}
              className="relative z-10 flex flex-col items-center gap-4 px-8 py-7 rounded-2xl border border-zinc-700/50 bg-zinc-900/85 shadow-2xl w-[520px] max-w-[95vw] max-h-[85vh] overflow-y-auto scrollbar-thin scrollbar-thumb-zinc-700 scrollbar-track-transparent"
            >


              {/* Story blurb */}
              <div className="w-full rounded-lg bg-zinc-800/50 border border-zinc-700/40 px-4 py-3 text-left space-y-1.5">
                <p className="text-[11px] text-zinc-300 leading-relaxed">
                  When a GPU crashes at 3 AM, a human SRE takes 6 minutes to triage, 4 minutes to draft a fix, and another 3 minutes to validate it — costing $238 in operational burn before the incident closes. <span className="text-emerald-400 font-semibold">Swarm-OS resolves the same incident in under 10 seconds for $0.003.</span>
                </p>
                <p className="text-[11px] text-zinc-500 leading-relaxed">
                  A multi-agent swarm — Commander, Detective, Coder, SRE — trained with GRPO reinforcement learning on a real OpenEnv simulation. The model runs locally inside this Space: no cloud API, no data leaving the container.
                </p>
              </div>

              {/* Task list */}
              <div className="w-full flex flex-col gap-1.5 text-left">
                <p className="text-[9px] text-zinc-500 uppercase tracking-widest font-semibold mb-0.5">Three live OpenEnv incidents</p>
                <div className="flex items-start gap-2.5 px-3 py-2 rounded-lg bg-emerald-500/10 border border-emerald-500/20">
                  <span className="px-1.5 py-0.5 rounded text-[9px] font-bold bg-emerald-500/20 text-emerald-400 shrink-0 mt-0.5">EASY</span>
                  <div>
                    <span className="text-[11px] font-semibold text-emerald-300">Single-GPU OOM Triage</span>
                    <p className="text-[9px] text-zinc-500 leading-tight mt-0.5">CUDA OOM on A10 GPU — audit artifacts, propose memory-safe fix, pass Docker GPU validator</p>
                  </div>
                </div>
                <div className="flex items-start gap-2.5 px-3 py-2 rounded-lg bg-amber-500/10 border border-amber-500/20">
                  <span className="px-1.5 py-0.5 rounded text-[9px] font-bold bg-amber-500/20 text-amber-400 shrink-0 mt-0.5">MED</span>
                  <div>
                    <span className="text-[11px] font-semibold text-amber-300">Analytics Schema Drift</span>
                    <p className="text-[9px] text-zinc-500 leading-tight mt-0.5">Broken dashboards from upstream API rename — map field drift, backfill, validate schema</p>
                  </div>
                </div>
                <div className="flex items-start gap-2.5 px-3 py-2 rounded-lg bg-red-500/10 border border-red-500/20">
                  <span className="px-1.5 py-0.5 rounded text-[9px] font-bold bg-red-500/20 text-red-400 shrink-0 mt-0.5">HARD</span>
                  <div>
                    <span className="text-[11px] font-semibold text-red-300">Canary Rollout Regression</span>
                    <p className="text-[9px] text-zinc-500 leading-tight mt-0.5">p95 latency spike from canary deploy — rollback, communicate stakeholders, close SLA</p>
                  </div>
                </div>
              </div>

              <div className="flex gap-3 w-full">
                <button
                  onClick={handleRunInference}
                  className="flex-[1.6] py-3 rounded-xl bg-gradient-to-r from-emerald-600 to-emerald-500 hover:from-emerald-500 hover:to-emerald-400 text-white text-sm font-semibold shadow-lg shadow-emerald-500/25 transition-all hover:shadow-emerald-500/40 hover:scale-[1.01] active:scale-[0.99]"
                >
                  Start inference.py
                </button>
                
                <button
                  onClick={() => setOverlayDismissed(true)}
                  className="flex-1 py-3 rounded-xl bg-zinc-800 hover:bg-zinc-700 border border-zinc-700 text-zinc-300 text-[10px] font-semibold transition-all"
                >
                  Use Custom Prompt
                </button>
              </div>

              {/* Sample Custom Prompts */}
              <div className="w-full flex flex-col gap-1.5 text-left mt-2">
                <p className="text-[9px] text-zinc-500 uppercase tracking-widest font-semibold mb-0.5">Sample Custom Prompts</p>
                <div className="flex flex-col gap-2 max-h-[140px] overflow-y-auto pr-1 scrollbar-thin scrollbar-thumb-zinc-700 scrollbar-track-transparent">
                  {SAMPLE_PROMPT_DATA.map((sp, idx) => (
                    <div
                      key={idx}
                      className="px-3 py-2 rounded-lg bg-zinc-800/40 border border-zinc-700/50 cursor-pointer hover:border-amber-500/50 transition-colors shrink-0"
                      onClick={() => { setPendingPrompt(sp.prompt); setOverlayDismissed(true); }}
                    >
                      <span className="text-[10px] font-semibold text-zinc-300">{sp.title}</span>
                      <p className="text-[9px] text-zinc-500 leading-tight mt-0.5">"{sp.prompt.length > 100 ? sp.prompt.slice(0, 100) + '...' : sp.prompt}"</p>
                    </div>
                  ))}
                </div>
              </div>
              <p className="text-[9px] text-zinc-600 text-center">All three incidents run sequentially · Results shown live · Logs in HF Space → Logs tab</p>
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>

      <div className="flex-1 min-h-0 overflow-hidden">
        <AnimatePresence mode="wait">
          {activeTab === "live" ? (
            <motion.div
              key="live"
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
              exit={{ opacity: 0, x: 20 }}
              transition={{ duration: 0.2 }}
              className="h-full p-2 flex flex-col gap-2"
            >
              <div className="flex-1 min-h-0 flex gap-2 overflow-hidden">
                <div className="w-[280px] xl:w-[320px] shrink-0 flex flex-col gap-2 min-h-0">
                  <div className="flex-1 min-h-0 overflow-y-auto overscroll-contain scrollbar-thin scrollbar-thumb-zinc-700 scrollbar-track-transparent">
                    <DockerPhysicsMonitor />
                  </div>
                  {!scenarioComplete && (
                    <div className="shrink-0 h-[200px] flex flex-col">
                      <CommandPrompt pendingPrompt={pendingPrompt} onPendingConsumed={() => setPendingPrompt("")} />
                    </div>
                  )}
                </div>

                <div className="flex-1 flex flex-col gap-2 min-w-0 min-h-0 overflow-hidden">
                  <div className="flex-[3] min-h-0 overflow-hidden">
                    <EnterpriseChat />
                  </div>
                  <div className="flex-[1] min-h-[180px] max-h-[250px] shrink-0 overflow-hidden">
                    <CausalDAG />
                  </div>
                </div>

                <div className="w-[380px] xl:w-[430px] shrink-0 flex flex-col gap-2 min-h-0 overflow-y-auto overscroll-contain scrollbar-thin scrollbar-thumb-zinc-700 scrollbar-track-transparent pr-1">
                  <div className="shrink-0">
                    <GitRCAPanel />
                  </div>
                  <div className="shrink-0">
                    <DeadTimeline />
                  </div>
                </div>
              </div>

              {/* Global FinOps Summary — full-width bar at the bottom */}
              <FinOpsSummaryBar />
            </motion.div>
          ) : (
            <motion.div
              key="training"
              initial={{ opacity: 0, x: 20 }}
              animate={{ opacity: 1, x: 0 }}
              exit={{ opacity: 0, x: -20 }}
              transition={{ duration: 0.2 }}
              className="h-full max-h-full overflow-auto min-h-0"
            >
              <TrainingGrid />
            </motion.div>
          )}
        </AnimatePresence>
      </div>
    </div>
  );
}
