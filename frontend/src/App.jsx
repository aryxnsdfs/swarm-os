import { useState } from "react";
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

export default function App() {
  const [activeTab, setActiveTab] = useState("live");
  const { isRunning, scenarioComplete } = useSimulationState();
  const { orchestrate } = useSimulation();
  const showPrompt = !isRunning && !scenarioComplete;
  const [overlayDismissed, setOverlayDismissed] = useState(false);

  const showOverlay = !isRunning && !overlayDismissed;

  const handleStartSimulation = () => {
    setOverlayDismissed(true);
    orchestrate(DEFAULT_PROMPT);
  };

  const handleClearReset = () => {
    setOverlayDismissed(false);
  };

  // Direct route detection for isolated dashboard views
  const isTrainingPage = window.location.pathname === "/training-proof";

  const TrainingGrid = () => (
    <div className="h-full w-full overflow-y-auto overflow-x-hidden pr-1 scrollbar-thin scrollbar-thumb-zinc-700 scrollbar-track-transparent">
      <div
        className="min-h-[720px] p-2 gap-3 grid bg-zinc-950 text-zinc-300"
        style={{
          gridTemplateColumns: "minmax(0, 1fr) 380px",
          gridTemplateRows: "390px 300px",
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

      {/* Full-screen overlay — Start Simulation */}
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
              initial={{ scale: 0.85, opacity: 0 }}
              animate={{ scale: 1, opacity: 1 }}
              exit={{ scale: 0.85, opacity: 0 }}
              transition={{ duration: 0.35, ease: "easeOut" }}
              className="relative z-10 flex flex-col items-center gap-6 p-10 rounded-2xl border border-zinc-700/50 bg-zinc-900/80 shadow-2xl max-w-lg"
            >
              <div className="w-16 h-16 rounded-2xl bg-gradient-to-br from-blue-500 to-purple-600 flex items-center justify-center shadow-lg shadow-blue-500/20">
                <span className="text-white text-2xl font-bold">F</span>
              </div>
              <div className="text-center">
                <h2 className="text-2xl font-bold text-zinc-100 mb-2">FrontierLabs Swarm-OS</h2>
                <p className="text-sm text-zinc-400 leading-relaxed max-w-sm">
                  Multi-agent incident response with GRPO-trained Llama-3.
                  Click below to launch a live OpenEnv simulation.
                </p>
              </div>
              <button
                onClick={handleStartSimulation}
                className="px-8 py-3.5 rounded-xl bg-gradient-to-r from-emerald-600 to-emerald-500 hover:from-emerald-500 hover:to-emerald-400 text-white text-base font-semibold shadow-lg shadow-emerald-500/25 transition-all hover:shadow-emerald-500/40 hover:scale-[1.03] active:scale-[0.98]"
              >
                Start Simulation
              </button>
              <span className="text-[10px] text-zinc-500 font-mono">inference.py &middot; OpenEnv &middot; Docker Sandbox</span>
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
              className="h-full p-2 flex gap-2"
            >
              <div className="w-[280px] xl:w-[320px] shrink-0 flex flex-col gap-2 overflow-hidden">
                <div className="flex-1 min-h-0 overflow-y-auto scrollbar-thin scrollbar-thumb-zinc-700 scrollbar-track-transparent">
                  <DockerPhysicsMonitor />
                </div>
                {showPrompt && (
                  <div className="shrink-0 h-[200px] flex flex-col">
                    <CommandPrompt />
                  </div>
                )}
              </div>

              <div className="flex-1 flex flex-col gap-2 min-w-0 overflow-hidden">
                <div className="flex-1 min-h-0">
                  <EnterpriseChat />
                </div>
                <div className="h-[250px] shrink-0 overflow-hidden">
                  <CausalDAG />
                </div>
              </div>

              {/* Consolidated Right Column with Single Scrollbar */}
              <div className="w-[380px] xl:w-[430px] shrink-0 flex flex-col gap-2 overflow-y-auto scrollbar-thin scrollbar-thumb-zinc-700 scrollbar-track-transparent pr-1">
                <div className="shrink-0">
                  <GitRCAPanel />
                </div>
                <div className="shrink-0">
                  <DeadTimeline />
                </div>
              </div>
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
