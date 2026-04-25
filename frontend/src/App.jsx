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



export default function App() {
  const [activeTab, setActiveTab] = useState("live");
  const { isRunning, scenarioComplete } = useSimulationState();
  const showPrompt = !isRunning && !scenarioComplete;

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
    <div className="flex flex-col h-screen bg-zinc-950 text-zinc-300">
      <Header />
      <TabBar activeTab={activeTab} onTabChange={setActiveTab} />

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
