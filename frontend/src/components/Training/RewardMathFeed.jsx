import { useRef, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { useSimulationState } from '../../store/simulationStore';

const AGENT_COLORS = {
  MANAGER:  'text-violet-400',
  SRE:      'text-sky-400',
  SRE_AGENT:'text-sky-400',
  CODER:    'text-amber-400',
  ENGINEER: 'text-amber-400',
  DETECTIVE:'text-orange-400',
};

function agentColor(agent) {
  const key = String(agent || '').toUpperCase().split('_')[0];
  return AGENT_COLORS[agent?.toUpperCase()] || AGENT_COLORS[key] || 'text-zinc-300';
}

export default function RewardMathFeed() {
  const { rewardFeed, totalReward, scenarioComplete } = useSimulationState();
  const scrollRef = useRef(null);

  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [rewardFeed]);

  return (
    <div className="panel-card p-3 flex flex-col h-full min-h-0">
      <div className="flex items-center justify-between mb-3 pb-2 border-b border-zinc-800/50">
        <span className="text-[10px] font-bold text-zinc-400 uppercase tracking-widest flex items-center gap-1.5">
          <span className="w-1.5 h-1.5 rounded-full bg-emerald-500 animate-pulse" />
          Real-Time Reward Feed
        </span>
        <span className={`text-xs font-mono font-bold ${totalReward >= 0 ? 'text-emerald-400' : 'text-red-400'} px-2 py-0.5 rounded bg-zinc-950 border border-zinc-800`}>
          Σ {(totalReward >= 0 ? '+' : '') + totalReward.toFixed(2)}
        </span>
      </div>
      <div
        ref={scrollRef}
        className="flex-1 min-h-0 overflow-y-auto space-y-1.5 pr-1 scrollbar-thin scrollbar-thumb-zinc-700 scrollbar-track-transparent"
      >
        {rewardFeed.length > 0 ? (
          <AnimatePresence initial={false}>
            {rewardFeed.map((entry) => (
              <motion.div
                key={entry.id}
                initial={{ opacity: 0, x: -8 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ duration: 0.15 }}
                className="flex items-start gap-2 py-1.5 border-b border-zinc-900/40 last:border-0 min-w-0"
              >
                <span className="text-zinc-700 shrink-0 select-none mt-0.5">↳</span>
                <div className="flex-1 min-w-0 flex flex-col gap-0.5">
                  <div className="grid grid-cols-[minmax(0,1fr)_auto] items-start gap-2">
                    <span className="min-w-0 text-[11px] font-mono uppercase tracking-wide leading-snug">
                      <span className={`font-semibold ${agentColor(entry.agent)}`}>{entry.agent}</span>
                      <span className="text-zinc-600 font-normal"> ▸ </span>
                      <span className="text-zinc-400 font-normal break-words">{entry.target}</span>
                    </span>
                    <span className={`text-[12px] font-mono font-bold shrink-0 ${entry.value >= 0 ? 'text-emerald-400' : 'text-red-400'}`}>
                      {(entry.value >= 0 ? '+' : '') + entry.value.toFixed(2)}
                    </span>
                  </div>
                  <span className="text-zinc-600 text-[9px] font-mono">{entry.timestamp}</span>
                </div>
              </motion.div>
            ))}
            {scenarioComplete && (
              <motion.div
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                className="mt-4 pt-3 border-t border-zinc-800 bg-emerald-950/5 p-2 rounded"
              >
                <span className="text-emerald-500 font-bold text-xs">
                  [SUCCESS] Incident Closed. Total Σ {(totalReward >= 0 ? '+' : '') + totalReward.toFixed(2)}
                </span>
              </motion.div>
            )}
          </AnimatePresence>
        ) : (
          <div className="h-full w-full flex items-center justify-center">
            <div className="text-center opacity-40">
              <div className="text-zinc-500 text-xs mb-1 font-mono">{'>'} Σ +0.00</div>
              <p className="text-zinc-600 text-[10px] font-mono">Awaiting real reward events...</p>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
