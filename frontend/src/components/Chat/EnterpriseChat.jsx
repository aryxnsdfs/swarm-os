import { useRef, useEffect, useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { useSimulationDispatch, useSimulationState } from '../../store/simulationStore';
import TranslateToggle from './TranslateToggle';
import { ShieldAlert, Bot, Terminal, Briefcase, Users, Gauge } from 'lucide-react';

export default function EnterpriseChat() {
  const dispatch = useSimulationDispatch();
  const { messages, disagreement, scenarioContext, taskViews, selectedTaskView, scenarioComplete, spent, budget, burnRate, rewardFeed, totalReward } = useSimulationState();
  const scrollRef = useRef(null);
  const [translateMode, setTranslateMode] = useState(false);
  const [expandedThink, setExpandedThink] = useState({});

  const activeTask = scenarioContext?.task_id;
  const availableTasks = new Set([
    ...Object.keys(taskViews || {}),
    ...(activeTask ? [activeTask] : []),
  ]);
  const taskButtons = [
    { key: 'task_easy_gpu_oom', label: 'Easy' },
    { key: 'task_medium_schema_drift', label: 'Medium' },
    { key: 'task_hard_canary_regression', label: 'Hard' },
  ];

  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [messages]);

  const toggleThink = (id) => {
    setExpandedThink((prev) => ({ ...prev, [id]: !prev[id] }));
  };

  // Map agent id to Lucide icon component
  const getAgentIcon = (agentId, color) => {
    const iconProps = { size: 12, color };
    switch (agentId) {
      case 'commander': return <ShieldAlert {...iconProps} />;
      case 'detective': return <Bot {...iconProps} />;
      case 'coder': return <Terminal {...iconProps} />;
      case 'manager': return <Users {...iconProps} />;
      case 'evaluator': return <Gauge {...iconProps} />;
      case 'db_admin': return <Briefcase {...iconProps} />;
      case 'dba_agent': return <Briefcase {...iconProps} />;
      case 'sre_agent': return <Gauge {...iconProps} />;
      case 'security_agent': return <ShieldAlert {...iconProps} />;
      case 'compliance_agent': return <ShieldAlert {...iconProps} />;
      default: return (
        <span className="text-[8px] font-bold font-mono tracking-tight" style={{ color }}>
          {agentId?.substring(0, 3).toUpperCase() || '?'}
        </span>
      );
    }
  };

  return (
    <div className="flex flex-col h-full panel-card overflow-hidden">
      {/* Header */}
      <div className="flex items-center justify-between px-3 py-2 border-b border-zinc-800">
        <div className="flex items-center gap-2">
          <svg className="w-3.5 h-3.5 text-zinc-400" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.5">
            <path d="M2 4h12v8a1 1 0 0 1-1 1H3a1 1 0 0 1-1-1V4z" />
            <path d="M2 4l6 4 6-4" />
          </svg>
          <span className="text-xs font-semibold text-zinc-300">AI Chat</span>
          <span className="text-[10px] px-1.5 py-0.5 rounded bg-zinc-800 text-zinc-500 font-mono">
            {messages.length} msgs
          </span>
        </div>
        <div className="flex items-center gap-2">
          <div className="flex items-center gap-1 rounded-md border border-zinc-800 bg-zinc-900/70 p-1">
            {taskButtons.map((button) => {
              const isSelected = selectedTaskView === button.key;
              const isEnabled = availableTasks.has(button.key);
              return (
                <button
                  key={button.key}
                  type="button"
                  disabled={!isEnabled}
                  onClick={() => isEnabled && dispatch({ type: 'SELECT_TASK_VIEW', payload: button.key })}
                  className={`px-2 py-1 rounded text-[10px] font-mono transition-colors ${
                    isSelected
                      ? 'bg-emerald-950/60 text-emerald-300 border border-emerald-500/30'
                      : isEnabled
                      ? 'bg-zinc-800 text-zinc-300 border border-zinc-700 hover:bg-zinc-700'
                      : 'text-zinc-600 cursor-not-allowed'
                  }`}
                >
                  {button.label}
                </button>
              );
            })}
          </div>
          <TranslateToggle enabled={translateMode} onToggle={setTranslateMode} />
        </div>
      </div>

      {/* Disagreement Banner */}
      <AnimatePresence>
        {disagreement.active && (
          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: 'auto', opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            className="bg-amber-900/20 border-b border-amber-800/50 px-3 py-2 overflow-hidden"
          >
            <div className="flex items-center gap-2 mb-1">
              <svg className="w-3.5 h-3.5 text-amber-400" viewBox="0 0 16 16" fill="currentColor">
                <path d="M8 1l7 14H1L8 1zm0 4v4m0 2v1" stroke="#18181b" strokeWidth="1.2" fill="none" />
                <path d="M8 1l7 14H1L8 1z" fillOpacity="0.2" />
              </svg>
              <span className="text-amber-400 text-xs font-bold">REASONING FORK</span>
            </div>
            <div className="grid grid-cols-2 gap-2 text-[10px]">
              <div className="bg-zinc-900/80 rounded p-2 border border-amber-800/30">
                <span className="text-blue-400 font-mono">{disagreement.position1?.agent}</span>
                <p className="text-zinc-400 mt-0.5">{disagreement.position1?.action}</p>
                <p className="text-red-400 font-mono mt-0.5">{disagreement.position1?.cost}</p>
              </div>
              <div className="bg-zinc-900/80 rounded p-2 border border-amber-800/30">
                <span className="text-amber-400 font-mono">{disagreement.position2?.agent}</span>
                <p className="text-zinc-400 mt-0.5">{disagreement.position2?.action}</p>
                <p className="text-emerald-400 font-mono mt-0.5">{disagreement.position2?.cost}</p>
              </div>
            </div>
            {disagreement.resolution && (
              <p className="text-emerald-400 text-[10px] mt-1 font-mono">
                <span className="text-emerald-500 mr-1">[RESOLVED]</span>
                {disagreement.resolution}
              </p>
            )}
          </motion.div>
        )}
      </AnimatePresence>

      {/* Messages */}
      <div ref={scrollRef} className="flex-1 overflow-y-auto p-3 space-y-2">
        <AnimatePresence initial={false}>
          {messages.map((msg) => (
            <motion.div
              key={msg.id}
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.2 }}
              className="group"
            >
              <div className="flex items-start gap-2">
                {/* Agent Avatar — Professional badge */}
                <div
                  className="w-6 h-6 rounded flex items-center justify-center shrink-0 mt-0.5"
                  style={{ backgroundColor: (msg.agent.color || '#71717a') + '18', border: `1px solid ${msg.agent.color || '#71717a'}35` }}
                >
                  {getAgentIcon(msg.agent.id, msg.agent.color || '#71717a')}
                </div>

                <div className="flex-1 min-w-0">
                  {/* Agent Name + Timestamp */}
                  <div className="flex items-center gap-2 mb-0.5">
                    <span className="text-[11px] font-semibold" style={{ color: msg.agent.color || '#71717a' }}>
                      {msg.agent.name || msg.agent.id || 'Agent'}
                    </span>
                    <span className="text-[9px] text-zinc-600 font-mono">{msg.timestamp}</span>
                    {msg.think && (
                      <button
                        onClick={() => toggleThink(msg.id)}
                        className="text-[9px] px-1.5 py-0.5 rounded bg-zinc-800 text-zinc-500 hover:text-zinc-300 hover:bg-zinc-700 transition-colors font-mono"
                      >
                        {expandedThink[msg.id] ? 'HIDE COT' : 'DEBUG'}
                      </button>
                    )}
                  </div>

                  {/* Message Content */}
                  <AnimatePresence mode="wait">
                    <motion.div
                      key={translateMode ? 'english' : 'm2m'}
                      initial={{ opacity: 0 }}
                      animate={{ opacity: 1 }}
                      exit={{ opacity: 0 }}
                      transition={{ duration: 0.15 }}
                    >
                      {translateMode ? (
                        <div className="flex flex-wrap items-center gap-2">
                           <p className="text-xs text-zinc-300 leading-relaxed">{(msg.english || '').replace(/\*\*/g, '').replace(/\*/g, '')}</p>
                           {msg.points !== undefined && msg.points !== 0 && (
                             <span className={`text-[10px] font-mono px-1.5 py-0.5 rounded ${msg.points > 0 ? 'bg-emerald-900/30 text-emerald-400' : 'bg-red-900/30 text-red-500'}`}>
                                {msg.points > 0 ? '+' : ''}{msg.points.toFixed(2)} pts
                             </span>
                           )}
                        </div>
                      ) : (
                        <div className="flex flex-wrap items-center gap-2 mt-0.5">
                           <p className="text-xs font-mono text-emerald-400 bg-zinc-900/50 px-2 py-1 rounded inline-block">
                             {(msg.m2m || '').replace(/\*\*/g, '').replace(/\*/g, '')}
                           </p>
                           {msg.points !== undefined && msg.points !== 0 && (
                             <span className={`text-[10px] font-mono px-1.5 py-0.5 rounded ${msg.points > 0 ? 'bg-emerald-500/20 text-emerald-400 border border-emerald-500/30' : 'bg-red-500/20 text-red-500 border border-red-500/30'}`}>
                                {msg.points > 0 ? '+' : ''}{msg.points.toFixed(2)} pts
                             </span>
                           )}
                        </div>
                      )}
                      <div className="flex items-center gap-3 mt-1.5 pt-1 border-t border-zinc-800/40 text-[9px] font-mono text-zinc-600">
                        <span>Budget Left: <span className="text-zinc-400">${(Number(budget || 50) - Number(spent || 0)).toFixed(3)}</span> <span className="text-emerald-600/70">(healthy)</span></span>
                        <span>Cost Accrued: <span className="text-zinc-400">${Number(spent || 0).toFixed(3)}</span></span>
                        <span>Burn Rate: <span className="text-zinc-400">${Number(burnRate || 2.5).toFixed(3)}/hr</span></span>
                      </div>
                    </motion.div>
                  </AnimatePresence>

                  {/* Hidden CoT Block */}
                  <AnimatePresence>
                    {msg.think && expandedThink[msg.id] && (
                      <motion.div
                        initial={{ height: 0, opacity: 0 }}
                        animate={{ height: 'auto', opacity: 1 }}
                        exit={{ height: 0, opacity: 0 }}
                        transition={{ duration: 0.2 }}
                        className="overflow-hidden"
                      >
                        <div className="mt-1.5 p-2 rounded bg-purple-900/10 border border-purple-800/20">
                          <span className="text-[9px] text-purple-400 font-mono block mb-1">{'<think>'}</span>
                          <p className="text-[10px] text-zinc-400 leading-relaxed italic">{(msg.think || '').replace(/\*\*/g, '').replace(/\*/g, '')}</p>
                          <span className="text-[9px] text-purple-400 font-mono block mt-1">{'</think>'}</span>
                        </div>
                      </motion.div>
                    )}
                  </AnimatePresence>
                </div>
              </div>
            </motion.div>
          ))}
        </AnimatePresence>

        {scenarioComplete && messages.length > 0 && (
          <motion.div
            initial={{ opacity: 0, y: 8 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.3, delay: 0.1 }}
            className="mt-3 rounded-md border border-zinc-700 bg-zinc-900/80 font-mono text-[10px]"
          >
            <div className="px-3 py-2 border-b border-zinc-800 text-center text-zinc-500 tracking-widest text-[9px]">
              {'═'.repeat(40)} INCIDENT SUMMARY {'═'.repeat(40)}
            </div>
            <div className="px-4 py-2.5 space-y-1">
              <div className="flex justify-between">
                <span className="text-zinc-500">Success</span>
                <span className="text-emerald-400 font-bold">true</span>
              </div>
              <div className="flex justify-between">
                <span className="text-zinc-500">Steps</span>
                <span className="text-zinc-300">{messages.length}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-zinc-500">Final Score</span>
                <span className="text-emerald-400 font-bold">{totalReward.toFixed(3)}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-zinc-500">Rewards</span>
                <span className="text-zinc-300">{rewardFeed.map((r) => r.value.toFixed(2)).join(', ') || '—'}</span>
              </div>
            </div>
            <div className="px-3 py-1.5 border-t border-zinc-800 text-center text-zinc-600 tracking-widest text-[9px]">
              {'═'.repeat(90)}
            </div>
          </motion.div>
        )}

        {messages.length === 0 && (
          <div className="flex flex-col items-center justify-center h-full text-zinc-600">
            <svg className="w-8 h-8 text-zinc-700 mb-2" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
              <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z" />
            </svg>
            <p className="text-xs">Awaiting scenario start...</p>
          </div>
        )}
      </div>

      {/* Global FinOps Summary — always-visible footer below chat */}
      {messages.length > 0 && (() => {
        const incidentCount = Object.keys(taskViews || {}).length || 1;
        const aiCost = Number(spent || 0);
        const humanCost = incidentCount * 79.50;
        const saved = humanCost - aiCost;
        const budgetLeft = Number(budget || 50) - aiCost;
        const isComplete = scenarioComplete;
        return (
          <div className={`shrink-0 border-t px-4 py-2.5 ${isComplete ? 'border-emerald-500/30 bg-emerald-500/5' : 'border-zinc-800 bg-zinc-900/50'}`}>
            <div className="flex items-center gap-2 mb-1.5">
              <span className={`w-1.5 h-1.5 rounded-full ${isComplete ? 'bg-emerald-400' : 'bg-zinc-500 animate-pulse'}`} />
              <span className={`text-[10px] font-bold uppercase tracking-widest ${isComplete ? 'text-emerald-300' : 'text-zinc-400'}`}>
                {isComplete ? 'Global FinOps Summary' : 'Live FinOps Tracker'}
              </span>
              {isComplete && <span className="text-[9px] font-mono text-emerald-600 ml-auto">[ SUCCESS ]</span>}
            </div>
            <div className="flex items-center gap-4 text-[10px] font-mono flex-wrap">
              <span className="text-zinc-500">Incidents: <span className={isComplete ? 'text-emerald-300 font-bold' : 'text-zinc-300'}>{incidentCount}</span></span>
              <span className="text-zinc-500">Human Cost: <span className="text-red-400 font-bold">${humanCost.toFixed(2)}</span></span>
              <span className="text-zinc-500">AI Cost: <span className={isComplete ? 'text-emerald-300 font-bold' : 'text-zinc-300'}>${aiCost.toFixed(3)}</span></span>
              <span className="text-zinc-500">Saved: <span className="text-emerald-400 font-bold">${saved.toFixed(2)}</span></span>
              <span className="text-zinc-500">Budget: <span className="text-zinc-300">${budgetLeft.toFixed(3)}</span></span>
            </div>
          </div>
        );
      })()}
    </div>
  );
}
