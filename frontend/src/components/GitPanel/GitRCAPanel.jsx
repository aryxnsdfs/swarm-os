import { motion } from 'framer-motion';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import { useSimulationState } from '../../store/simulationStore';

export default function GitRCAPanel() {
  const { rcaDocument, scenarioContext, scenarioComplete, causalNodes, lastValidatorResult } = useSimulationState();

  return (
    <div className="panel-card p-3 flex flex-col min-w-0">
      <div className="flex items-center gap-1.5 mb-3">
        <div className="w-5 h-5 rounded bg-emerald-500/10 border border-emerald-500/20 flex items-center justify-center">
          <svg className="w-3 h-3 text-emerald-400" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.5">
            <rect x="2" y="1.5" width="12" height="13" rx="1.5" />
            <line x1="5" y1="5" x2="11" y2="5" />
            <line x1="5" y1="8" x2="11" y2="8" />
          </svg>
        </div>
        <span className="text-[10px] font-bold text-zinc-300 uppercase tracking-widest">
          Root Cause Analysis
        </span>
        <span className="ml-auto text-[9px] font-mono text-zinc-600 animate-pulse">
          {rcaDocument ? 'live evidence ready' : scenarioComplete ? 'awaiting RCA text' : 'collecting incident evidence'}
        </span>
      </div>

      {!rcaDocument ? (
        <motion.div
          initial={{ opacity: 0.4 }}
          animate={{ opacity: 1 }}
          className="rounded-lg border border-zinc-800 bg-zinc-900/40 px-3 py-3 text-[11px] text-zinc-400 leading-relaxed"
        >
          <div className="mb-4">
            <h3 className="text-zinc-200 font-bold mb-1.5 text-xs flex items-center gap-1.5">
              <span className="w-1 h-3 bg-blue-500 rounded-full" />
              The Trigger
            </h3>
            <p className="text-zinc-300 font-medium mb-1 px-2.5">
              {scenarioContext?.title || 'Waiting for incident run...'}
            </p>
            <p className="px-2.5 text-zinc-500 italic">
              {scenarioContext?.incident_summary || 'Start a backend run or bridge in python inference.py to populate the RCA stream here.'}
            </p>
          </div>
          <div className="mb-4">
            <h3 className="text-zinc-200 font-bold mb-2 text-xs flex items-center gap-1.5">
              <span className="w-1 h-3 bg-amber-500 rounded-full" />
              Phase 2: The Causal Chain
            </h3>
            {causalNodes?.length > 0 ? (
              <div className="rounded border border-zinc-800 bg-zinc-950/30 overflow-hidden">
                <table className="w-full text-left border-collapse text-[9px] table-fixed">
                  <thead className="bg-zinc-900/50 text-zinc-500 uppercase tracking-tighter border-b border-zinc-800">
                    <tr>
                      <th className="p-1.5 font-bold w-[42%]">Node</th>
                      <th className="p-1.5 font-bold w-[58%]">Evidence Detail</th>
                    </tr>
                  </thead>
                  <tbody className="text-zinc-400">
                    {causalNodes.map((n, i) => (
                      <tr key={i} className="border-b border-zinc-900/30 last:border-0">
                        <td className="p-1.5 font-semibold text-zinc-300 whitespace-normal break-words leading-tight">{n.data?.label || n.id}</td>
                        <td className="p-1.5 italic leading-tight text-[8px] whitespace-normal break-words">{n.data?.detail || 'No detail.'}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            ) : (
              <p className="px-2.5 text-zinc-600 italic">Collecting sandbox execution traces...</p>
            )}
          </div>
          <div>
            <h3 className="text-zinc-200 font-bold mb-2 text-xs flex items-center gap-1.5">
              <span className="w-1 h-3 bg-emerald-500 rounded-full" />
              Phase 3: The Validation Proof
            </h3>
            {lastValidatorResult ? (
              <div className="px-2.5 space-y-1.5">
                <div className="flex items-center gap-2">
                  <span className={`px-1.5 py-0.5 rounded text-[9px] font-bold ${lastValidatorResult.status === 'PASS' ? 'bg-emerald-500/20 text-emerald-400' : 'bg-red-500/20 text-red-400'}`}>
                    {lastValidatorResult.status}
                  </span>
                  <span className="text-zinc-500 font-mono text-[9px]">
                    Peak VRAM: {lastValidatorResult.vram_peak_mb || 0}MB
                  </span>
                </div>
                <p className="text-[10px] text-zinc-400 leading-relaxed italic">
                  {lastValidatorResult.validator_detail || 'Validated with automated environment suite.'}
                </p>
              </div>
            ) : (
              <p className="px-2.5 text-zinc-600 italic">Awaiting final validation checks...</p>
            )}
          </div>
        </motion.div>
      ) : (
        <motion.div
          initial={{ opacity: 0, y: 8 }}
          animate={{ opacity: 1, y: 0 }}
          className="text-zinc-300 text-xs leading-relaxed rca-markdown-container"
        >
          <ReactMarkdown
            remarkPlugins={[remarkGfm]}
            components={{
              h1: ({ children }) => <h1 className="hidden">{children}</h1>,
              h2: ({ children }) => <h2 className="text-sm font-bold text-zinc-100 border-b border-zinc-800 pb-1 mt-4 mb-2">{children}</h2>,
              h3: ({ children }) => <h3 className="text-xs font-bold text-blue-400 mt-4 mb-2 flex items-center gap-2"><span className="w-1 h-3 bg-blue-500 rounded-full" />{children}</h3>,
              p: ({ children }) => <p className="mb-2 text-zinc-400 px-1">{children}</p>,
              ul: ({ children }) => <ul className="list-disc pl-5 mb-3 space-y-1 text-zinc-400">{children}</ul>,
              li: ({ children }) => <li className="text-[10px]">{children}</li>,
              table: ({ children }) => (
                <div className="my-2 overflow-x-auto rounded border border-zinc-800 bg-zinc-950/50">
                  <table className="w-full border-collapse text-[8px] text-zinc-400 table-fixed">{children}</table>
                </div>
              ),
              thead: ({ children }) => <thead className="bg-zinc-900 text-zinc-500 uppercase tracking-tighter border-b border-zinc-800">{children}</thead>,
              th: ({ children }) => <th className="p-1 text-left font-bold">{children}</th>,
              td: ({ children }) => <td className="p-1 border-t border-zinc-900/50 break-words">{children}</td>,
              code: ({ children }) => <code className="bg-zinc-800 text-emerald-400 px-1 rounded text-[9px] font-mono">{children}</code>,
            }}
          >
            {rcaDocument}
          </ReactMarkdown>
        </motion.div>
      )}
    </div>
  );
}
