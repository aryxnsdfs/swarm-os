import { useSimulationState } from '../../store/simulationStore';

export default function ModelSwitcher() {
  const { modelConfig } = useSimulationState();

  return (
    <div className="panel-card p-3 h-full flex flex-col">
      <span className="text-[10px] font-semibold text-zinc-400 uppercase tracking-wider mb-2">
        Model Configuration
      </span>

      {/* GPU Info */}
      <div className="flex items-center justify-between mb-3 pb-2 border-b border-zinc-800">
        <span className="text-[10px] text-zinc-500">Current GPU</span>
        <span className="text-[10px] font-mono text-blue-400">RTX 3060 12GB</span>
      </div>

      {/* Active Model */}
      <div className="mb-3 pb-2 border-b border-zinc-800">
        <span className="text-[9px] text-zinc-500 block mb-1">PRIMARY MODEL</span>
        <div className="flex items-center gap-2">
          <div className="w-1.5 h-1.5 rounded-full bg-emerald-500 status-dot-live" />
          <span className="text-[10px] font-mono text-zinc-200">
            {modelConfig.models[modelConfig.active_model]?.name}
          </span>
        </div>
        <span className="text-[9px] font-mono text-zinc-600 mt-0.5 block">
          {modelConfig.models[modelConfig.active_model]?.hf_id}
        </span>
      </div>

      {/* Per-Agent Overrides */}
      <div className="mb-3 pb-2 border-b border-zinc-800">
        <span className="text-[9px] text-zinc-500 block mb-1">AGENT MODEL ASSIGNMENTS</span>
        <div className="space-y-1">
          {['COMMANDER', 'DETECTIVE', 'CODER'].map((agent) => {
            const overrideKey = modelConfig.agent_model_overrides?.[agent];
            const modelKey = overrideKey || modelConfig.active_model;
            const model = modelConfig.models[modelKey];
            const isOverride = !!overrideKey;
            return (
              <div key={agent} className="flex items-center justify-between">
                <span className="text-[10px] font-mono text-zinc-400">{agent}</span>
                <div className="flex items-center gap-1">
                  {isOverride && (
                    <span className="text-[8px] px-1 py-0.5 rounded bg-purple-900/30 text-purple-400 font-mono">
                      OVERRIDE
                    </span>
                  )}
                  <span className="text-[9px] font-mono text-zinc-500 truncate max-w-[120px]">
                    {model?.name || modelKey}
                  </span>
                </div>
              </div>
            );
          })}
        </div>
      </div>

      {/* All Models */}
      <div className="flex-1 overflow-y-auto">
        <span className="text-[9px] text-zinc-500 block mb-1">AVAILABLE MODELS</span>
        <div className="space-y-1.5">
          {Object.entries(modelConfig.models).map(([key, model]) => {
            const isActive = key === modelConfig.active_model;
            const vramOk = model.max_vram_gb <= 12;
            return (
              <div
                key={key}
                className={`rounded p-2 border text-[10px] ${
                  isActive
                    ? 'border-emerald-800/50 bg-emerald-950/10'
                    : 'border-zinc-800 bg-zinc-900/50'
                }`}
              >
                <div className="flex items-center justify-between mb-0.5">
                  <span className={`font-medium ${isActive ? 'text-emerald-400' : 'text-zinc-300'}`}>
                    {model.name}
                  </span>
                  {isActive && <span className="text-[8px] text-emerald-400 font-mono">ACTIVE</span>}
                </div>
                <div className="flex items-center gap-3 text-[9px] text-zinc-500 font-mono">
                  <span>VRAM: {model.max_vram_gb}GB</span>
                  <span>{model.tokens_per_sec} tok/s</span>
                  {!vramOk && <span className="text-red-400">[!] Exceeds 12GB</span>}
                </div>
                {model.notes && <p className="text-[8px] text-zinc-600 mt-0.5">{model.notes}</p>}
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
}
