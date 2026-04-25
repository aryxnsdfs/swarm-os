import { AreaChart, Area, XAxis, YAxis, Tooltip, ResponsiveContainer, ReferenceLine } from 'recharts';
import { useSimulationState } from '../../store/simulationStore';

export default function RewardCurve() {
  const { rewardHistory } = useSimulationState();

  const latest = rewardHistory.length > 0
    ? rewardHistory[rewardHistory.length - 1]?.reward
    : null;

  return (
    <div className="panel-card p-3 h-full flex flex-col">
      <div className="flex items-center justify-between mb-2">
        <span className="text-[10px] font-semibold text-zinc-400 uppercase tracking-wider">
          Evaluator Reward Trace
        </span>
        <span className="text-[10px] font-mono text-emerald-400">
          {latest !== null ? `Total: ${latest.toFixed(2)}` : 'Awaiting Evidence'}
        </span>
      </div>
      <div className="flex-1 min-h-0 relative">
        {rewardHistory.length === 0 ? (
          <div className="absolute inset-0 flex items-center justify-center">
            <div className="text-center">
              <div className="w-10 h-10 mx-auto mb-2 rounded-lg bg-zinc-800/50 flex items-center justify-center">
                <svg className="w-5 h-5 text-zinc-600" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
                  <polyline points="22,12 18,12 15,21 9,3 6,12 2,12" />
                </svg>
              </div>
              <p className="text-[10px] text-zinc-600 font-mono">Chart updates from real validator reward events</p>
            </div>
          </div>
        ) : (
          <ResponsiveContainer width="100%" height="100%">
            <AreaChart data={rewardHistory} margin={{ top: 5, right: 5, bottom: 5, left: -20 }}>
              <defs>
                <linearGradient id="rewardGradient" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor="#10b981" stopOpacity={0.3} />
                  <stop offset="95%" stopColor="#10b981" stopOpacity={0} />
                </linearGradient>
              </defs>
              <XAxis
                dataKey="episode"
                axisLine={false}
                tickLine={false}
                tick={{ fontSize: 9, fill: '#52525b' }}
                label={{ value: 'Evaluator Event', position: 'insideBottom', offset: -2, fontSize: 9, fill: '#52525b' }}
              />
              <YAxis
                axisLine={false}
                tickLine={false}
                tick={{ fontSize: 9, fill: '#52525b' }}
              />
              <ReferenceLine y={0} stroke="#3f3f46" strokeDasharray="3 3" />
              <Tooltip
                contentStyle={{
                  backgroundColor: '#18181b',
                  border: '1px solid #27272a',
                  borderRadius: '6px',
                  fontSize: '10px',
                  fontFamily: 'JetBrains Mono, monospace',
                  color: '#a1a1aa',
                }}
                formatter={(value) => [value.toFixed(2), 'Reward']}
                labelFormatter={(label) => `Event ${label}`}
              />
              <Area
                type="linear"
                dataKey="reward"
                stroke="#10b981"
                strokeWidth={2}
                fill="url(#rewardGradient)"
                dot={false}
                activeDot={{ r: 3, fill: '#10b981', stroke: '#18181b', strokeWidth: 2 }}
                isAnimationActive={true}
                animationDuration={400}
              />
            </AreaChart>
          </ResponsiveContainer>
        )}
      </div>
    </div>
  );
}
