import { motion } from 'framer-motion';

const TABS = [
  { id: 'live', label: 'Live Environment' },
  { id: 'training', label: 'Execution Evidence' },
];

export default function TabBar({ activeTab, onTabChange }) {
  return (
    <div className="flex items-center gap-1 px-4 py-1 bg-zinc-900 border-b border-zinc-800 shrink-0">
      {TABS.map((tab) => (
        <button
          key={tab.id}
          onClick={() => onTabChange(tab.id)}
          className={`relative px-4 py-1.5 text-xs font-medium rounded-md transition-colors ${
            activeTab === tab.id
              ? 'text-zinc-100'
              : 'text-zinc-500 hover:text-zinc-300'
          }`}
        >
          <span className="relative z-10 flex items-center gap-1.5">
            {tab.label}
          </span>
          {activeTab === tab.id && (
            <motion.div
              layoutId="tab-indicator"
              className="absolute inset-0 bg-zinc-800 rounded-md"
              style={{ zIndex: 0 }}
              transition={{ type: 'spring', stiffness: 500, damping: 35 }}
            />
          )}
        </button>
      ))}
    </div>
  );
}
