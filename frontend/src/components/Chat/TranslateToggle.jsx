import * as Switch from '@radix-ui/react-switch';

export default function TranslateToggle({ enabled, onToggle }) {
  return (
    <div className="flex items-center gap-2">
      <span className="text-[10px] text-zinc-500 font-mono">
        {enabled ? 'HUMAN' : 'M2M'}
      </span>
      <Switch.Root
        checked={enabled}
        onCheckedChange={onToggle}
        className={`relative w-8 h-4 rounded-full outline-none transition-colors ${
          enabled ? 'bg-blue-600' : 'bg-zinc-700'
        }`}
      >
        <Switch.Thumb
          className={`block w-3 h-3 rounded-full bg-white transition-transform ${
            enabled ? 'translate-x-[18px]' : 'translate-x-[2px]'
          }`}
        />
      </Switch.Root>
      {enabled ? (
        <svg className="w-3 h-3 text-blue-400" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.5">
          <circle cx="8" cy="8" r="6" />
          <path d="M2 8h12M8 2c-2 2-2 10 0 12M8 2c2 2 2 10 0 12" />
        </svg>
      ) : (
        <svg className="w-3 h-3 text-zinc-500" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.5">
          <path d="M8 2v4l3 2-3 2v4" />
          <path d="M4 6h8M4 10h8" />
        </svg>
      )}
    </div>
  );
}
