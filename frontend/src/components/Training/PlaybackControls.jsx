export default function PlaybackControls() {
  return (
    <div className="panel-card p-4 h-full flex flex-col items-center justify-center relative overflow-hidden">
      <span className="text-[10px] font-semibold text-zinc-400 uppercase tracking-wider mb-6 z-10">
        Local Training Endpoint
      </span>

      <button
        disabled
        className="group relative flex items-center justify-center w-full max-w-[240px] py-4 rounded font-mono text-sm tracking-widest font-bold bg-zinc-800/50 text-zinc-600 border border-zinc-800 cursor-not-allowed z-10"
      >
        <span className="flex items-center gap-2">
          <svg className="w-4 h-4" viewBox="0 0 24 24" fill="currentColor">
            <path d="M8 5v14l11-7z" />
          </svg>
          DISABLED
        </span>
      </button>

      <p className="text-[10px] text-zinc-500 mt-4 text-center z-10 max-w-[260px] leading-relaxed">
        Mock training playback has been removed. This endpoint stays disabled until a real local training pipeline is wired into the app.
      </p>
    </div>
  );
}
