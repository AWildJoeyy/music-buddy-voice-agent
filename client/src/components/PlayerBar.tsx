import React from "react";

type Props = {
  playing: boolean;
  current: number;    
  duration: number;     
  volume: number;       

  onPrev: () => void;
  onPlay: () => void;
  onPause: () => void;
  onNext: () => void;
  onSeek: (sec: number) => void;
  onVolume: (vol: number) => void;
};

function fmt(sec: number) {
  if (!isFinite(sec) || sec < 0) sec = 0;
  const m = Math.floor(sec / 60);
  const s = Math.floor(sec % 60);
  return `${m}:${s.toString().padStart(2, "0")}`;
}

export default function PlayerBar({
  playing, current, duration, volume,
  onPrev, onPlay, onPause, onNext, onSeek, onVolume,
}: Props) {
  const p = duration > 0 ? Math.min(100, Math.max(0, (current / duration) * 100)) : 0;

  return (
    <div className="w-full border-t border-white/10 bg-black/60 backdrop-blur supports-[backdrop-filter]:bg-black/40">
      <div className="mx-auto max-w-6xl px-3 py-2">
        <div className="flex items-center gap-4">
          {/* LEFT: transport */}
          <div className="flex items-center gap-2 w-[240px]">
            <button
              onClick={onPrev}
              className="h-9 w-9 grid place-items-center rounded-xl bg-white/5 hover:bg-white/10"
              aria-label="Previous"
              title="Previous"
            >
              ⏮️
            </button>
            {playing ? (
              <button
                onClick={onPause}
                className="h-10 w-10 grid place-items-center rounded-xl bg-white text-black hover:opacity-90"
                aria-label="Pause"
                title="Pause"
              >
                ⏸
              </button>
            ) : (
              <button
                onClick={onPlay}
                className="h-10 w-10 grid place-items-center rounded-xl bg-white text-black hover:opacity-90"
                aria-label="Play"
                title="Play"
              >
                ▶
              </button>
            )}
            <button
              onClick={onNext}
              className="h-9 w-9 grid place-items-center rounded-xl bg-white/5 hover:bg-white/10"
              aria-label="Next"
              title="Next"
            >
              ⏭️
            </button>
          </div>

          {/* CENTER: progress */}
          <div className="flex-1 flex items-center gap-3 min-w-0">
            <span className="text-xs tabular-nums text-white/70 w-10 text-right">
              {fmt(current)}
            </span>
            <input
              type="range"
              min={0}
              max={100}
              value={p}
              onChange={(e) => {
                const pct = Number(e.target.value);
                const sec = duration * (pct / 100);
                onSeek(sec);
              }}
              className="w-full accent-violet-500"
            />
            <span className="text-xs tabular-nums text-white/70 w-10">
              {fmt(duration)}
            </span>
          </div>

          {/* RIGHT: volume */}
          <div className="flex items-center gap-3 w-[240px] justify-end">
            <span className="text-sm text-white/80">🔊 {Math.round(volume)}</span>
            <input
              type="range"
              min={0}
              max={100}
              value={volume}
              onChange={(e) => onVolume(Number(e.target.value))}
              className="w-40 accent-violet-500"
            />
          </div>
        </div>
      </div>
    </div>
  );
}
