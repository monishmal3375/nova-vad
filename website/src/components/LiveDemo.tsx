"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import { motion, useReducedMotion } from "framer-motion";
import { Reveal } from "./Reveal";
import { AudioVisualizer } from "./AudioVisualizer";

type FeatureDriver = {
  feature: string;
  importance: number;
  value: number;
  meaning: string;
};

type Explanation = {
  file: string;
  label: "SPEECH" | "NO SPEECH";
  confidence: number;
  top_features: FeatureDriver[];
};

type ClipKind = "speech" | "noise";

type Clip = {
  id: string;
  title: string;
  kind: ClipKind;
  category: string;
  audioSrc: string;
  explanationSrc: string;
  credit?: { label: string; url: string };
};

const CLIPS: Clip[] = [
  {
    id: "speech_demo_1",
    title: "Human speech A",
    kind: "speech",
    category: "Speech Commands",
    audioSrc: "/demo/speech_demo_1.wav",
    explanationSrc: "/demo/speech_demo_1.explanation.json",
  },
  {
    id: "speech_demo_2",
    title: "Human speech B",
    kind: "speech",
    category: "Speech Commands",
    audioSrc: "/demo/speech_demo_2.wav",
    explanationSrc: "/demo/speech_demo_2.explanation.json",
  },
  {
    id: "noise_demo_6",
    title: "Traffic",
    kind: "noise",
    category: "Passing car",
    audioSrc: "/demo/noise_demo_6.wav",
    explanationSrc: "/demo/noise_demo_6.explanation.json",
    credit: { label: "Breviceps — Freesound.org (CC0)", url: "https://freesound.org/people/Breviceps/sounds/462862/" },
  },
  {
    id: "noise_demo_4",
    title: "Siren",
    kind: "noise",
    category: "Ambulance / fire truck",
    audioSrc: "/demo/noise_demo_4.wav",
    explanationSrc: "/demo/noise_demo_4.explanation.json",
    credit: { label: "Breviceps — Freesound.org (CC0)", url: "https://freesound.org/people/Breviceps/sounds/535776/" },
  },
  {
    id: "noise_demo_5",
    title: "Construction",
    kind: "noise",
    category: "Jackhammer",
    audioSrc: "/demo/noise_demo_5.wav",
    explanationSrc: "/demo/noise_demo_5.explanation.json",
    credit: { label: "mindgraveyard — Freesound.org (CC0)", url: "https://freesound.org/people/mindgraveyard/sounds/511509/" },
  },
  {
    id: "noise_demo_3",
    title: "Dishes",
    kind: "noise",
    category: "Household noise",
    audioSrc: "/demo/noise_demo_3.wav",
    explanationSrc: "/demo/noise_demo_3.explanation.json",
  },
  {
    id: "noise_demo_1",
    title: "White noise",
    kind: "noise",
    category: "Synthetic noise",
    audioSrc: "/demo/noise_demo_1.wav",
    explanationSrc: "/demo/noise_demo_1.explanation.json",
  },
  {
    id: "noise_demo_2",
    title: "Pink noise",
    kind: "noise",
    category: "Synthetic noise",
    audioSrc: "/demo/noise_demo_2.wav",
    explanationSrc: "/demo/noise_demo_2.explanation.json",
  },
];

function SpeechIcon() {
  return (
    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" aria-hidden="true">
      <path
        d="M12 15a3 3 0 0 0 3-3V6a3 3 0 0 0-6 0v6a3 3 0 0 0 3 3Z"
        stroke="currentColor"
        strokeWidth="1.6"
      />
      <path
        d="M5 11v1a7 7 0 0 0 14 0v-1M12 19v3"
        stroke="currentColor"
        strokeWidth="1.6"
        strokeLinecap="round"
      />
    </svg>
  );
}

function NoiseIcon() {
  return (
    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" aria-hidden="true">
      <path
        d="M3 12h2.5l2-6 3 12 2.5-9 1.5 4.5H21"
        stroke="currentColor"
        strokeWidth="1.6"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
    </svg>
  );
}

function PlayIcon() {
  return (
    <svg width="16" height="16" viewBox="0 0 24 24" fill="currentColor" aria-hidden="true">
      <path d="M8 5.14v13.72c0 .68.73 1.1 1.32.76l11-6.86a.87.87 0 0 0 0-1.52l-11-6.86A.87.87 0 0 0 8 5.14Z" />
    </svg>
  );
}

function PauseIcon() {
  return (
    <svg width="16" height="16" viewBox="0 0 24 24" fill="currentColor" aria-hidden="true">
      <rect x="6" y="4.5" width="4.5" height="15" rx="1" />
      <rect x="13.5" y="4.5" width="4.5" height="15" rx="1" />
    </svg>
  );
}

function formatTime(sec: number) {
  if (!Number.isFinite(sec)) return "0:00";
  const m = Math.floor(sec / 60);
  const s = Math.floor(sec % 60)
    .toString()
    .padStart(2, "0");
  return `${m}:${s}`;
}

export function LiveDemo() {
  const [activeId, setActiveId] = useState<string>(CLIPS[0].id);
  const [explanations, setExplanations] = useState<Record<string, Explanation>>({});
  const [isPlaying, setIsPlaying] = useState(false);
  const [progress, setProgress] = useState(0);
  const [duration, setDuration] = useState(0);
  const [currentTime, setCurrentTime] = useState(0);
  const [loadError, setLoadError] = useState<string | null>(null);
  const audioRef = useRef<HTMLAudioElement | null>(null);
  const prefersReducedMotion = useReducedMotion();

  const activeClip = useMemo(() => CLIPS.find((c) => c.id === activeId)!, [activeId]);
  const activeExplanation = explanations[activeId];

  // Fetch the precomputed explanation JSON for a clip the first time it's selected.
  useEffect(() => {
    if (explanations[activeId]) return;
    let cancelled = false;
    const clip = CLIPS.find((c) => c.id === activeId)!;
    fetch(clip.explanationSrc)
      .then((res) => {
        if (!res.ok) throw new Error(`Failed to load explanation (${res.status})`);
        return res.json();
      })
      .then((data: Explanation) => {
        if (!cancelled) {
          setExplanations((prev) => ({ ...prev, [activeId]: data }));
          setLoadError(null);
        }
      })
      .catch((err: Error) => {
        if (!cancelled) setLoadError(err.message);
      });
    return () => {
      cancelled = true;
    };
  }, [activeId, explanations]);

  const resetPlayback = () => {
    setIsPlaying(false);
    setProgress(0);
    setCurrentTime(0);
    setDuration(0);
    if (audioRef.current) {
      audioRef.current.pause();
      audioRef.current.currentTime = 0;
    }
  };

  const selectClip = (id: string) => {
    if (id === activeId) return;
    setLoadError(null);
    resetPlayback();
    setActiveId(id);
  };

  const togglePlay = () => {
    const audio = audioRef.current;
    if (!audio) return;
    if (isPlaying) {
      audio.pause();
    } else {
      void audio.play();
    }
  };

  const onTimeUpdate = () => {
    const audio = audioRef.current;
    if (!audio || !audio.duration) return;
    setCurrentTime(audio.currentTime);
    setProgress(audio.currentTime / audio.duration);
  };

  const onLoadedMetadata = () => {
    const audio = audioRef.current;
    if (!audio) return;
    setDuration(audio.duration);
  };

  const onSeek = (e: React.ChangeEvent<HTMLInputElement>) => {
    const audio = audioRef.current;
    if (!audio || !audio.duration) return;
    const ratio = Number(e.target.value);
    audio.currentTime = ratio * audio.duration;
    setProgress(ratio);
  };

  const maxImportance = activeExplanation
    ? Math.max(...activeExplanation.top_features.map((f) => f.importance))
    : 1;

  const speechClips = CLIPS.filter((c) => c.kind === "speech");
  const noiseClips = CLIPS.filter((c) => c.kind === "noise");

  return (
    <section id="live-demo" className="mx-auto max-w-5xl px-6 py-24 sm:py-32">
      <Reveal className="mx-auto max-w-2xl text-center">
        <p className="text-sm font-medium uppercase tracking-widest text-primary">Hear it work</p>
        <h2 className="mt-3 text-balance font-display text-3xl font-semibold tracking-tight sm:text-4xl">
          Play a clip, see the real prediction
        </h2>
        <p className="mt-4 text-balance text-muted">
          Pick any clip below. Each one plays real audio and shows NOVA-VAD&apos;s actual,
          precomputed prediction for that exact file — the same confidence score and feature
          drivers the model produced offline, not a live in-browser guess.
        </p>
      </Reveal>

      <Reveal delay={0.1} className="mt-12">
        <audio
          ref={audioRef}
          src={activeClip.audioSrc}
          preload="metadata"
          onPlay={() => setIsPlaying(true)}
          onPause={() => setIsPlaying(false)}
          onEnded={() => setIsPlaying(false)}
          onTimeUpdate={onTimeUpdate}
          onLoadedMetadata={onLoadedMetadata}
        />

        <div className="grid grid-cols-1 gap-3 sm:grid-cols-2">
          <div>
            <p className="mb-2 flex items-center gap-1.5 text-xs font-medium uppercase tracking-wide text-muted">
              <SpeechIcon /> Human speech
            </p>
            <div className="flex flex-wrap gap-2">
              {speechClips.map((clip) => (
                <ClipButton key={clip.id} clip={clip} active={clip.id === activeId} onSelect={selectClip} />
              ))}
            </div>
          </div>
          <div>
            <p className="mb-2 flex items-center gap-1.5 text-xs font-medium uppercase tracking-wide text-muted">
              <NoiseIcon /> Noise / non-speech
            </p>
            <div className="flex flex-wrap gap-2">
              {noiseClips.map((clip) => (
                <ClipButton key={clip.id} clip={clip} active={clip.id === activeId} onSelect={selectClip} />
              ))}
            </div>
          </div>
        </div>

        <div className="mt-8 rounded-2xl border border-border bg-surface p-6 shadow-[0_0_40px_rgba(0,0,0,0.3)] sm:p-8">
          <div className="flex flex-col gap-4 border-b border-border pb-5 sm:flex-row sm:items-center sm:justify-between">
            <div className="flex items-center gap-3">
              <button
                type="button"
                onClick={togglePlay}
                aria-label={isPlaying ? `Pause ${activeClip.title}` : `Play ${activeClip.title}`}
                aria-pressed={isPlaying}
                className="inline-flex h-11 w-11 shrink-0 cursor-pointer items-center justify-center rounded-full bg-primary text-background transition-transform duration-150 hover:scale-105 focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-primary"
              >
                {isPlaying ? <PauseIcon /> : <PlayIcon />}
              </button>
              <div>
                <p className="font-mono text-sm text-foreground">{activeClip.title}</p>
                <p className="text-xs text-muted">{activeClip.category}</p>
              </div>
            </div>
            {activeExplanation && (
              <span
                className={`inline-flex w-fit items-center rounded-full px-3 py-1 text-xs font-semibold ${
                  activeExplanation.label === "SPEECH"
                    ? "bg-primary/15 text-primary"
                    : "bg-surface-2 text-muted"
                }`}
              >
                {activeExplanation.label}
              </span>
            )}
          </div>

          <AudioVisualizer audioRef={audioRef} isPlaying={isPlaying} />

          <div className="mt-5 flex items-center gap-3">
            <span className="w-9 shrink-0 font-mono text-[11px] text-muted">
              {formatTime(currentTime)}
            </span>
            <input
              type="range"
              min={0}
              max={1}
              step={0.001}
              value={Number.isFinite(progress) ? progress : 0}
              onChange={onSeek}
              aria-label={`Seek within ${activeClip.title}`}
              className="h-1.5 flex-1 cursor-pointer appearance-none rounded-full bg-surface-2 accent-primary"
              style={{
                background: `linear-gradient(to right, var(--primary) ${((Number.isFinite(progress) ? progress : 0) * 100).toFixed(1)}%, var(--surface-2) 0%)`,
              }}
            />
            <span className="w-9 shrink-0 font-mono text-[11px] text-muted">
              {formatTime(duration)}
            </span>
          </div>

          {loadError && (
            <p className="mt-4 text-xs text-danger">Could not load the prediction for this clip: {loadError}</p>
          )}

          {activeExplanation && (
            <>
              <div className="mt-6 flex items-center gap-4">
                <span className="text-xs text-muted">Confidence</span>
                <div className="relative h-2 flex-1 overflow-hidden rounded-full bg-surface-2">
                  <motion.div
                    key={`${activeId}-confidence`}
                    className="absolute inset-y-0 left-0 rounded-full bg-gradient-to-r from-primary-dim to-primary"
                    initial={{ width: prefersReducedMotion ? `${activeExplanation.confidence}%` : 0 }}
                    animate={{ width: `${activeExplanation.confidence}%` }}
                    transition={{ duration: prefersReducedMotion ? 0 : 0.8, ease: [0.16, 1, 0.3, 1] }}
                  />
                </div>
                <span className="font-mono text-sm font-semibold text-primary">
                  {activeExplanation.confidence.toFixed(2)}%
                </span>
              </div>

              <p className="mt-6 text-xs font-medium uppercase tracking-wide text-muted">
                Why this decision was made
              </p>

              <div className="mt-4 space-y-4">
                {activeExplanation.top_features.slice(0, 5).map((d, i) => (
                  <motion.div
                    key={`${activeId}-${d.feature}`}
                    initial={{ opacity: prefersReducedMotion ? 1 : 0, x: prefersReducedMotion ? 0 : -12 }}
                    animate={{ opacity: 1, x: 0 }}
                    transition={{
                      duration: prefersReducedMotion ? 0 : 0.5,
                      delay: prefersReducedMotion ? 0 : i * 0.08,
                      ease: [0.16, 1, 0.3, 1],
                    }}
                  >
                    <div className="flex items-center justify-between text-xs">
                      <span className="font-medium text-foreground">{d.feature}</span>
                      <span className="font-mono text-primary">{d.importance.toFixed(2)}%</span>
                    </div>
                    <div className="mt-1.5 h-1.5 overflow-hidden rounded-full bg-surface-2">
                      <motion.div
                        className="h-full rounded-full bg-wave-blue"
                        initial={{ width: prefersReducedMotion ? `${(d.importance / maxImportance) * 100}%` : 0 }}
                        animate={{ width: `${(d.importance / maxImportance) * 100}%` }}
                        transition={{
                          duration: prefersReducedMotion ? 0 : 0.8,
                          delay: prefersReducedMotion ? 0 : i * 0.08 + 0.15,
                          ease: [0.16, 1, 0.3, 1],
                        }}
                      />
                    </div>
                    <p className="mt-1 text-[11px] leading-snug text-muted">{d.meaning}</p>
                  </motion.div>
                ))}
              </div>
            </>
          )}

          {!activeExplanation && !loadError && (
            <p className="mt-6 text-xs text-muted">Loading prediction…</p>
          )}
        </div>

        <p className="mt-4 text-[11px] leading-relaxed text-muted">
          Predictions shown are precomputed, real model output (saved from{" "}
          <code className="font-mono text-foreground/80">src.explainer</code>), not live
          in-browser inference. Speech clips are from the Google Speech Commands Dataset (CC BY
          4.0). Traffic, siren, and construction clips are CC0 field recordings from Freesound.org:{" "}
          {CLIPS.filter((c) => c.credit).map((c, i, arr) => (
            <span key={c.id}>
              <a
                href={c.credit!.url}
                target="_blank"
                rel="noreferrer"
                className="underline decoration-dotted underline-offset-2 hover:text-primary"
              >
                {c.credit!.label}
              </a>
              {i < arr.length - 1 ? ", " : "."}
            </span>
          ))}{" "}
          No dedicated pure-silence clip is included in this demo set — the closest available
          non-speech example is white noise, shown above.
        </p>
      </Reveal>
    </section>
  );
}

function ClipButton({
  clip,
  active,
  onSelect,
}: {
  clip: Clip;
  active: boolean;
  onSelect: (id: string) => void;
}) {
  return (
    <button
      type="button"
      onClick={() => onSelect(clip.id)}
      aria-pressed={active}
      aria-label={`Load ${clip.title} (${clip.kind === "speech" ? "human speech" : "noise"} clip)`}
      className={`inline-flex min-h-11 cursor-pointer items-center gap-2 rounded-full px-4 text-xs font-medium transition-colors duration-200 ${
        active
          ? "bg-primary text-background"
          : "border border-border bg-surface text-muted hover:text-foreground"
      }`}
    >
      {clip.kind === "speech" ? <SpeechIcon /> : <NoiseIcon />}
      {clip.title}
    </button>
  );
}
