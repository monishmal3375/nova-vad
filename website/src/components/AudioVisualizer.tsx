"use client";

import { useEffect, useRef, useState } from "react";
import { useReducedMotion } from "framer-motion";
import { AudioLines, Waves } from "lucide-react";

type VisualMode = "bars" | "waveform";

// Tracks which <audio> elements already have a MediaElementAudioSourceNode
// attached. createMediaElementSource throws if called twice on the same
// element, which would otherwise happen under React StrictMode's dev-only
// double-invocation of effects (mount -> cleanup -> mount).
const connectedElements = new WeakSet<HTMLMediaElement>();

/**
 * Live, real-time visualization of the audio actually playing in the given
 * <audio> element. Connects a single AudioContext + AnalyserNode to the
 * element via createMediaElementSource and paints either the live frequency
 * spectrum (getByteFrequencyData) or the live time-domain waveform
 * (getByteTimeDomainData) onto a canvas, every animation frame, only while
 * the element is actually playing.
 *
 * No fabricated data: every pixel drawn comes from the AnalyserNode reading
 * the real, currently-playing signal. The only numeric readout (fps) is a
 * live measurement of this component's own rendering rate, computed from
 * actual frame timestamps.
 */
export function AudioVisualizer({
  audioRef,
  isPlaying,
}: {
  audioRef: React.RefObject<HTMLAudioElement | null>;
  isPlaying: boolean;
}) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const containerRef = useRef<HTMLDivElement | null>(null);
  const prefersReducedMotion = useReducedMotion();

  const [mode, setMode] = useState<VisualMode>("bars");
  const [fps, setFps] = useState<number | null>(null);
  const [audioError, setAudioError] = useState<string | null>(null);

  // Persistent Web Audio graph — created once per audio element and reused
  // across clip changes (createMediaElementSource may only be called once
  // per <audio> element for its lifetime).
  const audioCtxRef = useRef<AudioContext | null>(null);
  const analyserRef = useRef<AnalyserNode | null>(null);
  const sourceRef = useRef<MediaElementAudioSourceNode | null>(null);

  // Buffers + rAF handle + fps bookkeeping, all allocated once and reused
  // every frame — nothing is allocated inside the animation loop.
  const freqDataRef = useRef<Uint8Array<ArrayBuffer> | null>(null);
  const timeDataRef = useRef<Uint8Array<ArrayBuffer> | null>(null);
  const rafRef = useRef<number | null>(null);
  const frameCountRef = useRef(0);
  const fpsWindowStartRef = useRef(0);
  const modeRef = useRef<VisualMode>("bars");
  const dprRef = useRef(1);

  useEffect(() => {
    modeRef.current = mode;
  }, [mode]);

  // Set up the AudioContext / AnalyserNode graph once per <audio> element.
  // createMediaElementSource may only be called once per element for its
  // entire lifetime, so we track that on the element itself (via a WeakSet)
  // to stay correct under React StrictMode's dev-only double-invoke of
  // effects, which would otherwise try to create a second source node (or
  // close a context that a second mount still needs) and throw.
  useEffect(() => {
    const audio = audioRef.current;
    if (!audio) return;

    let cancelled = false;

    if (!audioCtxRef.current && !connectedElements.has(audio)) {
      try {
        const Ctor: typeof AudioContext | undefined =
          window.AudioContext ||
          (window as unknown as { webkitAudioContext?: typeof AudioContext })
            .webkitAudioContext;
        if (!Ctor) {
          queueMicrotask(() =>
            setAudioError("Web Audio API is not supported in this browser.")
          );
        } else {
          const ctx = new Ctor();
          const analyser = ctx.createAnalyser();
          analyser.fftSize = 2048;
          analyser.smoothingTimeConstant = 0.8;

          // Route element audio -> analyser -> destination so playback is
          // unaffected and still audible, while we tap the real signal.
          const source = ctx.createMediaElementSource(audio);
          source.connect(analyser);
          analyser.connect(ctx.destination);
          connectedElements.add(audio);

          if (cancelled) {
            // StrictMode already unmounted us before this finished — leave
            // the graph connected (it's tied to the element permanently)
            // but don't stash it on refs that are about to be torn down.
            void ctx.close();
          } else {
            audioCtxRef.current = ctx;
            analyserRef.current = analyser;
            sourceRef.current = source;
            freqDataRef.current = new Uint8Array(
              new ArrayBuffer(analyser.frequencyBinCount)
            );
            timeDataRef.current = new Uint8Array(
              new ArrayBuffer(analyser.frequencyBinCount)
            );
          }
        }
      } catch (err) {
        const message =
          err instanceof Error ? err.message : "Could not start audio analysis.";
        queueMicrotask(() => setAudioError(message));
      }
    }

    return () => {
      cancelled = true;
      if (rafRef.current !== null) {
        cancelAnimationFrame(rafRef.current);
        rafRef.current = null;
      }
      // Don't close/disconnect here: the graph is tied to the <audio>
      // element for its whole lifetime, and this effect (dev StrictMode
      // aside) only tears down when the section itself unmounts, at which
      // point the tab is navigating away and the context is reclaimed
      // naturally. Closing on every effect cleanup would break re-mounts.
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Hoisted canvas + drawing setup, sized once (and on resize), never
  // recreated inside the animation loop.
  useEffect(() => {
    const canvas = canvasRef.current;
    const container = containerRef.current;
    if (!canvas || !container) return;

    const ctx2d = canvas.getContext("2d");
    if (!ctx2d) return;

    const resize = () => {
      const dpr = Math.min(window.devicePixelRatio || 1, 2);
      dprRef.current = dpr;
      const rect = container.getBoundingClientRect();
      canvas.width = Math.max(1, Math.round(rect.width * dpr));
      canvas.height = Math.max(1, Math.round(rect.height * dpr));
    };

    resize();
    const observer = new ResizeObserver(resize);
    observer.observe(container);
    return () => observer.disconnect();
  }, []);

  // The animation loop itself: started/stopped based on isPlaying and
  // reduced-motion, never re-allocating the canvas or gradient.
  useEffect(() => {
    const canvas = canvasRef.current;
    const analyser = analyserRef.current;
    const freqData = freqDataRef.current;
    const timeData = timeDataRef.current;
    if (!canvas || !analyser || !freqData || !timeData) return;

    const ctx2d = canvas.getContext("2d");
    if (!ctx2d) return;

    if (!isPlaying) {
      if (rafRef.current !== null) {
        cancelAnimationFrame(rafRef.current);
        rafRef.current = null;
      }
      return;
    }

    if (audioCtxRef.current?.state === "suspended") {
      void audioCtxRef.current.resume();
    }

    // Gradient objects allocated once per (start of) playback, reused every
    // frame inside the loop below — not recreated per-frame.
    let barGradient: CanvasGradient | null = null;
    let waveGradient: CanvasGradient | null = null;
    const buildGradients = () => {
      const w = canvas.width;
      const h = canvas.height;
      barGradient = ctx2d.createLinearGradient(0, h, 0, 0);
      barGradient.addColorStop(0, "#2a9fc4");
      barGradient.addColorStop(1, "#4cd3f5");
      waveGradient = ctx2d.createLinearGradient(0, 0, w, 0);
      waveGradient.addColorStop(0, "#2a9fc4");
      waveGradient.addColorStop(0.5, "#4cd3f5");
      waveGradient.addColorStop(1, "#2a9fc4");
    };
    buildGradients();

    // Precompute a logarithmic bin-range lookup once per playback start (not
    // per frame): real-world audio (speech, most environmental noise) is
    // heavily weighted toward lower frequencies, and these demo clips are
    // 16 kHz recordings running through a browser AudioContext that is
    // typically 44.1/48 kHz — so a linear bin-per-bar mapping would waste
    // most of the canvas on empty high-frequency bins. A log-frequency
    // mapping is also how real spectrum analyzers (e.g. media players,
    // DAWs) display audio, so this stays a faithful real-time
    // representation of the analyser's actual output, just remapped for
    // legibility.
    const barCount = 64;
    const binCount = freqData.length;
    const barBinRanges: Array<[number, number]> = [];
    for (let i = 0; i < barCount; i++) {
      const p0 = i / barCount;
      const p1 = (i + 1) / barCount;
      // log-scale from bin 1 (skip DC bin 0) to binCount
      const lo = Math.floor(Math.pow(binCount, p0));
      const hi = Math.max(lo + 1, Math.floor(Math.pow(binCount, p1)));
      barBinRanges.push([Math.max(1, lo), Math.min(binCount, hi)]);
    }

    frameCountRef.current = 0;
    fpsWindowStartRef.current = performance.now();

    const draw = () => {
      rafRef.current = requestAnimationFrame(draw);

      const w = canvas.width;
      const h = canvas.height;

      ctx2d.clearRect(0, 0, w, h);
      ctx2d.fillStyle = "rgba(11, 15, 24, 0.35)";
      ctx2d.fillRect(0, 0, w, h);

      if (modeRef.current === "bars") {
        analyser.getByteFrequencyData(freqData);
        const gap = Math.max(1, w * 0.003);
        const barWidth = w / barCount - gap;

        ctx2d.fillStyle = barGradient!;
        for (let i = 0; i < barCount; i++) {
          const [lo, hi] = barBinRanges[i];
          let sum = 0;
          for (let j = lo; j < hi; j++) sum += freqData[j];
          const avg = sum / (hi - lo);
          const barHeight = (avg / 255) * h;
          const x = i * (barWidth + gap);
          ctx2d.fillRect(x, h - barHeight, barWidth, barHeight);
        }
      } else {
        analyser.getByteTimeDomainData(timeData);
        ctx2d.lineWidth = Math.max(1.5, w * 0.0025);
        ctx2d.strokeStyle = waveGradient!;
        ctx2d.beginPath();
        const sliceWidth = w / timeData.length;
        let x = 0;
        for (let i = 0; i < timeData.length; i++) {
          const v = timeData[i] / 128.0;
          const y = (v * h) / 2;
          if (i === 0) ctx2d.moveTo(x, y);
          else ctx2d.lineTo(x, y);
          x += sliceWidth;
        }
        ctx2d.lineTo(w, h / 2);
        ctx2d.stroke();
      }

      frameCountRef.current += 1;
      const now = performance.now();
      const elapsed = now - fpsWindowStartRef.current;
      if (elapsed >= 500) {
        setFps(Math.round((frameCountRef.current * 1000) / elapsed));
        frameCountRef.current = 0;
        fpsWindowStartRef.current = now;
      }
    };

    if (prefersReducedMotion) {
      // Draw a single static frame from the real current data, then stop —
      // no continuous rAF loop when the user asked for reduced motion.
      analyser.getByteFrequencyData(freqData);
      const gap = Math.max(1, canvas.width * 0.003);
      const barWidth = canvas.width / barCount - gap;
      ctx2d.clearRect(0, 0, canvas.width, canvas.height);
      ctx2d.fillStyle = "rgba(11, 15, 24, 0.35)";
      ctx2d.fillRect(0, 0, canvas.width, canvas.height);
      ctx2d.fillStyle = barGradient!;
      for (let i = 0; i < barCount; i++) {
        const [lo, hi] = barBinRanges[i];
        let sum = 0;
        for (let j = lo; j < hi; j++) sum += freqData[j];
        const avg = sum / (hi - lo);
        const barHeight = (avg / 255) * canvas.height;
        const x = i * (barWidth + gap);
        ctx2d.fillRect(x, canvas.height - barHeight, barWidth, barHeight);
      }
      queueMicrotask(() => setFps(null));
      return;
    }

    rafRef.current = requestAnimationFrame(draw);

    return () => {
      if (rafRef.current !== null) {
        cancelAnimationFrame(rafRef.current);
        rafRef.current = null;
      }
    };
    // mode changes are read via modeRef so we don't need to restart the loop
  }, [isPlaying, prefersReducedMotion]);

  return (
    <div className="mt-5">
      <div className="mb-2 flex items-center justify-between gap-3">
        <p className="flex items-center gap-1.5 text-xs font-medium uppercase tracking-wide text-muted">
          Live signal
        </p>
        <div className="flex items-center gap-2">
          {fps !== null && isPlaying && !prefersReducedMotion && (
            <span className="font-mono text-[10px] text-muted" title="Measured render rate of this visualization">
              {fps} fps
            </span>
          )}
          <div
            role="group"
            aria-label="Visualization mode"
            className="inline-flex overflow-hidden rounded-full border border-border"
          >
            <button
              type="button"
              onClick={() => setMode("bars")}
              aria-pressed={mode === "bars"}
              aria-label="Show frequency bars"
              title="Frequency bars"
              className={`inline-flex min-h-11 min-w-11 cursor-pointer items-center justify-center px-3 transition-colors duration-150 ${
                mode === "bars"
                  ? "bg-primary text-background"
                  : "bg-surface text-muted hover:text-foreground"
              }`}
            >
              <AudioLines size={16} aria-hidden="true" />
            </button>
            <button
              type="button"
              onClick={() => setMode("waveform")}
              aria-pressed={mode === "waveform"}
              aria-label="Show waveform trace"
              title="Waveform trace"
              className={`inline-flex min-h-11 min-w-11 cursor-pointer items-center justify-center border-l border-border px-3 transition-colors duration-150 ${
                mode === "waveform"
                  ? "bg-primary text-background"
                  : "bg-surface text-muted hover:text-foreground"
              }`}
            >
              <Waves size={16} aria-hidden="true" />
            </button>
          </div>
        </div>
      </div>

      <div
        ref={containerRef}
        className="relative h-28 w-full overflow-hidden rounded-xl border border-border bg-surface-2 sm:h-32"
      >
        <canvas ref={canvasRef} className="absolute inset-0 h-full w-full" />
        {!isPlaying && (
          <div className="absolute inset-0 flex items-center justify-center">
            <p className="text-[11px] text-muted">Press play to see the live signal</p>
          </div>
        )}
        {audioError && (
          <div className="absolute inset-0 flex items-center justify-center bg-surface-2 px-4">
            <p className="text-center text-[11px] text-danger">{audioError}</p>
          </div>
        )}
      </div>
    </div>
  );
}
