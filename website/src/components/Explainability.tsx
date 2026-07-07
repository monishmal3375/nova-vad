"use client";

import { motion } from "framer-motion";
import { Reveal } from "./Reveal";

const DRIVERS = [
  {
    label: "MFCC Delta 1 std",
    weight: 10.63,
    note: "HIGH spectral change rate — dynamic audio like speech",
  },
  {
    label: "MFCC Delta 2 std",
    weight: 6.14,
    note: "HIGH acceleration — rapidly changing audio, speech-like",
  },
  {
    label: "Silence ratio",
    weight: 5.92,
    note: "56% silence — mix of speech and pauses",
  },
  {
    label: "Spectral centroid std",
    weight: 4.27,
    note: "HIGH variation — shifting frequency center",
  },
  {
    label: "Mel mean",
    weight: 3.5,
    note: "MODERATE energy — normal speech level",
  },
];

const MAX_WEIGHT = 10.63;

export function Explainability() {
  return (
    <section id="explainability" className="mx-auto max-w-5xl px-6 py-24 sm:py-32">
      <div className="grid grid-cols-1 items-center gap-14 lg:grid-cols-2">
        <Reveal>
          <p className="text-sm font-medium uppercase tracking-widest text-primary">Explainability</p>
          <h2 className="mt-3 text-balance font-display text-3xl font-semibold tracking-tight sm:text-4xl">
            Every decision comes with a reason
          </h2>
          <p className="mt-4 leading-relaxed text-muted">
            Most VADs are black boxes. NOVA-VAD returns a confidence score and the top features
            that drove the call — in plain English, not raw coefficients — so you can trust it
            or debug it.
          </p>
          <ul className="mt-8 space-y-3 text-sm text-muted">
            <li className="flex items-start gap-2.5">
              <span className="mt-1.5 h-1.5 w-1.5 shrink-0 rounded-full bg-primary" />
              Confidence score on every single prediction
            </li>
            <li className="flex items-start gap-2.5">
              <span className="mt-1.5 h-1.5 w-1.5 shrink-0 rounded-full bg-primary" />
              Top 10 feature drivers, ranked and explained
            </li>
            <li className="flex items-start gap-2.5">
              <span className="mt-1.5 h-1.5 w-1.5 shrink-0 rounded-full bg-primary" />
              Retrainable on your own noisy dataset
            </li>
          </ul>
          <div className="mt-8">
            <code className="rounded-md border border-border bg-surface px-3 py-1.5 font-mono text-xs text-primary">
              python3 -m src.explainer path/to/audio.wav
            </code>
          </div>
        </Reveal>

        <Reveal delay={0.1}>
          <div className="rounded-2xl border border-border bg-surface p-6 shadow-[0_0_40px_rgba(0,0,0,0.3)] sm:p-8">
            <div className="flex items-center justify-between border-b border-border pb-4">
              <div>
                <p className="text-xs text-muted">File</p>
                <p className="font-mono text-sm text-foreground">speech_001.wav</p>
              </div>
              <span className="rounded-full bg-primary/15 px-3 py-1 text-xs font-semibold text-primary">
                SPEECH
              </span>
            </div>

            <div className="mt-5 flex items-center gap-4">
              <span className="text-xs text-muted">Confidence</span>
              <div className="relative h-2 flex-1 overflow-hidden rounded-full bg-surface-2">
                <motion.div
                  className="absolute inset-y-0 left-0 rounded-full bg-gradient-to-r from-primary-dim to-primary"
                  initial={{ width: 0 }}
                  whileInView={{ width: "93.47%" }}
                  viewport={{ once: true, margin: "-80px" }}
                  transition={{ duration: 1, ease: [0.16, 1, 0.3, 1] }}
                />
              </div>
              <span className="font-mono text-sm font-semibold text-primary">93.47%</span>
            </div>

            <p className="mt-6 text-xs font-medium uppercase tracking-wide text-muted">
              Why this decision was made
            </p>

            <div className="mt-4 space-y-4">
              {DRIVERS.map((d, i) => (
                <motion.div
                  key={d.label}
                  initial={{ opacity: 0, x: -12 }}
                  whileInView={{ opacity: 1, x: 0 }}
                  viewport={{ once: true, margin: "-80px" }}
                  transition={{ duration: 0.5, delay: i * 0.08, ease: [0.16, 1, 0.3, 1] }}
                >
                  <div className="flex items-center justify-between text-xs">
                    <span className="font-medium text-foreground">{d.label}</span>
                    <span className="font-mono text-primary">{d.weight.toFixed(2)}%</span>
                  </div>
                  <div className="mt-1.5 h-1.5 overflow-hidden rounded-full bg-surface-2">
                    <motion.div
                      className="h-full rounded-full bg-wave-blue"
                      initial={{ width: 0 }}
                      whileInView={{ width: `${(d.weight / MAX_WEIGHT) * 100}%` }}
                      viewport={{ once: true, margin: "-80px" }}
                      transition={{ duration: 0.8, delay: i * 0.08 + 0.15, ease: [0.16, 1, 0.3, 1] }}
                    />
                  </div>
                  <p className="mt-1 text-[11px] leading-snug text-muted">{d.note}</p>
                </motion.div>
              ))}
            </div>
          </div>
        </Reveal>
      </div>
    </section>
  );
}
