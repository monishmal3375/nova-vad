"use client";

import { useState } from "react";
import { motion } from "framer-motion";
import { Terminal } from "lucide-react";
import { Reveal, RevealGroup, RevealItem } from "./Reveal";

function ReproCallout() {
  const [copied, setCopied] = useState(false);
  const cmd = "python3 -m src.fair_comparison";

  return (
    <div className="mt-6 flex flex-col items-center justify-between gap-4 rounded-xl border border-primary/25 bg-primary/5 px-5 py-4 sm:flex-row">
      <div className="flex items-start gap-3">
        <Terminal className="mt-0.5 h-4 w-4 shrink-0 text-primary" aria-hidden="true" />
        <p className="text-sm leading-relaxed text-muted">
          <span className="font-medium text-foreground">Don&apos;t take our word for it.</span>{" "}
          Clone the repo and re-run this entire table on your machine with one command.
        </p>
      </div>
      <button
        onClick={async () => {
          await navigator.clipboard.writeText(cmd);
          setCopied(true);
          setTimeout(() => setCopied(false), 1800);
        }}
        className="inline-flex min-h-11 shrink-0 cursor-pointer items-center gap-2 rounded-lg border border-border bg-surface px-4 font-mono text-xs text-primary transition-colors duration-150 hover:border-primary/50"
      >
        {copied ? "Copied" : cmd}
      </button>
    </div>
  );
}

const MODELS = [
  { name: "WebRTC VAD", accuracy: 36.74, precision: 28.90, recall: 71.79, f1: 41.21, highlight: false },
  { name: "TEN-VAD", accuracy: 80.43, precision: 65.99, recall: 75.58, f1: 70.46, highlight: false },
  { name: "Pyannote VAD", accuracy: 90.57, precision: 78.25, recall: 96.21, f1: 86.31, highlight: false },
  { name: "SpeechBrain VAD", accuracy: 93.37, precision: 88.61, recall: 90.11, f1: 89.35, highlight: false },
  { name: "Silero VAD", accuracy: 95.19, precision: 91.34, recall: 93.26, f1: 92.29, highlight: false },
  { name: "NOVA-VAD", accuracy: 99.80, precision: 99.58, recall: 99.79, f1: 99.68, highlight: true },
];

const DIFFERENTIATORS = [
  { feature: "Accurate on noisy audio", webrtc: "no", silero: "partial", pyannote: "partial", nova: "yes" },
  { feature: "Lightweight core classifier", webrtc: "yes", silero: "no", pyannote: "no", nova: "yes" },
  { feature: "Fully open source", webrtc: "yes", silero: "partial", pyannote: "yes", nova: "yes" },
  { feature: "Explains every decision", webrtc: "no", silero: "no", pyannote: "no", nova: "yes" },
  { feature: "Retrainable on custom data", webrtc: "no", silero: "no", pyannote: "no", nova: "yes" },
  { feature: "Confidence scores", webrtc: "no", silero: "no", pyannote: "no", nova: "yes" },
];

function Mark({ state }: { state: string }) {
  if (state === "yes") {
    return (
      <span className="inline-flex h-6 w-6 items-center justify-center rounded-full bg-primary/15 text-primary">
        <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="3">
          <path d="M5 13l4 4L19 7" strokeLinecap="round" strokeLinejoin="round" />
        </svg>
        <span className="sr-only">Yes</span>
      </span>
    );
  }
  if (state === "partial") {
    return (
      <span className="inline-flex h-6 w-6 items-center justify-center rounded-full bg-muted/15 text-muted">
        <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="3">
          <path d="M6 12h12" strokeLinecap="round" />
        </svg>
        <span className="sr-only">Partial</span>
      </span>
    );
  }
  return (
    <span className="inline-flex h-6 w-6 items-center justify-center rounded-full bg-danger/10 text-danger/70">
      <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="3">
        <path d="M6 6l12 12M18 6L6 18" strokeLinecap="round" />
      </svg>
      <span className="sr-only">No</span>
    </span>
  );
}

export function Benchmarks() {
  return (
    <section id="benchmarks" className="mx-auto max-w-5xl px-6 py-24 sm:py-32">
      <Reveal className="mx-auto max-w-2xl text-center">
        <p className="text-sm font-medium uppercase tracking-widest text-primary">Benchmarks</p>
        <h2 className="mt-3 text-balance font-display text-3xl font-semibold tracking-tight sm:text-4xl">
          Tested against the VADs you already know
        </h2>
        <p className="mt-4 text-balance text-muted">
          1,538 held-out files across all 10 UrbanSound8K noise categories &mdash; traffic,
          sirens, jackhammers, construction, and more &mdash; grouped by source recording and
          speaker so no clip leaks between train and test. Scoped to this repo&apos;s
          noisy-audio test setup, not a universal claim across every speech domain.
        </p>
      </Reveal>

      <RevealGroup className="mt-16 space-y-4" stagger={0.1}>
        {MODELS.map((model) => (
          <RevealItem key={model.name}>
            <div className="flex items-center gap-4 sm:gap-6">
              <span
                className={`w-28 shrink-0 text-sm font-medium sm:w-32 ${
                  model.highlight ? "text-primary" : "text-muted"
                }`}
              >
                {model.name}
              </span>
              <div className="relative h-9 flex-1 overflow-hidden rounded-md bg-surface">
                <motion.div
                  className={`absolute inset-y-0 left-0 rounded-md ${
                    model.highlight
                      ? "bg-gradient-to-r from-primary-dim to-primary shadow-[0_0_18px_rgba(76,211,245,0.4)]"
                      : "bg-wave-blue-dim"
                  }`}
                  initial={{ width: 0 }}
                  whileInView={{ width: `${model.accuracy}%` }}
                  viewport={{ once: true, margin: "-80px" }}
                  transition={{ duration: 1, ease: [0.16, 1, 0.3, 1] }}
                />
              </div>
              <span
                className={`w-14 shrink-0 text-right font-mono text-xs font-semibold sm:w-16 ${
                  model.highlight ? "text-primary" : "text-foreground"
                }`}
              >
                {model.accuracy.toFixed(1)}%
              </span>
            </div>
          </RevealItem>
        ))}
      </RevealGroup>

      <Reveal delay={0.1} className="mt-8 overflow-x-auto rounded-xl border border-border">
        <table className="w-full min-w-[560px] border-collapse text-sm">
          <caption className="sr-only">Full precision, recall, and F1 comparison across VAD models</caption>
          <thead>
            <tr className="border-b border-border bg-surface text-left text-xs uppercase tracking-wide text-muted">
              <th scope="col" className="px-4 py-3 font-medium">Model</th>
              <th scope="col" className="px-4 py-3 font-medium">Accuracy</th>
              <th scope="col" className="px-4 py-3 font-medium">Precision</th>
              <th scope="col" className="px-4 py-3 font-medium">Recall</th>
              <th scope="col" className="px-4 py-3 font-medium">F1</th>
            </tr>
          </thead>
          <tbody>
            {MODELS.map((model, i) => (
              <tr
                key={model.name}
                className={`${i !== MODELS.length - 1 ? "border-b border-border" : ""} ${
                  model.highlight ? "bg-primary/5" : ""
                }`}
              >
                <td className={`px-4 py-3 font-medium ${model.highlight ? "text-primary" : "text-foreground"}`}>
                  {model.name}
                </td>
                <td className="px-4 py-3 tabular-nums text-muted">{model.accuracy.toFixed(2)}%</td>
                <td className="px-4 py-3 tabular-nums text-muted">{model.precision.toFixed(2)}%</td>
                <td className="px-4 py-3 tabular-nums text-muted">{model.recall.toFixed(2)}%</td>
                <td className="px-4 py-3 tabular-nums text-muted">{model.f1.toFixed(2)}%</td>
              </tr>
            ))}
          </tbody>
        </table>
      </Reveal>

      <Reveal delay={0.12}>
        <ReproCallout />
      </Reveal>

      <Reveal delay={0.15} className="mt-24">
        <h3 className="text-center font-display text-2xl font-semibold tracking-tight">
          What makes NOVA-VAD different
        </h3>
        <div className="mt-10 overflow-x-auto rounded-xl border border-border">
          <table className="w-full min-w-[640px] border-collapse text-sm">
            <thead>
              <tr className="border-b border-border bg-surface text-left text-xs uppercase tracking-wide text-muted">
                <th scope="col" className="px-4 py-3 font-medium">Feature</th>
                <th scope="col" className="px-4 py-3 text-center font-medium">WebRTC</th>
                <th scope="col" className="px-4 py-3 text-center font-medium">Silero</th>
                <th scope="col" className="px-4 py-3 text-center font-medium">Pyannote</th>
                <th scope="col" className="px-4 py-3 text-center font-medium text-primary">NOVA-VAD</th>
              </tr>
            </thead>
            <tbody>
              {DIFFERENTIATORS.map((row, i) => (
                <tr key={row.feature} className={i !== DIFFERENTIATORS.length - 1 ? "border-b border-border" : ""}>
                  <td className="px-4 py-3 text-foreground">{row.feature}</td>
                  <td className="px-4 py-3 text-center"><Mark state={row.webrtc} /></td>
                  <td className="px-4 py-3 text-center"><Mark state={row.silero} /></td>
                  <td className="px-4 py-3 text-center"><Mark state={row.pyannote} /></td>
                  <td className="bg-primary/5 px-4 py-3 text-center"><Mark state={row.nova} /></td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Reveal>
    </section>
  );
}
