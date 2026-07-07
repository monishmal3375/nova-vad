"use client";

import {
  Cpu,
  ScrollText,
  RefreshCw,
  AudioLines,
  MessageSquareText,
} from "lucide-react";
import { motion } from "framer-motion";
import { Reveal, RevealGroup, RevealItem } from "./Reveal";

const MINI_BARS = [
  { label: "NOVA", value: 99.8, highlight: true },
  { label: "Silero", value: 95.19, highlight: false },
  { label: "SpeechBrain", value: 93.37, highlight: false },
  { label: "Pyannote", value: 90.57, highlight: false },
];

function CardShell({
  children,
  className = "",
}: {
  children: React.ReactNode;
  className?: string;
}) {
  return (
    <div
      className={`group relative flex flex-col overflow-hidden rounded-2xl border border-border bg-surface p-6 transition-colors duration-200 hover:border-primary/40 ${className}`}
    >
      {children}
    </div>
  );
}

export function Bento() {
  return (
    <section className="mx-auto max-w-5xl px-6 pb-24 sm:pb-32">
      <Reveal className="mx-auto max-w-2xl text-center">
        <h2 className="text-balance font-display text-3xl font-semibold tracking-tight sm:text-4xl">
          Everything a VAD should be
        </h2>
      </Reveal>

      <RevealGroup className="mt-12 grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-3" stagger={0.07}>
        {/* Dominant card — accuracy */}
        <RevealItem className="sm:col-span-2 lg:row-span-2">
          <CardShell className="h-full justify-between">
            <div>
              <p className="text-xs font-medium uppercase tracking-widest text-primary">
                Noisy-audio accuracy
              </p>
              <p className="mt-3 font-display text-6xl font-semibold text-foreground">
                99.8<span className="text-3xl text-primary">%</span>
              </p>
              <p className="mt-3 max-w-sm text-sm leading-relaxed text-muted">
                1,538 held-out files across all 10 UrbanSound8K noise categories, grouped
                by source recording and speaker so nothing leaks between train and test.
                Measured with the open benchmark script in the repo, so you can re-run
                every number yourself.
              </p>
            </div>
            <div className="mt-8 space-y-2.5">
              {MINI_BARS.map((bar) => (
                <div key={bar.label} className="flex items-center gap-3">
                  <span
                    className={`w-16 shrink-0 text-xs font-medium ${
                      bar.highlight ? "text-primary" : "text-muted"
                    }`}
                  >
                    {bar.label}
                  </span>
                  <div className="relative h-2 flex-1 overflow-hidden rounded-full bg-surface-2">
                    <motion.div
                      className={`absolute inset-y-0 left-0 rounded-full ${
                        bar.highlight
                          ? "bg-gradient-to-r from-primary-dim to-primary"
                          : "bg-wave-blue-dim"
                      }`}
                      initial={{ width: 0 }}
                      whileInView={{ width: `${bar.value}%` }}
                      viewport={{ once: true, margin: "-60px" }}
                      transition={{ duration: 0.9, ease: [0.16, 1, 0.3, 1] }}
                    />
                  </div>
                  <span
                    className={`w-12 shrink-0 text-right font-mono text-xs ${
                      bar.highlight ? "text-primary" : "text-muted"
                    }`}
                  >
                    {bar.value.toFixed(1)}%
                  </span>
                </div>
              ))}
            </div>
          </CardShell>
        </RevealItem>

        <RevealItem>
          <CardShell className="h-full">
            <MessageSquareText className="h-5 w-5 text-primary" aria-hidden="true" />
            <h3 className="mt-4 font-display text-base font-semibold text-foreground">
              Explains every decision
            </h3>
            <p className="mt-2 text-sm leading-relaxed text-muted">
              Confidence score plus the top feature drivers in plain English — not a
              black-box yes/no.
            </p>
          </CardShell>
        </RevealItem>

        <RevealItem>
          <CardShell className="h-full">
            <Cpu className="h-5 w-5 text-primary" aria-hidden="true" />
            <h3 className="mt-4 font-display text-base font-semibold text-foreground">
              No GPU. 1&nbsp;MB model.
            </h3>
            <p className="mt-2 text-sm leading-relaxed text-muted">
              A scikit-learn ensemble that runs anywhere Python runs — laptops, servers,
              edge boxes.
            </p>
          </CardShell>
        </RevealItem>

        <RevealItem>
          <CardShell className="h-full">
            <RefreshCw className="h-5 w-5 text-primary" aria-hidden="true" />
            <h3 className="mt-4 font-display text-base font-semibold text-foreground">
              Retrainable on your noise
            </h3>
            <p className="mt-2 text-sm leading-relaxed text-muted">
              Call-center hum, factory floor, car cabin — retrain on your own labeled
              clips in minutes, no fine-tuning pipeline.
            </p>
          </CardShell>
        </RevealItem>

        <RevealItem>
          <CardShell className="h-full">
            <AudioLines className="h-5 w-5 text-primary" aria-hidden="true" />
            <h3 className="mt-4 font-display text-base font-semibold text-foreground">
              Real-time streaming
            </h3>
            <p className="mt-2 text-sm leading-relaxed text-muted">
              Live microphone input with device selection and flicker-free smoothing,
              built in.
            </p>
          </CardShell>
        </RevealItem>

        <RevealItem className="sm:col-span-2 lg:col-span-1">
          <CardShell className="h-full">
            <ScrollText className="h-5 w-5 text-primary" aria-hidden="true" />
            <h3 className="mt-4 font-display text-base font-semibold text-foreground">
              MIT licensed, fully open
            </h3>
            <p className="mt-2 text-sm leading-relaxed text-muted">
              Every line of the model, benchmark, and training pipeline is public. No
              API keys, no per-call fees, no lock-in.
            </p>
          </CardShell>
        </RevealItem>
      </RevealGroup>
    </section>
  );
}
