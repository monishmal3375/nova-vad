"use client";

import { Mail, ArrowUpRight } from "lucide-react";
import { Reveal } from "./Reveal";

export function ContactCTA() {
  return (
    <section id="contact" className="border-t border-border px-6 py-24 sm:py-32">
      <Reveal className="mx-auto max-w-3xl text-center">
        <p className="text-sm font-medium uppercase tracking-widest text-primary">
          Evaluating a VAD?
        </p>
        <h2 className="mt-3 text-balance font-display text-3xl font-semibold tracking-tight sm:text-4xl">
          Building a voice agent, transcription pipeline, or anything that listens?
        </h2>
        <p className="mx-auto mt-4 max-w-xl text-balance text-muted">
          If your stack currently guesses at speech boundaries in noisy audio, let&apos;s
          talk. Benchmark NOVA-VAD on your own data — and if it gets your audio wrong,
          that failure case makes the model better.
        </p>
        <div className="mt-10 flex flex-col items-center justify-center gap-4 sm:flex-row">
          <a
            href="mailto:novavad4@gmail.com?subject=NOVA-VAD%20evaluation"
            className="inline-flex min-h-11 items-center gap-2 rounded-full bg-primary px-6 text-sm font-semibold text-background transition-all duration-200 hover:bg-primary-dim hover:shadow-[0_0_24px_rgba(76,211,245,0.35)] cursor-pointer"
          >
            <Mail className="h-4 w-4" aria-hidden="true" />
            novavad4@gmail.com
          </a>
          <a
            href="https://x.com/Nova_vad"
            target="_blank"
            rel="noreferrer"
            className="inline-flex min-h-11 items-center gap-2 rounded-full border border-border bg-surface px-6 text-sm font-semibold text-foreground transition-all duration-200 hover:border-primary/60 hover:bg-surface-2 cursor-pointer"
          >
            Follow the build on X
            <ArrowUpRight className="h-4 w-4" aria-hidden="true" />
          </a>
        </div>
      </Reveal>
    </section>
  );
}
