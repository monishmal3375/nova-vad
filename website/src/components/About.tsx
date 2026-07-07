"use client";

import { Reveal, RevealGroup, RevealItem } from "./Reveal";

const AUDIENCE = [
  "Voice-agent builders who need cleaner speech boundaries before ASR",
  "Speech researchers testing VAD behavior in noisy environments",
  "Edge/audio developers who want a lightweight baseline without a GPU",
  "Open-source contributors interested in explainable audio ML",
];

const ROADMAP = [
  { label: "pip install nova-vad packaging", done: false },
  { label: "Simple CLI: nova-vad predict", done: false },
  { label: "Harden realtime streaming support", done: false },
  { label: "Research paper writeup", done: false },
];

export function About() {
  return (
    <section id="about" className="mx-auto max-w-5xl px-6 py-24 sm:py-32">
      <div className="grid grid-cols-1 gap-16 lg:grid-cols-2">
        <Reveal>
          <p className="text-sm font-medium uppercase tracking-widest text-primary">Who it&apos;s for</p>
          <h2 className="mt-3 text-balance font-display text-3xl font-semibold tracking-tight">
            Built for people shipping real audio pipelines
          </h2>
          <RevealGroup className="mt-8 space-y-4" stagger={0.06}>
            {AUDIENCE.map((line) => (
              <RevealItem key={line}>
                <div className="flex items-start gap-3">
                  <span className="mt-1.5 h-1.5 w-1.5 shrink-0 rounded-full bg-primary" />
                  <p className="text-sm leading-relaxed text-muted">{line}</p>
                </div>
              </RevealItem>
            ))}
          </RevealGroup>
          <p className="mt-8 text-sm leading-relaxed text-muted">
            NOVA-VAD is early, and useful test coverage matters more than polished hype. If you
            try it on your own noisy dataset,{" "}
            <a
              href="https://github.com/monishmal3375/nova-vad/issues"
              target="_blank"
              rel="noreferrer"
              className="text-primary underline-offset-4 hover:underline"
            >
              open an issue
            </a>{" "}
            with the result &mdash; hard failure cases are especially useful.
          </p>
        </Reveal>

        <Reveal delay={0.1}>
          <p className="text-sm font-medium uppercase tracking-widest text-primary">Roadmap</p>
          <h2 className="mt-3 text-balance font-display text-3xl font-semibold tracking-tight">
            What&apos;s next
          </h2>
          <ul className="mt-8 space-y-3">
            {ROADMAP.map((item) => (
              <li
                key={item.label}
                className="flex items-center gap-3 rounded-lg border border-border bg-surface px-4 py-3"
              >
                <span className="flex h-5 w-5 shrink-0 items-center justify-center rounded-full border border-border text-[10px] text-muted">
                  &middot;
                </span>
                <span className="text-sm text-foreground">{item.label}</span>
              </li>
            ))}
          </ul>
          <a
            href="https://github.com/monishmal3375/nova-vad/blob/main/ROADMAP.md"
            target="_blank"
            rel="noreferrer"
            className="mt-6 inline-flex min-h-11 items-center gap-1.5 text-sm font-medium text-primary hover:underline underline-offset-4"
          >
            Full roadmap
            <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5">
              <path d="M5 12h14M13 6l6 6-6 6" strokeLinecap="round" strokeLinejoin="round" />
            </svg>
          </a>
        </Reveal>
      </div>

      <Reveal delay={0.15} className="mx-auto mt-24 max-w-2xl rounded-2xl border border-border bg-surface p-8 text-center sm:p-12">
        <p className="font-display text-xl font-semibold tracking-tight text-foreground sm:text-2xl">
          Built by Monish, in the open.
        </p>
        <p className="mt-3 text-sm leading-relaxed text-muted">
          NOVA-VAD is early and improving in public. Every noisy-audio failure case reported makes
          the next version better.
        </p>
        <div className="mt-6 flex flex-wrap items-center justify-center gap-3">
          <a
            href="https://github.com/monishmal3375/nova-vad"
            target="_blank"
            rel="noreferrer"
            className="inline-flex min-h-11 items-center gap-2 rounded-full bg-primary px-5 text-sm font-semibold text-background transition-all duration-200 hover:bg-primary-dim cursor-pointer"
          >
            Star on GitHub
          </a>
          <a
            href="https://x.com/Nova_vad"
            target="_blank"
            rel="noreferrer"
            className="inline-flex min-h-11 items-center gap-2 rounded-full border border-border bg-surface-2 px-5 text-sm font-medium text-foreground transition-all duration-200 hover:border-primary/60 cursor-pointer"
          >
            Follow updates
          </a>
        </div>
      </Reveal>
    </section>
  );
}
