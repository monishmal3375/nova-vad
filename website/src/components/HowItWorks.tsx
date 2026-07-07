"use client";

import { Reveal, RevealGroup, RevealItem } from "./Reveal";

const STEPS = [
  {
    title: "Raw audio",
    detail: "Any input clip or stream, unprocessed.",
  },
  {
    title: "Denoiser",
    detail: "Strips background noise before feature extraction.",
  },
  {
    title: "150+ features",
    detail: "MFCCs, ZCR, RMS energy, spectral flux, harmonic ratio, tempo.",
  },
  {
    title: "Ensemble classifier",
    detail: "Random Forest + Gradient Boosting vote together.",
  },
  {
    title: "Speech / no speech",
    detail: "Confidence score plus plain-English explanation.",
  },
];

const FEATURES = [
  { name: "MFCCs + deltas", count: "78 features", detail: "Spectral shape and change over time" },
  { name: "Zero crossing rate", count: "", detail: "Speech is more consistent than noise" },
  { name: "RMS energy pattern", count: "", detail: "Speech rises and falls rhythmically" },
  { name: "Spectral flux", count: "", detail: "Speech transitions smoothly, noise changes randomly" },
  { name: "Harmonic/percussive ratio", count: "", detail: "Human voice is mostly harmonic" },
  { name: "Tempo / rhythm", count: "", detail: "Speech has syllable rhythm, noise does not" },
  { name: "Mel spectrogram stats", count: "", detail: "Energy distribution across frequency bands" },
  { name: "Silence ratio", count: "", detail: "Proportion of frames below energy threshold" },
];

export function HowItWorks() {
  return (
    <section id="how-it-works" className="border-t border-border bg-surface/40 px-6 py-24 sm:py-32">
      <div className="mx-auto max-w-5xl">
        <Reveal className="mx-auto max-w-2xl text-center">
          <p className="text-sm font-medium uppercase tracking-widest text-primary">How it works</p>
          <h2 className="mt-3 text-balance font-display text-3xl font-semibold tracking-tight sm:text-4xl">
            One pipeline, five stages
          </h2>
          <p className="mt-4 text-balance text-muted">
            Every clip moves through the same explainable path — no black box in between.
          </p>
        </Reveal>

        <RevealGroup
          className="mt-16 grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-5 lg:gap-0"
          stagger={0.08}
        >
          {STEPS.map((step, i) => (
            <RevealItem key={step.title} className="relative flex flex-col items-center text-center">
              <div className="flex flex-col items-center gap-3 rounded-xl border border-border bg-surface px-5 py-6 lg:w-full">
                <span className="flex h-9 w-9 items-center justify-center rounded-full border border-primary/40 font-display text-sm font-semibold text-primary">
                  {i + 1}
                </span>
                <h3 className="font-display text-sm font-semibold text-foreground">{step.title}</h3>
                <p className="text-xs leading-relaxed text-muted">{step.detail}</p>
              </div>
              {i !== STEPS.length - 1 && (
                <svg
                  aria-hidden="true"
                  className="my-2 hidden text-border lg:block"
                  width="24"
                  height="16"
                  viewBox="0 0 24 16"
                  fill="none"
                >
                  <path
                    d="M1 8h20m0 0l-5-6m5 6l-5 6"
                    stroke="currentColor"
                    strokeWidth="1.5"
                    strokeLinecap="round"
                    strokeLinejoin="round"
                  />
                </svg>
              )}
            </RevealItem>
          ))}
        </RevealGroup>

        <Reveal delay={0.1} className="mt-24">
          <h3 className="text-center font-display text-xl font-semibold tracking-tight">
            150+ features extracted per file
          </h3>
          <RevealGroup
            className="mt-8 grid grid-cols-1 gap-3 sm:grid-cols-2 lg:grid-cols-4"
            stagger={0.05}
          >
            {FEATURES.map((f) => (
              <RevealItem key={f.name}>
                <div className="h-full rounded-lg border border-border bg-surface p-4 transition-colors duration-200 hover:border-primary/40">
                  <div className="flex items-start justify-between gap-2">
                    <span className="text-sm font-medium text-foreground">{f.name}</span>
                    {f.count && (
                      <span className="shrink-0 rounded-full bg-primary/10 px-2 py-0.5 text-[10px] font-medium text-primary">
                        {f.count}
                      </span>
                    )}
                  </div>
                  <p className="mt-2 text-xs leading-relaxed text-muted">{f.detail}</p>
                </div>
              </RevealItem>
            ))}
          </RevealGroup>
        </Reveal>
      </div>
    </section>
  );
}
