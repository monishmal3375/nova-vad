"use client";

import { Bug, FlaskConical, Terminal } from "lucide-react";
import { Reveal, RevealGroup, RevealItem } from "./Reveal";

const STEPS = [
  {
    icon: Bug,
    title: "We found our own bug",
    body: "While benchmarking, a naive volume-threshold baseline suspiciously beat trained models. We dug in: our denoising step was accidentally erasing noise clips' own energy before evaluation. We fixed the pipeline instead of shipping the flattering number.",
  },
  {
    icon: FlaskConical,
    title: "Methodology before results",
    body: "Held-out test set carved off first and never touched during tuning. Stratified across every noise category. Cross-validation reported as a distribution, not a cherry-picked single run. Every model sees the identical test files.",
  },
  {
    icon: Terminal,
    title: "Re-run every number yourself",
    body: "The full benchmark — NOVA-VAD and every baseline — is one command in the open repo. If your numbers don't match ours, open an issue. Hard failure cases are especially welcome.",
  },
];

export function Methodology() {
  return (
    <section id="methodology" className="border-t border-border bg-surface/40 px-6 py-24 sm:py-32">
      <div className="mx-auto max-w-5xl">
        <Reveal className="mx-auto max-w-2xl text-center">
          <p className="text-sm font-medium uppercase tracking-widest text-primary">
            Benchmarked honestly
          </p>
          <h2 className="mt-3 text-balance font-display text-3xl font-semibold tracking-tight sm:text-4xl">
            Numbers you can check, not just trust
          </h2>
          <p className="mt-4 text-balance text-muted">
            Most model benchmarks ask for your faith. Ours asks for five minutes and a
            terminal.
          </p>
        </Reveal>

        <RevealGroup className="mt-14 grid grid-cols-1 gap-4 md:grid-cols-3" stagger={0.08}>
          {STEPS.map((step) => (
            <RevealItem key={step.title}>
              <div className="h-full rounded-2xl border border-border bg-surface p-6 transition-colors duration-200 hover:border-primary/40">
                <step.icon className="h-5 w-5 text-primary" aria-hidden="true" />
                <h3 className="mt-4 font-display text-base font-semibold text-foreground">
                  {step.title}
                </h3>
                <p className="mt-2 text-sm leading-relaxed text-muted">{step.body}</p>
              </div>
            </RevealItem>
          ))}
        </RevealGroup>
      </div>
    </section>
  );
}
