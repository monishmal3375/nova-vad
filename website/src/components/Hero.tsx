"use client";

import { motion } from "framer-motion";
import { Logo } from "./Logo";
import { WaveformBackdrop } from "./WaveformBackdrop";

const fadeUp = {
  hidden: { opacity: 0, y: 20 },
  visible: (delay: number) => ({
    opacity: 1,
    y: 0,
    transition: { duration: 0.7, delay, ease: [0.16, 1, 0.3, 1] as const },
  }),
};

export function Hero() {
  return (
    <section id="top" className="relative isolate flex flex-col items-center overflow-hidden px-6 pb-24 pt-20 sm:pt-28">
      <div aria-hidden="true" className="blueprint-grid pointer-events-none absolute inset-0" />
      <WaveformBackdrop />

      <motion.div initial="hidden" animate="visible" custom={0} variants={fadeUp}>
        <Logo size={72} />
      </motion.div>

      <motion.div
        initial="hidden"
        animate="visible"
        custom={0.15}
        variants={fadeUp}
        className="mt-8 inline-flex items-center gap-2 rounded-full border border-border bg-surface px-4 py-1.5 text-xs font-medium text-muted"
      >
        <span className="h-1.5 w-1.5 rounded-full bg-primary shadow-[0_0_8px_var(--primary)]" />
        Open source &middot; MIT licensed &middot; no GPU required
      </motion.div>

      <motion.h1
        initial="hidden"
        animate="visible"
        custom={0.25}
        variants={fadeUp}
        className="mt-6 max-w-3xl text-balance text-center font-display text-4xl font-semibold leading-[1.1] tracking-tight text-foreground sm:text-6xl"
      >
        Voice activity detection that{" "}
        <span className="text-primary glow-text">works in real noise</span>
      </motion.h1>

      <motion.p
        initial="hidden"
        animate="visible"
        custom={0.35}
        variants={fadeUp}
        className="mt-6 max-w-xl text-balance text-center text-base leading-relaxed text-muted sm:text-lg"
      >
        NOVA-VAD is a lightweight, explainable voice activity detector built for noisy
        real-world audio &mdash; traffic, sirens, construction, AC hum. 99.8% accuracy, every
        decision explained, no GPU required.
      </motion.p>

      <motion.div
        initial="hidden"
        animate="visible"
        custom={0.45}
        variants={fadeUp}
        className="mt-10 flex flex-col items-center gap-4 sm:flex-row"
      >
        <a
          href="https://github.com/monishmal3375/nova-vad"
          target="_blank"
          rel="noreferrer"
          className="group inline-flex items-center gap-2 rounded-full bg-primary px-6 py-3 text-sm font-semibold text-background transition-all duration-200 hover:bg-primary-dim hover:shadow-[0_0_24px_rgba(76,211,245,0.35)] cursor-pointer"
        >
          View on GitHub
          <svg
            width="14"
            height="14"
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            strokeWidth="2.5"
            className="transition-transform duration-200 group-hover:translate-x-0.5"
          >
            <path d="M5 12h14M13 6l6 6-6 6" strokeLinecap="round" strokeLinejoin="round" />
          </svg>
        </a>
        <a
          href="https://huggingface.co/monishmal0204/nova-vad"
          target="_blank"
          rel="noreferrer"
          className="inline-flex items-center gap-2 rounded-full border border-border bg-surface px-6 py-3 text-sm font-semibold text-foreground transition-all duration-200 hover:border-primary/60 hover:bg-surface-2 cursor-pointer"
        >
          Try on Hugging Face
        </a>
      </motion.div>

    </section>
  );
}
