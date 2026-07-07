"use client";

import { motion } from "framer-motion";

export function Logo({ size = 40, animated = true }: { size?: number; animated?: boolean }) {
  const waveTransition = animated
    ? { duration: 1.4, ease: [0.16, 1, 0.3, 1] as const }
    : { duration: 0 };

  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 100 100"
      fill="none"
      xmlns="http://www.w3.org/2000/svg"
      aria-hidden="true"
    >
      <circle cx="50" cy="50" r="48" fill="var(--surface)" />
      <circle cx="50" cy="50" r="47.5" stroke="var(--primary)" strokeWidth="1" opacity="0.9" />

      <motion.path
        d="M14 50 C 19 34, 24 34, 29 50 C 34 66, 39 40, 44 50"
        stroke="var(--wave-blue)"
        strokeWidth="2"
        strokeLinecap="round"
        fill="none"
        initial={animated ? { pathLength: 0, opacity: 0 } : undefined}
        animate={animated ? { pathLength: 1, opacity: 1 } : undefined}
        transition={waveTransition}
      />
      <motion.path
        d="M56 50 C 61 40, 66 66, 71 50 C 76 34, 81 34, 86 50"
        stroke="var(--wave-blue)"
        strokeWidth="2"
        strokeLinecap="round"
        fill="none"
        initial={animated ? { pathLength: 0, opacity: 0 } : undefined}
        animate={animated ? { pathLength: 1, opacity: 1 } : undefined}
        transition={{ ...waveTransition, delay: animated ? 0.15 : 0 }}
      />

      <circle cx="34" cy="34" r="1.6" fill="var(--primary)" opacity="0.7" />
      <circle cx="66" cy="34" r="1.6" fill="var(--wave-blue-dim)" opacity="0.7" />
      <circle cx="34" cy="66" r="1.6" fill="var(--wave-blue-dim)" opacity="0.7" />
      <circle cx="66" cy="66" r="1.6" fill="var(--primary)" opacity="0.7" />

      <circle cx="50" cy="50" r="21" fill="var(--surface-2)" stroke="var(--primary)" strokeWidth="1" opacity="0.95" />

      <text
        x="50"
        y="49"
        textAnchor="middle"
        fontFamily="var(--font-display), sans-serif"
        fontWeight="700"
        fontSize="13"
        fill="var(--primary)"
        letterSpacing="0.5"
      >
        NOVA
      </text>
      <text
        x="50"
        y="60"
        textAnchor="middle"
        fontFamily="var(--font-sans), sans-serif"
        fontWeight="500"
        fontSize="6"
        fill="var(--foreground)"
        letterSpacing="2.5"
      >
        VAD
      </text>
    </svg>
  );
}
