"use client";

import { useEffect, useState } from "react";
import { Logo } from "./Logo";

const LINKS = [
  { href: "#benchmarks", label: "Benchmarks" },
  { href: "#how-it-works", label: "How it works" },
  { href: "#explainability", label: "Explainability" },
  { href: "#live-demo", label: "Hear it work" },
  { href: "#quickstart", label: "Quickstart" },
  { href: "#about", label: "About" },
];

export function Nav() {
  const [scrolled, setScrolled] = useState(false);
  const [stars, setStars] = useState<number | null>(null);

  useEffect(() => {
    const onScroll = () => setScrolled(window.scrollY > 8);
    onScroll();
    window.addEventListener("scroll", onScroll, { passive: true });
    return () => window.removeEventListener("scroll", onScroll);
  }, []);

  useEffect(() => {
    // Real star count from the public GitHub API; silently omitted if
    // rate-limited or offline — never show a made-up number.
    fetch("https://api.github.com/repos/monishmal3375/nova-vad")
      .then((r) => (r.ok ? r.json() : null))
      .then((data) => {
        if (data && typeof data.stargazers_count === "number") {
          setStars(data.stargazers_count);
        }
      })
      .catch(() => {});
  }, []);

  return (
    <header
      className={`sticky top-0 z-50 w-full transition-colors duration-300 ${
        scrolled ? "bg-background/80 backdrop-blur-md border-b border-border" : "bg-transparent"
      }`}
    >
      <nav className="mx-auto flex max-w-6xl items-center justify-between px-6 py-4">
        <a href="#top" className="group flex min-h-11 items-center gap-2.5">
          <Logo size={32} animated={false} />
          <span className="font-display text-[15px] font-semibold tracking-tight text-foreground">
            NOVA<span className="text-primary">-VAD</span>
          </span>
        </a>

        <ul className="hidden items-center gap-8 md:flex">
          {LINKS.map((link) => (
            <li key={link.href}>
              <a
                href={link.href}
                className="text-sm text-muted transition-colors duration-200 hover:text-foreground"
              >
                {link.label}
              </a>
            </li>
          ))}
        </ul>

        <a
          href="https://github.com/monishmal3375/nova-vad"
          target="_blank"
          rel="noreferrer"
          className="inline-flex min-h-11 items-center gap-2 rounded-full border border-border bg-surface px-4 text-sm font-medium text-foreground transition-all duration-200 hover:border-primary/60 hover:bg-surface-2 cursor-pointer"
        >
          <svg width="16" height="16" viewBox="0 0 24 24" fill="currentColor" aria-hidden="true">
            <path d="M12 .5C5.65.5.5 5.65.5 12c0 5.09 3.29 9.4 7.86 10.93.57.1.78-.25.78-.55 0-.27-.01-1.17-.02-2.12-3.2.7-3.88-1.36-3.88-1.36-.52-1.34-1.28-1.69-1.28-1.69-1.04-.72.08-.7.08-.7 1.16.08 1.77 1.19 1.77 1.19 1.03 1.77 2.7 1.26 3.36.96.1-.75.4-1.26.73-1.55-2.55-.29-5.24-1.28-5.24-5.69 0-1.26.45-2.28 1.19-3.09-.12-.29-.52-1.46.11-3.05 0 0 .97-.31 3.18 1.18a11.1 11.1 0 0 1 2.9-.39c.98 0 1.97.13 2.9.39 2.2-1.49 3.17-1.18 3.17-1.18.63 1.59.23 2.76.11 3.05.74.81 1.19 1.83 1.19 3.09 0 4.42-2.69 5.4-5.25 5.68.41.36.78 1.07.78 2.15 0 1.55-.01 2.8-.01 3.18 0 .3.2.66.79.55A10.52 10.52 0 0 0 23.5 12c0-6.35-5.15-11.5-11.5-11.5Z" />
          </svg>
          <span className="hidden sm:inline">GitHub</span>
          {stars !== null && (
            <span className="hidden items-center gap-1 border-l border-border pl-2 font-mono text-xs text-muted sm:inline-flex">
              <svg width="11" height="11" viewBox="0 0 24 24" fill="currentColor" aria-hidden="true">
                <path d="M12 2l2.9 6.26L21.5 9.27l-4.75 4.37L17.9 20.5 12 17.27 6.1 20.5l1.15-6.86L2.5 9.27l6.6-1.01L12 2z" />
              </svg>
              {stars}
            </span>
          )}
        </a>
      </nav>
    </header>
  );
}
