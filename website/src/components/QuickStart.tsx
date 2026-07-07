"use client";

import { useState } from "react";
import { Reveal } from "./Reveal";
import { CodeBlock } from "./CodeBlock";

const TABS = [
  {
    id: "install",
    label: "Install",
    code: `git clone https://github.com/monishmal3375/nova-vad.git
cd nova-vad
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python3 download_data.py
python3 -m src.pipeline`,
  },
  {
    id: "explain",
    label: "Explain a prediction",
    code: `# after the pipeline has run once and saved local models
python3 -m src.explainer path/to/your_audio.wav`,
  },
  {
    id: "benchmark",
    label: "Run benchmark",
    code: `python3 -m src.benchmark`,
  },
  {
    id: "stream",
    label: "Realtime streaming",
    code: `# for better streaming behavior, calibrate first
python3 retrain_streaming.py
python3 -m src.stream`,
  },
];

export function QuickStart() {
  const [active, setActive] = useState(TABS[0].id);
  const activeTab = TABS.find((t) => t.id === active)!;

  return (
    <section id="quickstart" className="border-t border-border bg-surface/40 px-6 py-24 sm:py-32">
      <div className="mx-auto max-w-3xl">
        <Reveal className="text-center">
          <p className="text-sm font-medium uppercase tracking-widest text-primary">Quickstart</p>
          <h2 className="mt-3 text-balance font-display text-3xl font-semibold tracking-tight sm:text-4xl">
            Running in under five minutes
          </h2>
          <p className="mt-4 text-balance text-muted">
            No account, no API key, no GPU. Clone it, train once locally, start predicting.
          </p>
        </Reveal>

        <Reveal delay={0.1} className="mt-10">
          <div className="mb-4 flex flex-wrap justify-center gap-2">
            {TABS.map((tab) => (
              <button
                key={tab.id}
                onClick={() => setActive(tab.id)}
                className={`inline-flex min-h-11 cursor-pointer items-center rounded-full px-4 text-xs font-medium transition-colors duration-200 ${
                  active === tab.id
                    ? "bg-primary text-background"
                    : "border border-border bg-surface text-muted hover:text-foreground"
                }`}
              >
                {tab.label}
              </button>
            ))}
          </div>
          <CodeBlock code={activeTab.code} label={`${activeTab.label.toLowerCase()}.sh`} />
        </Reveal>
      </div>
    </section>
  );
}
