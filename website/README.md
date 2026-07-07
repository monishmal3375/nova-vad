# NOVA-VAD website

Marketing/demo site for [NOVA-VAD](../README.md) — Next.js 16 (App Router) + Tailwind CSS v4 + Framer Motion.

## Sections

- Hero with animated waveform backdrop and a live GitHub star count
- Bento-grid feature overview
- Animated benchmark comparison (accuracy bars + full precision/recall/F1 table vs. every baseline)
- A reproduce-it-yourself callout with a copyable benchmark command
- "Benchmarked honestly" section — the actual bug-found-and-fixed story
- 5-stage architecture pipeline diagram
- Explainability showcase with a worked example
- **"Hear it work"** — an interactive audio demo. Pick a real clip (speech, traffic, siren, construction, etc.), play it, and see NOVA-VAD's real, precomputed prediction (confidence + feature drivers) alongside a live Web Audio frequency/waveform visualizer reacting to the actual playing audio. Predictions are precomputed real model output (see `../demo_assets/`), not live in-browser inference — stated explicitly on the page.
- Quickstart code tabs
- Contact CTA

## Running locally

```bash
npm install
npm run dev
```

Open [http://localhost:3000](http://localhost:3000).

## Stack

Next.js 16, Tailwind CSS v4, Framer Motion, lucide-react. Dark navy/electric-cyan theme matching the NOVA-VAD logo. See `src/app/globals.css` for design tokens.

## Deployment

Not yet deployed to a public URL — this runs locally only for now.
