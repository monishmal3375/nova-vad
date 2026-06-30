# NOVA-VAD X Launch Pack

Account: https://x.com/Nova_vad

Use these as drafts. Edit for your voice before posting.

## Pinned Launch Post

Launching NOVA-VAD: a lightweight, explainable voice activity detector for noisy real-world audio.

Current benchmark:
- NOVA-VAD: 93.0% accuracy / 92.63 F1
- Silero: 87.0% / 87.13 F1
- Pyannote: 62.0% / 71.21 F1
- WebRTC: 58.0% / 58.82 F1

Built for ASR, diarization, voice agents, edge audio, and anyone who needs to know when speech is actually present.

GitHub: https://github.com/monishmal3375/nova-vad
Hugging Face: https://huggingface.co/monishmal0204/nova-vad

If you have hard noisy audio, send it. I want the failure cases.

## Launch Thread 1: Product

1. Launching NOVA-VAD: a lightweight, explainable voice activity detector for noisy real-world audio.

2. VAD looks simple until you put it near traffic, fans, sirens, construction noise, bad mics, and realtime voice-agent audio.

3. Most tools force a tradeoff: lightweight, accurate, or explainable. NOVA-VAD is my attempt to push all three together.

4. Current held-out noisy-audio benchmark:
WebRTC: 58.0% accuracy / 58.82 F1
Pyannote: 62.0% / 71.21 F1
Silero: 87.0% / 87.13 F1
NOVA-VAD: 93.0% / 92.63 F1

5. The pipeline is simple:
raw audio -> denoiser -> 150+ audio features -> ensemble classifier -> speech/no speech + explanation

6. Every prediction includes confidence plus the top features that drove the decision.

7. Repo is MIT licensed. Star/fork if you build ASR, diarization, voice agents, or noisy-audio pipelines.

GitHub: https://github.com/monishmal3375/nova-vad
HF: https://huggingface.co/monishmal0204/nova-vad

## Launch Thread 2: Technical

1. NOVA-VAD detects speech without depending on a deep neural net at inference time.

2. It extracts 150+ temporal and spectral features: MFCCs, deltas, ZCR, RMS patterns, spectral flux, harmonic/percussive ratio, rhythm, mel statistics, and silence ratio.

3. That feature vector goes into a Random Forest + Gradient Boosting ensemble.

4. The reason I like this approach: when the model says speech/no speech, it can also show which audio features mattered most.

5. This matters in production. If VAD fails, everything downstream gets worse: ASR, diarization, call analytics, agents, latency, and token spend.

6. I am looking for hard noisy samples and contributors for streaming support, packaging, and more benchmarks.

GitHub: https://github.com/monishmal3375/nova-vad

## Launch Thread 3: Builder Ask

1. If you build voice agents, ASR, diarization, or call transcription, I want your hardest noisy audio cases.

2. I built NOVA-VAD because VAD failures are usually invisible until the rest of the pipeline starts acting weird.

3. Bad VAD means clipped speech, false speech segments, wasted compute, worse transcripts, and strange realtime agent behavior.

4. NOVA-VAD is lightweight, explainable, retrainable, and MIT licensed.

5. Try it on your own noisy dataset and open an issue with the result. Especially if it fails.

6. Stars/forks help, but failure cases help even more.

GitHub: https://github.com/monishmal3375/nova-vad
HF: https://huggingface.co/monishmal0204/nova-vad

## Single Posts

Most VADs are accurate, lightweight, or explainable.

NOVA-VAD is my attempt to make all three work together for noisy real-world audio.

Current benchmark: 93.0% accuracy / 92.63 F1 on held-out noisy audio.

https://github.com/monishmal3375/nova-vad

---

VAD is the first quality gate in a voice pipeline.

If it gets speech boundaries wrong, your ASR, diarization, call analytics, and voice agent all inherit the mistake.

That is why I built NOVA-VAD to return a prediction, confidence, and explanation.

---

Open-source request:

Send NOVA-VAD your hardest noisy speech samples.

Traffic, fans, music bleed, bad mics, car audio, construction, overlapping sound.

I want to publish the failures and improve the benchmark.

---

NOVA-VAD pipeline:

raw audio -> denoiser -> 150+ features -> RF + GBT ensemble -> speech/no speech -> explanation

Simple, inspectable, and built for noisy audio.

---

Silero is strong. WebRTC is lightweight. Pyannote is widely used.

NOVA-VAD is trying to win a different wedge: noisy-audio performance + lightweight deployment + explainable decisions.

## Reply Templates

When someone discusses ASR/VAD/audio noise:

> This is exactly where VAD quality matters. A false speech segment can make the whole downstream ASR/agent pipeline look worse. I have been testing an explainable VAD on noisy audio and the failure cases are the most useful part.

When someone asks for tools:

> If you want an open-source option to test, I am building NOVA-VAD. It is lightweight and returns confidence + feature explanations. I would compare it against Silero/WebRTC on your own audio before trusting any benchmark.

When someone posts a noisy audio challenge:

> This would be a great VAD stress test. If you can share a short non-private clip, I would like to run NOVA-VAD and publish whether it passes or fails.
