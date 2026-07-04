# demo_assets/

Precomputed "hear it work" demo clips for the NOVA-VAD marketing site. Each clip is
paired with the real `src/explainer.py` JSON output (confidence + ranked feature
drivers) computed once, offline, against the trained model in `models/`. The website
is expected to play the audio and render the precomputed explanation in sync — this
is **not** a live-inference demo.

## Files

For each demo clip `X`:
- `X.wav` — the audio clip (a few seconds)
- `X.explanation.json` — the exact output of `python3 -m src.explainer X.wav`, i.e.
  `{"file", "label", "confidence", "top_features": [...]}`

See `manifest.json` for the full list of clips, their source, and their noise category.

## Licensing — read before publishing these publicly

Audio clips in this folder are only included when their source license clearly
permits public redistribution:

- **Google Speech Commands v0.02** clips (`data/speech/`, `data/tmp/speech_commands`
  during `download_data.py`) are released under **CC BY 4.0** by Google
  (https://creativecommons.org/licenses/by/4.0/, dataset:
  http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz). CC BY 4.0
  permits redistribution, including commercially, with attribution. Any speech clip
  in this folder sourced from Speech Commands is safe to publish with attribution:
  "Speech clip from the Google Speech Commands Dataset, CC BY 4.0."

- **UrbanSound8K** noise clips (used elsewhere in this repo via `download_noise.py`
  for benchmarking) are **deliberately excluded from this folder**. UrbanSound8K
  does not grant a single blanket license for the compiled dataset — each clip
  retains the license of its original Freesound.org upload (which varies per file:
  CC0, CC BY, CC BY-NC, or Sampling+), and the standard `UrbanSound8K.csv` metadata
  does not include a per-file license column. Redistributing an UrbanSound8K clip
  on a public marketing site would require looking up that specific clip's `fsID`
  on freesound.org and confirming its license permits redistribution — this was not
  done here, so no UrbanSound8K audio is shipped in `demo_assets/`.

- If you want noise-category demo clips with clean redistribution rights, prefer
  **MUSAN** (CC BY 4.0 / US public domain, per-file LICENSE manifests bundled with
  the corpus, explicitly designed for redistribution — see
  https://www.openslr.org/17/) over UrbanSound8K, or record/license your own noise
  clips. This repo's `download_data.py` already pulls MUSAN noise/music clips for
  training — those would be the safer public-redistribution source if you want a
  noisy (not just quiet) clip on the marketing site.

**Bottom line:** every `.wav` in this folder must have a verifiable redistributable
license before publishing. Don't add new UrbanSound8K clips here without personally
checking that specific clip's Freesound license first.

### Real-world urban noise clips added 2026-07-03 (`noise_demo_4/5/6`)

The site's marketing copy describes "traffic, sirens, construction, AC hum" but the
original demo clips above (white noise / pink noise / dishes) don't back that up —
none of them are actual urban environmental noise. To fix the mismatch, three
individually-verified **CC0 1.0 (Public Domain)** Freesound.org clips were added,
each checked on its own sound page (not assumed clear from a dataset or pack-level
license):

| Clip | Category | Freesound source | License | Verified |
|---|---|---|---|---|
| `noise_demo_4.wav` | siren | ["Distant ambulance & fire truck sirens (Germany)" by Breviceps](https://freesound.org/people/Breviceps/sounds/535776/) | CC0 1.0 | Checked the sound's own license page directly; real field recording (recorded on a mobile phone), not synthesized |
| `noise_demo_5.wav` | construction / jackhammer | ["Jackhammer.WAV" by mindgraveyard](https://freesound.org/people/mindgraveyard/sounds/511509/) | CC0 1.0 | Checked the sound's own license page directly; real field recording (Zoom H4 + SM58) |
| `noise_demo_6.wav` | traffic | ["Passing Car (Wet road)" by Breviceps](https://freesound.org/people/Breviceps/sounds/462862/) | CC0 1.0 | Checked the sound's own license page directly; real field recording of a car passing at ~100km/h |

CC0 means no attribution is legally required, but the source URLs are kept in
`manifest.json` anyway for traceability.

**What was deliberately excluded, and why:**
- A CC0-licensed "City Traffic Ambience" clip by DataJuggler
  (https://freesound.org/people/DataJuggler/sounds/750144/) was found and is CC0, but
  its own description says it's an AI-generated soundscape (Stable Audio / Pinokio 2),
  not a real recording. Since the whole point of these clips is to back up "real-world
  traffic, sirens, construction" marketing copy with actual real-world noise, a
  synthetic clip would undermine that claim even though its license is clean. Skipped.
- A real traffic field recording, ["Far Away City Traffic Ambience" by
  Dpoggioli](https://freesound.org/people/Dpoggioli/sounds/317271/), and a real
  construction recording, ["Construction, Jackhammer Excavator, A.wav" by
  InspectorJ](https://freesound.org/people/InspectorJ/sounds/400991/), were both found
  and are genuine field recordings — but both are **CC BY 4.0, not CC0** (attribution
  required). They were not used here only because a CC0 alternative was available for
  each category; either would also be safe to add with attribution if more clips are
  wanted later (same standard already applied to the Speech Commands CC BY 4.0 clips
  above).
- A traffic-focused Freesound pack by Robinhood76 was found but the pack listing page
  didn't expose a clear per-file license for each individual sound, and per-file
  license pages weren't checked for every file in it — skipped rather than guessing.
- BBC Sound Effects library was considered per the original ask, but its free-tier
  RemArc license is restricted to "personal, educational, or research" use — it does
  not clear commercial redistribution on a public marketing site, so no BBC clips were
  used.

**Provenance caveat — read before treating these as pristine source audio:** Freesound
requires an authenticated account (via their API) to download a sound's original
uploaded file, and no such credentials were available when these clips were sourced.
What's actually in `noise_demo_4/5/6.wav` was extracted from Freesound's **public,
unauthenticated preview stream** for each sound (the same 128kbps MP3 preview embedded
in the sound's public web page — e.g.
`https://cdn.freesound.org/previews/535/535776_9159316-hq.mp3`), then downsampled to
16kHz mono WAV to match this folder's format. The preview stream is served under the
same license as the original file (it's Freesound's own public playback copy, not a
third-party re-encode), so the CC0 status holds — but it means these three clips are a
lossy transcode of the original recording, not the original bit-for-bit master. If you
have Freesound API credentials, consider re-downloading the original masters and
re-running the pipeline below.

Trim/resample notes:
- `noise_demo_4.wav`: first 4.0s of the 18.6s original (siren is loudest at the start).
- `noise_demo_5.wav`: 4.0s window from 13.0s-17.0s of the ~114s original — the earlier
  part of the recording (checked via per-second RMS) is comparatively quiet/idle, this
  window captures the jackhammer actively running.
- `noise_demo_6.wav`: the original is only 5.76s total; used the first 4.0s, which
  includes the loudest part of the pass-by (peaks around 2-3s in).

## Regenerating

```bash
source venv/bin/activate
python3 -m src.pipeline          # train models if not already trained
python3 -m src.explainer demo_assets/<clip>.wav > /tmp/explain.json
```

The helper script used to build this folder's current contents pulls clips from
`data/speech/` (Speech Commands, safe to redistribute) and `data/clean_speech/` /
`data/clean_noise/` only where the underlying source license was verified above.
