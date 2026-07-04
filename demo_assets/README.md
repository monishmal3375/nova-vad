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

## Regenerating

```bash
source venv/bin/activate
python3 -m src.pipeline          # train models if not already trained
python3 -m src.explainer demo_assets/<clip>.wav > /tmp/explain.json
```

The helper script used to build this folder's current contents pulls clips from
`data/speech/` (Speech Commands, safe to redistribute) and `data/clean_speech/` /
`data/clean_noise/` only where the underlying source license was verified above.
