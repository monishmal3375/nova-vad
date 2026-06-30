# Contributing to NOVA-VAD

Thanks for helping improve NOVA-VAD. The most valuable contributions right now are real noisy-audio tests, benchmark additions, and packaging improvements.

## Best First Contributions

- Run NOVA-VAD on your own noisy speech clips and open an issue with the result.
- Add a failing audio case that should be classified differently.
- Improve the quickstart or Hugging Face model card.
- Add tests around feature extraction and explanation output.
- Help package the project for `pip install nova-vad`.

## Reporting A Noisy-Audio Result

Please include:

- What kind of audio it is: call audio, voice-agent recording, podcast, car noise, street noise, etc.
- Clip duration and sample rate.
- Expected label: `SPEECH` or `NO SPEECH`.
- NOVA-VAD output and confidence.
- Whether another VAD handled it better or worse.

Do not upload private calls or personal recordings without permission. A short synthetic or anonymized clip is enough.

## Benchmark Contributions

If you add or change benchmark results, include:

- Dataset/source description.
- Train/test split method.
- Number of files tested.
- Exact commands used.
- Accuracy, precision, recall, F1, and any failure notes.

Please keep benchmark claims scoped to the tested setup.

## Development Setup

```bash
git clone https://github.com/monishmal3375/nova-vad.git
cd nova-vad
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python3 download_data.py
python3 -m src.pipeline
```

## Pull Request Checklist

- The change is focused on one clear improvement.
- README or docs are updated when behavior changes.
- New benchmark claims include methodology.
- No private audio or secrets are committed.
