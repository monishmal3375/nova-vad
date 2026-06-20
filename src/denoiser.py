import os
import numpy as np
import soundfile as sf
import noisereduce as nr

def denoise_file(input_path: str, output_path: str) -> str:
    """
    Takes a raw .wav file, removes background noise,
    saves the cleaned version to output_path.
    Returns the output_path.
    """
    audio, sr = sf.read(input_path)

    # if stereo, convert to mono
    if audio.ndim > 1:
        audio = audio[:, 0]

    # use first 0.5 seconds as noise profile for reduction
    noise_sample = audio[:int(sr * 0.5)]
    reduced = nr.reduce_noise(y=audio, sr=sr, y_noise=noise_sample)
    sf.write(output_path, reduced, sr)
    return output_path

def denoise_folder(input_dir: str, output_dir: str) -> list:
    """
    Denoises all .wav files in input_dir,
    saves cleaned files to output_dir.
    Returns list of output file paths.
    """
    os.makedirs(output_dir, exist_ok=True)
    output_files = []

    files = [f for f in os.listdir(input_dir) if f.endswith(".wav")]
    print(f"Denoising {len(files)} files from {input_dir}...")

    for i, filename in enumerate(files):
        input_path  = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, filename)
        denoise_file(input_path, output_path)
        output_files.append(output_path)
        print(f"  ✓ [{i+1}/{len(files)}] {filename}")

    print(f"✅ Done. Cleaned files saved to {output_dir}/\n")
    return output_files