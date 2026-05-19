"""
Drop-in replacement for asr.py using mlx-whisper instead of NeMo.
Runs natively on Apple Silicon (M-series) via Metal — no CUDA required.

Usage (same interface as asr.py):
    python get_transcript/asr_mlx.py --root_dir /path/to/data/icc_backchannel
    python get_transcript/asr_mlx.py --root_dir /path/to/data/synthetic_user_interruption --task user_interruption
"""

import os
import json
import argparse
import tempfile
from glob import glob

import soundfile as sf
import mlx_whisper
from tqdm import tqdm

MODEL_NAME = "mlx-community/whisper-large-v3-mlx"


def get_time_aligned_transcription(data_path, task, audio_name="output.wav"):
    audio_paths = sorted(glob(f"{data_path}/*/{audio_name}"))
    json_name = audio_name.rsplit(".", 1)[0] + "_mlx.json"

    print(f"Found {len(audio_paths)} audio files.")

    for audio_path in tqdm(audio_paths):
        print(audio_path)
        waveform, sr = sf.read(audio_path)
        if waveform.ndim > 1:
            waveform = waveform.mean(axis=1)

        offset = 0.0

        if task == "user_interruption":
            meta_path = audio_path.replace(audio_name, "interrupt.json")
            with open(meta_path, "r") as f:
                interrupt_meta = json.load(f)
            _, end_interrupt = interrupt_meta[0]["timestamp"]
            offset = end_interrupt
            start_idx = int(end_interrupt * sr)
            waveform = waveform[start_idx:]

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            sf.write(tmp.name, waveform, sr)
            tmp_path = tmp.name

        result = mlx_whisper.transcribe(
            tmp_path,
            path_or_hf_repo=MODEL_NAME,
            word_timestamps=True,
        )
        os.unlink(tmp_path)

        chunks = []
        text = ""
        for segment in result.get("segments", []):
            for w in segment.get("words", []):
                start_time = w["start"] + offset
                end_time = w["end"] + offset
                word = w["word"].strip()
                text += word + " "
                chunks.append({
                    "text": word,
                    "timestamp": [start_time, end_time],
                })

        output_dict = {
            "text": text.strip(),
            "chunks": chunks,
        }

        result_path = audio_path.replace(audio_name, json_name)
        os.makedirs(os.path.dirname(result_path), exist_ok=True)
        with open(result_path, "w") as f:
            json.dump(output_dict, f, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Transcribe audio using mlx-whisper (Apple Silicon native)"
    )
    parser.add_argument("--root_dir", type=str, required=True)
    parser.add_argument(
        "--task",
        type=str,
        default="default",
        choices=["default", "user_interruption"],
    )
    parser.add_argument("--audio_name", type=str, default="output.wav")
    args = parser.parse_args()

    get_time_aligned_transcription(args.root_dir, args.task, args.audio_name)
