"""
Drop-in replacement for asr.py using AssemblyAI API.
No local GPU required — audio is uploaded and transcribed via REST API.
Free tier: 100 hours. Sufficient for the full benchmark dataset.

Usage (same interface as asr.py):
    python get_transcript/asr_assemblyai.py --root_dir /path/to/data/icc_backchannel
    python get_transcript/asr_assemblyai.py --root_dir /path/to/data/synthetic_user_interruption --task user_interruption

Requires ASSEMBLYAI_API_KEY in v1_v1.5/.env
"""

import os
import json
import argparse
import tempfile
import time
from glob import glob

import soundfile as sf
import requests
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv(os.path.join(os.path.dirname(__file__), "../.env"))

API_KEY = os.getenv("ASSEMBLYAI_API_KEY")
if not API_KEY:
    raise ValueError("ASSEMBLYAI_API_KEY not found in environment. Check v1_v1.5/.env")

UPLOAD_URL = "https://api.assemblyai.com/v2/upload"
TRANSCRIPT_URL = "https://api.assemblyai.com/v2/transcript"
HEADERS = {"authorization": API_KEY}


def upload_audio(file_path: str) -> str:
    with open(file_path, "rb") as f:
        response = requests.post(UPLOAD_URL, headers=HEADERS, data=f)
    response.raise_for_status()
    return response.json()["upload_url"]


def request_transcript(upload_url: str) -> str:
    payload = {
        "audio_url": upload_url,
        "punctuate": True,
        "format_text": False,
    }
    response = requests.post(TRANSCRIPT_URL, headers=HEADERS, json=payload)
    response.raise_for_status()
    return response.json()["id"]


def poll_transcript(transcript_id: str) -> dict:
    url = f"{TRANSCRIPT_URL}/{transcript_id}"
    while True:
        response = requests.get(url, headers=HEADERS)
        response.raise_for_status()
        result = response.json()
        status = result["status"]
        if status == "completed":
            return result
        elif status == "error":
            raise RuntimeError(f"AssemblyAI transcription error: {result.get('error')}")
        time.sleep(2)


def get_time_aligned_transcription(data_path, task, audio_name="output.wav"):
    audio_paths = sorted(glob(f"{data_path}/*/{audio_name}"))
    json_name = audio_name.rsplit(".", 1)[0] + "_assemblyai.json"

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

        try:
            upload_url = upload_audio(tmp_path)
            transcript_id = request_transcript(upload_url)
            result = poll_transcript(transcript_id)
        finally:
            os.unlink(tmp_path)

        chunks = []
        text = ""
        for w in result.get("words", []):
            start_time = w["start"] / 1000.0 + offset  # AssemblyAI uses milliseconds
            end_time = w["end"] / 1000.0 + offset
            word = w["text"].strip()
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
        description="Transcribe audio using AssemblyAI API"
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
