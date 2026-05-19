"""
ASR Backend Comparison Evaluator.

Runs the standard evaluation pipeline for each ASR backend by temporarily
symlinking the backend-specific transcript (e.g. output_mlx.json) to
output.json, running evaluation, then restoring the original.

Usage:
    python evaluate_compare.py --task backchannel --root_dir /path/to/data/icc_backchannel
    python evaluate_compare.py --task pause_handling --root_dir /path/to/data/candor_pause_handling
    python evaluate_compare.py --task user_interruption --root_dir /path/to/data/synthetic_user_interruption

Backends compared: mlx (whisper-large-v3), assemblyai
"""

import argparse
import os
import shutil
from glob import glob


BACKENDS = {
    "mlx": "output_mlx.json",
    "assemblyai": "output_assemblyai.json",
}


def swap_transcript(root_dir: str, backend_json: str) -> list[str]:
    """Copy backend transcript to output.json in every sample dir. Returns list of paths touched."""
    sample_dirs = sorted(glob(f"{root_dir}/*/"))
    touched = []
    for d in sample_dirs:
        src = os.path.join(d, backend_json)
        dst = os.path.join(d, "output.json")
        if os.path.exists(src):
            shutil.copy2(src, dst)
            touched.append(dst)
    return touched


def run_evaluation(task: str, root_dir: str):
    """Import and run the appropriate eval function."""
    import sys
    sys.path.insert(0, os.path.dirname(__file__))

    if task == "backchannel":
        from eval_backchannel import eval_backchannel
        eval_backchannel(root_dir)
    elif task == "pause_handling":
        from eval_pause_handling import eval_pause_handling
        eval_pause_handling(root_dir)
    elif task == "smooth_turn_taking":
        from eval_smooth_turn_taking import eval_smooth_turn_taking
        eval_smooth_turn_taking(root_dir)
    elif task == "user_interruption":
        from dotenv import load_dotenv
        from openai import OpenAI
        load_dotenv()
        from eval_user_interruption import eval_user_interruption
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        eval_user_interruption(root_dir, client)


def main():
    parser = argparse.ArgumentParser(description="Compare ASR backends on evaluation metrics.")
    parser.add_argument("--task", type=str, required=True,
                        choices=["backchannel", "pause_handling", "smooth_turn_taking", "user_interruption"])
    parser.add_argument("--root_dir", type=str, required=True)
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f"Task: {args.task} | Root: {args.root_dir}")
    print(f"{'='*60}\n")

    for backend_name, backend_json in BACKENDS.items():
        # Check if this backend's transcripts exist
        sample_dirs = sorted(glob(f"{args.root_dir}/*/"))
        available = sum(1 for d in sample_dirs if os.path.exists(os.path.join(d, backend_json)))
        if available == 0:
            print(f"[{backend_name}] No transcripts found ({backend_json}) — skipping.\n")
            continue

        print(f"\n{'─'*60}")
        print(f"Backend: {backend_name.upper()} ({backend_json}) — {available} transcripts found")
        print(f"{'─'*60}")

        swap_transcript(args.root_dir, backend_json)
        run_evaluation(args.task, args.root_dir)

    print(f"\n{'='*60}")
    print("Comparison complete.")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
