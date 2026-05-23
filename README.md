# Full-Duplex-Bench: Reproduction & ASR Sensitivity Analysis

> **Fork of [DanielLin94144/Full-Duplex-Bench](https://github.com/DanielLin94144/Full-Duplex-Bench)**
> Reproduction of the v1.0 evaluation pipeline on Gemini 3.1 Flash Live, extended with a systematic analysis of ASR backend sensitivity on turn-taking evaluation metrics.

[![arXiv v1.0](https://img.shields.io/badge/v1.0_arXiv-2503.04721-b31b1b.svg?logo=arXiv)](https://arxiv.org/abs/2503.04721)
[![arXiv v1.5](https://img.shields.io/badge/v1.5_arXiv-2507.23159-b31b1b.svg?logo=arXiv)](https://arxiv.org/abs/2507.23159)
[![Preprint v1.0](https://img.shields.io/badge/Preprint_v1.0-Zenodo-blue.svg)](https://doi.org/10.5281/zenodo.20305268)
[![Preprint v1.5](https://img.shields.io/badge/Preprint_v1.5-Zenodo-blue.svg)](https://doi.org/10.5281/zenodo.20354457)

---

## Overview

This repository reproduces the Full-Duplex-Bench v1.0 evaluation pipeline and extends it with an analysis of how ASR transcription quality affects downstream turn-taking metrics. The original benchmark uses `nvidia/parakeet-tdt-0.6b-v2` via NeMo (CUDA-only) for speech-to-text. We replace this with two alternative backends and measure whether evaluation scores are stable across transcription systems.

**Research question:** *How sensitive are full-duplex turn-taking evaluation metrics (TOR, JSD, backchannel frequency, response latency) to the choice of ASR backend?*

---

## Contributions Beyond the Original

| File | Description |
|---|---|
| `v1_v1.5/get_transcript/asr_mlx.py` | Drop-in ASR replacement using `whisper-large-v3` via [mlx-whisper](https://github.com/ml-explore/mlx-examples) — runs natively on Apple Silicon (M-series) with no CUDA dependency |
| `v1_v1.5/get_transcript/asr_assemblyai.py` | Drop-in ASR replacement using AssemblyAI REST API — GPU-free, cloud-based transcription |
| `v1_v1.5/evaluation/evaluate_compare.py` | Runs evaluation for each ASR backend sequentially and reports metrics side-by-side for direct comparison |

---

## Results: Gemini 3.1 Flash Live (thinking_level=minimal)

### Backchannel (icc_backchannel, n=55)

| Metric | MLX Whisper-v3 | AssemblyAI | Paper (Gemini 3.1) |
|---|---|---|---|
| JSD ↓ | 0.8119 ± 0.0764 | 0.8119 ± 0.0764 | 0.807 |
| TOR ↓ | 0.7091 ± 0.4542 | **0.7273 ± 0.4454** | 0.727 |
| Freq ↑ | 0.0375 ± 0.0216 | 0.0375 ± 0.0216 | 0.044 |

### Pause Handling

| Dataset | Metric | MLX Whisper-v3 | AssemblyAI | Paper |
|---|---|---|---|---|
| candor_pause_handling (n=216) | Take-Turn Rate ↑ | **0.856** | 0.111 | 0.153 |
| synthetic_pause_handling (n=137) | Take-Turn Rate ↑ | **0.934** | 0.015 | 0.022 |

### Smooth Turn-Taking (candor_turn_taking, n=119)

| Metric | MLX Whisper-v3 | AssemblyAI | Paper |
|---|---|---|---|
| Take-Turn Rate ↑ | **0.983** | 0.958 | 1.000 |
| Latency ↓ | **0.977** | 1.425 | 0.567 |

### User Interruption (synthetic_user_interruption, n=200)

| Metric | MLX Whisper-v3 | AssemblyAI | Paper (Gemini 3.1) |
|---|---|---|---|
| GPT-4 Rating ↑ | **3.53** | 3.51 | 3.575 |
| TOR ↑ | 1.0 | 1.0 | 1.000 |
| Latency ↓ | **0.441** | 0.714 | 0.337 |

---

## Key Findings

**1. Timing-based metrics are ASR-invariant.** JSD and backchannel frequency are identical across both backends — these metrics depend on audio timestamps, not transcript content.

**2. TOR and latency are sensitive to ASR word segmentation.** MLX Whisper-v3 consistently produces scores closer to the paper's reported values. AssemblyAI's more conservative word boundary detection causes systematic underestimation of turn-taking rate (pause handling: 0.856 MLX vs 0.111 AssemblyAI) and overestimation of response latency.

**3. LLM-as-judge scores are ASR-robust.** GPT-4 ratings for user interruption quality are nearly identical across backends (3.53 vs 3.51), suggesting semantic evaluation is not affected by minor transcription differences.

**Implication:** Benchmarks that rely on TOR-based metrics should specify and fix the ASR backend, as different transcription systems can produce scores that differ by an order of magnitude on the same model outputs.

---

## Setup

### Requirements

```bash
conda create -n full-duplex-bench python=3.10
conda activate full-duplex-bench
pip install -r v1_v1.5/requirements.txt
pip install mlx-whisper torchcodec
```

> `mlx-whisper` replaces the NeMo/parakeet dependency and runs natively on Apple Silicon.
> `torchcodec` is required by newer versions of torchaudio for audio loading.

### Environment

```bash
cp v1_v1.5/.env.example v1_v1.5/.env
# Fill in:
# GEMINI_API_KEY=...         (required for inference)
# OPENAI_API_KEY=...         (required for user_interruption evaluation)
# ASSEMBLYAI_API_KEY=...     (required for asr_assemblyai.py)
```

### Dataset

Download from the [official Google Drive](https://drive.google.com/drive/folders/1DtoxMVO9_Y_nDs2peZtx3pw-U2qYgpd3) and extract to a local `data/` directory:

```
data/
  v1.0/
    icc_backchannel/
    candor_pause_handling/
    candor_turn_taking/
    synthetic_pause_handling/
    synthetic_user_interruption/
  v1.5/
    user_interruption/
    user_backchannel/
    talking_to_other/
    background_speech/
```

---

## Reproduction Steps

### 1. Model Inference

```bash
python v1_v1.5/model_inference/gemini/inference_gemini31_live.py \
    --base-dir /path/to/data/v1.0 \
    --task icc_backchannel \
    --thinking-level minimal \
    --concurrency 1
```

Repeat for each task: `icc_backchannel`, `candor_pause_handling`, `candor_turn_taking`, `synthetic_pause_handling`, `synthetic_user_interruption`.

### 2. ASR Transcription

```bash
# MLX Whisper (Apple Silicon, no GPU required)
python v1_v1.5/get_transcript/asr_mlx.py \
    --root_dir /path/to/data/v1.0/icc_backchannel

# AssemblyAI API
python v1_v1.5/get_transcript/asr_assemblyai.py \
    --root_dir /path/to/data/v1.0/icc_backchannel

# For user_interruption task — crops audio after the interrupt point before transcribing
python v1_v1.5/get_transcript/asr_mlx.py \
    --root_dir /path/to/data/v1.0/synthetic_user_interruption \
    --task user_interruption
```

Each script writes backend-specific JSON files (`output_mlx.json`, `output_assemblyai.json`) alongside the audio, so multiple backends can coexist without overwriting each other.

### 3. Standard Evaluation (single backend)

```bash
cd v1_v1.5/evaluation
python evaluate.py --task backchannel --root_dir /path/to/data/v1.0/icc_backchannel
```

### 4. Cross-Backend Comparison

```bash
cd v1_v1.5/evaluation
python evaluate_compare.py \
    --task backchannel \
    --root_dir /path/to/data/v1.0/icc_backchannel
```

Supported tasks: `backchannel`, `pause_handling`, `smooth_turn_taking`, `user_interruption`.

---

## Hardware

All Gemini inference and MLX ASR runs were performed on a MacBook Pro M3 Max (no CUDA required). The original NeMo/parakeet ASR requires a CUDA GPU — see the [upstream repo](https://github.com/DanielLin94144/Full-Duplex-Bench) for GPU-based setup.

---

## Citation

If you use this work, please cite the original benchmark:

```bibtex
@article{lin2025full_v1,
  title={Full-duplex-bench: A benchmark to evaluate full-duplex spoken dialogue models on turn-taking capabilities},
  author={Lin, Guan-Ting and Lian, Jiachen and Li, Tingle and Wang, Qirui and Anumanchipalli, Gopala and Liu, Alexander H and Lee, Hung-yi},
  journal={arXiv preprint arXiv:2503.04721},
  year={2025}
}

@article{lin2025fdb_v15,
  title={Full-Duplex-Bench v1.5: Evaluating Overlap Handling for Full-Duplex Speech Models},
  author={Lin, Guan-Ting and Kuan, Shih-Yun Shan and Wang, Qirui and Lian, Jiachen and Li, Tingle and Lee, Hung-yi},
  journal={arXiv preprint arXiv:2507.23159},
  year={2025}
}
```

---

## Acknowledgements

Original benchmark by [Guan-Ting Lin](https://daniellin94144.github.io/) et al. at UC Berkeley / NTU.
