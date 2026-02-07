# vlmbench

[![PyPI version](https://badge.fury.io/py/vlmbench.svg)](https://pypi.org/project/vlmbench/)
[![PyPI - Downloads](https://img.shields.io/pypi/dm/vlmbench)](https://pypi.org/project/vlmbench/)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://github.com/vlm-run/vlmbench/blob/main/LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Discord](https://dcbadge.limes.pink/api/server/https://discord.gg/x3F6pBQZMY?style=flat)](https://discord.gg/x3F6pBQZMY)
[![Twitter](https://img.shields.io/twitter/follow/vlaborai)](https://x.com/vlaborai)

Single-file, drop-in VLM benchmark CLI for your agents. One command, one JSON, stackable into a leaderboard.

Benchmark any vision-language model on your own hardware with a single command. vlmbench auto-detects your platform, starts the right backend, and gives you reproducible results as JSON.

- **macOS** — Ollama (auto-starts, zero config)
- **Linux** — vLLM via Docker (`--gpus all`, auto-pulls) or native vLLM
- **SGLang** — coming soon

<img width="2468" height="1920" alt="image" src="https://github.com/user-attachments/assets/24f8ecca-6d46-49d4-80bc-df0a42a9c326" />

## Quick Start

No install needed. Just run with `uvx`:

```bash
# macOS (Ollama — auto-starts, auto-pulls the model)
uvx vlmbench run -m qwen3-vl:2b -i ./images/

# Linux (vLLM Docker — auto-starts with --gpus all)
uvx vlmbench run -m Qwen/Qwen3-VL-8B-Instruct -i ./images/

# Linux (native vLLM — requires vllm installed)
uvx vlmbench run -m Qwen/Qwen3-VL-8B-Instruct -i ./images/ --backend vllm
```

Or install it:

```bash
pip install vlmbench
```

## Example Run

```
╭─ Configuration ──────────────────────────────────────────────────────────────╮
│                                                                              │
│   Model      lightonai/LightOnOCR-2-1B @ main                                │
│   Server     http://localhost:8000/v1 • vLLM 0.15.1                          │
│   Hardware   NVIDIA RTX PRO 6000 • CUDA • 95.59 GB VRAM                      │
│   Input      ./docs/ -> 62 inputs (43 images, 19 PDF pages)                  │
│   Config     max_tokens=2048 • runs=3 • concurrency=8                        │
│   Tmux       vlmbench-vllm • tmux attach -t vlmbench-vllm                    │
│                                                                              │
╰──────────────────────────────────────────────────────────────────────────────╯

╭─ Results ────────────────────────────────────────────────────────────────────╮
│                                                                              │
│   TTFT           467 ms    (p95: 1975 ms)                                    │
│   TPOT           6.0 ms    (p95: 6.2 ms)                                     │
│   Throughput   1664.8 tok/s   9.20 images/s                                  │
│   Latency        0.87 s/img  (p95: 3.55 s)                                   │
│   Tokens          270 prompt    181 completion (avg)                         │
│   Reliability  186/186 ok, 0 retries                                         │
│                                                                              │
╰──────────────────────────────────────────────────────────────────────────────╯
  > Saved -> results/lightonocr-2-1b-20260207T104621.json
```

## Compare

```bash
vlmbench compare results/*.json
```

```
╭───────────────────────────────┬──────────┬──────────┬─────────┬────────┬──────────────┬─────────────┬──────────┬────────────┬──────────────────────────────────────────────────────╮
│                               │     TTFT │     TPOT │         │        │ Duration (s) │ num_workers │     VRAM │            │                                                      │
│ Model                         │     (ms) │     (ms) │ Tok/s ↓ │  Img/s │              │             │          │ Backend    │ Hardware                                             │
├───────────────────────────────┼──────────┼──────────┼─────────┼────────┼──────────────┼─────────────┼──────────┼────────────┼──────────────────────────────────────────────────────┤
│ lightonai/LightOnOCR-2-1B     │      467 │      6.0 │  1664.8 │   9.20 │        162.4 │           8 │  5.78 GB │ vLLM 0.15.1│ NVIDIA RTX PRO 6000 Blackwell Workstation Edition   │
├───────────────────────────────┼──────────┼──────────┼─────────┼────────┼──────────────┼─────────────┼──────────┼────────────┼──────────────────────────────────────────────────────┤
│ rednote-hilab/dots.ocr        │     1424 │     10.2 │   477.6 │   7.76 │        190.8 │           8 │  9.42 GB │ vLLM 0.15.1│ NVIDIA RTX PRO 6000 Blackwell Workstation Edition   │
├───────────────────────────────┼──────────┼──────────┼─────────┼────────┼──────────────┼─────────────┼──────────┼────────────┼──────────────────────────────────────────────────────┤
│ Qwen/Qwen3-VL-8B-Instruct-FP8│      698 │     17.2 │   461.6 │   6.40 │        232.0 │           8 │ 11.75 GB │ vLLM 0.15.1│ NVIDIA RTX PRO 6000 Blackwell Workstation Edition   │
├───────────────────────────────┼──────────┼──────────┼─────────┼────────┼──────────────┼─────────────┼──────────┼────────────┼──────────────────────────────────────────────────────┤
│ Qwen/Qwen3-VL-8B-Instruct    │      638 │     17.9 │   448.0 │   6.40 │        233.6 │           8 │ 17.41 GB │ vLLM 0.15.1│ NVIDIA RTX PRO 6000 Blackwell Workstation Edition   │
╰───────────────────────────────┴──────────┴──────────┴─────────┴────────┴──────────────┴─────────────┴──────────┴────────────┴──────────────────────────────────────────────────────╯

╭─ Summary ────────────────────────────────────────────────────────────────────╮
│  Runs       4 across 4 model(s)  total duration 818.8s                       │
│  Tok/s      1664.8 best   448.0 worst   763.0 avg                            │
│  Errors     0                                                                │
╰──────────────────────────────────────────────────────────── vlmbench v0.1.0 ─╯
```

## Usage

### Mac + Ollama

```bash
# Auto-detects Ollama at localhost:11434 (lowercase model names)
uvx vlmbench run -m qwen3-vl:2b -i ./images/
uvx vlmbench run -m glm-ocr:latest -i ./images/
```

### Linux + vLLM (Docker)

```bash
# Auto-starts vLLM via Docker with --gpus all (HuggingFace model IDs)
uvx vlmbench run -m Qwen/Qwen3-VL-2B-Instruct -i ./images/

# Nightly Docker image
uvx vlmbench run -m PaddlePaddle/PaddleOCR-VL-1.5 -i ./images/ \
  --backend vllm-openai:nightly

# Concurrency for throughput testing
uvx vlmbench run -m Qwen/Qwen3-VL-8B-Instruct -i ./images/ \
  --max-concurrency 8 --runs 3
```

### Linux + vLLM (native)

```bash
# Requires vllm installed (pip install vllm)
uvx vlmbench run -m Qwen/Qwen3-VL-2B-Instruct -i ./images/ --backend vllm
```

### Cloud API

```bash
uvx vlmbench run -m Qwen/Qwen3-VL-2B-Instruct -i ./images/ \
  --base-url https://api.together.xyz/v1 --api-key $TOGETHER_API_KEY
```

### Compare

```bash
uvx vlmbench compare results/*.json
```

## CLI Flags

| Flag | Default | Description |
|---|---|---|
| `--model` / `-m` | required | Model ID (vLLM: `Qwen/Qwen3-VL-2B-Instruct`, Ollama: `qwen3-vl:2b`) |
| `--input` / `-i` | required | File or directory (images, PDFs, videos) |
| `--base-url` | auto-detect | OpenAI-compatible base URL |
| `--api-key` | `no-key` | API key (also reads `OPENAI_API_KEY` env) |
| `--prompt` | `"Extract all text..."` | Prompt sent with each input |
| `--max-tokens` | `2048` | Max completion tokens |
| `--runs` | `3` | Timed runs per input |
| `--warmup` | `1` | Warmup runs (not recorded, fail-fast on errors) |
| `--max-concurrency` | `1` | Max parallel requests |
| `--save` | `./results/` | Output directory |
| `--backend` | `auto` | `auto`, `ollama`, `vllm` (native), `vllm-openai:<tag>` (Docker), `sglang:<tag>` |
| `--serve/--no-serve` | `--serve` | Auto-start server if none detected |
| `--serve-args` | none | Extra args passed to server (Docker or native) |
| `--tag` | none | Custom grouping label |
| `--quant` | `auto` | Quantization metadata: `fp16`, `bf16`, `q4_K_M`, etc. |
| `--revision` | `main` | Model revision metadata |

## Backends

| `--backend` | Resolves to | Serving |
|---|---|---|
| `auto` | `ollama` on macOS, `vllm-openai:latest` on Linux | Native / Docker |
| `ollama` | Ollama native | `ollama serve` in tmux |
| `vllm` | Native vLLM | `vllm serve` in tmux |
| `vllm-openai:latest` | `vllm/vllm-openai:latest` | `docker run --gpus all` |
| `vllm-openai:nightly` | `vllm/vllm-openai:nightly` | `docker run --gpus all` |
| `sglang:latest` | `lmsysorg/sglang:latest` | `docker run --gpus all` (coming soon) |

All Docker backends run with `--gpus all --ipc=host` and a deterministic container name (`vlmbench-vllm-openai`, `vlmbench-sglang`) for easy log access.

## Monitoring

Every run starts a tmux session with two panes:

- **Top**: server logs (`tail -f ~/.ollama/logs/server.log` or `docker logs -f`)
- **Bottom**: GPU monitor (`macmon` on macOS, `nvitop` on Linux)

Attach with `tmux attach -t vlmbench-vllm`.

## Supported Models

See [MODELS.md](.claude/skills/vlmbench/MODELS.md) for tested models and their required `--serve-args`.

## Input Types

| Type | Extensions | Processing |
|---|---|---|
| Image | `.png`, `.jpg`, `.jpeg`, `.webp`, `.tiff`, `.bmp` | Base64 encode |
| PDF | `.pdf` | `pdf2image` per-page -> base64 |
| Video | `.mp4`, `.mov`, `.avi`, `.mkv`, `.webm` | `ffmpeg` 1fps -> frames -> base64 |

Directories processed recursively, sorted alphabetically.

## Output

Results saved as JSON to `./results/{model-slug}-{timestamp}.json` with model metadata, environment info, benchmark stats (TTFT, TPOT, throughput, latency percentiles), and raw per-run data.

## Requirements

- Python >= 3.11
- [uv](https://docs.astral.sh/uv/) (recommended)
- Docker + NVIDIA GPU support (for `vllm-openai`/`sglang` Docker backends)
- vLLM (`uv pip install vllm`) for native `--backend vllm`
- tmux (for server management and monitoring)
- macmon (`brew install macmon`) or nvitop (GPU monitoring)
- ffmpeg (video input), poppler (PDF input) — optional
