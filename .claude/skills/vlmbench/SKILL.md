---
name: vlmbench
description: >
  VLM benchmark CLI — run, compare, and reproduce VLM inference benchmarks.
  Use when the user wants to: (1) benchmark a VLM model's throughput, latency,
  or VRAM usage, (2) compare results across models or configs, (3) choose or
  configure a backend (vLLM Docker, vLLM native, Ollama, SGLang, cloud API),
  (4) run text-only LLM benchmarks via HF dataset text columns, (5) look up
  model-specific --serve-args, or (6) debug server startup or GPU detection
  issues. Triggers on: vlmbench, VLM benchmarking, OCR model evaluation,
  inference throughput, tokens/sec, TTFT, TPOT, VRAM, text benchmark.
---

# vlmbench

All CLI logic is in `vlmbench/cli.py` (single file). Entry point: `main()`.

## Running benchmarks

Prefer `uvx` — no install needed:

```bash
# macOS (Ollama auto-starts)
uvx vlmbench run -m qwen3-vl:2b -i ./images/

# Linux (vLLM Docker auto-starts with --gpus all)
uvx vlmbench run -m Qwen/Qwen3-VL-8B-Instruct -i ./images/

# Linux (native vLLM)
uvx vlmbench run -m Qwen/Qwen3-VL-8B-Instruct -i ./images/ --backend vllm

# Remote/cloud server — model auto-detected from GET /v1/models
uvx vlmbench run -i ./images/ --base-url https://my-server.example.com/v1 --api-key $KEY

# HF image dataset
uvx vlmbench run -m Qwen/Qwen3-VL-2B-Instruct \
  -d hf://vlm-run/FineVision-vlmbench-mini --max-samples 64

# HF text dataset (text-only / LLM benchmark)
uvx vlmbench run -m meta-llama/Llama-3.1-8B-Instruct \
  -d hf://my-org/my-prompts --dataset-text-col prompt --prompt ""

# Compare results
uvx vlmbench compare results/*.json
```

`run` is the default subcommand — flags like `--model` implicitly invoke it.

## --model is optional

When `--base-url` points to a running server, `-m` / `--model` can be omitted.
vlmbench calls `GET /v1/models` on the server and uses the first returned model ID.
The `Authorization: Bearer <api-key>` header is included automatically when `--api-key` is set.
`--model` (or `--profile`) is still required when using `--serve` to start a new server.

## Text-only / LLM benchmarks

Use `--dataset-text-col <col>` to load a string column from an HF dataset instead of images.
Each row's value is sent as a `{"type": "text"}` content block before the `--prompt` instruction.

```bash
# Column value IS the full message (no instruction)
--dataset-text-col prompt --prompt ""

# Column value is the document; --prompt is the instruction
--dataset-text-col text --prompt "Summarize in one sentence."
```

Auto-detection: when no image column is found, vlmbench falls back to any column named
`text`, `prompt`, `input`, `content`, `query`, `question`, or `instruction`.

## Backend resolution

| Platform | `auto` resolves to | Model ID style |
|---|---|---|
| macOS | `ollama` | `qwen3-vl:2b` |
| Linux | `vllm-openai:latest` (Docker) | `Qwen/Qwen3-VL-2B-Instruct` |

Docker backends use `--gpus all --ipc=host`. Servers run in tmux sessions (`vlmbench-vllm`, `vlmbench-ollama`, `vlmbench-sglang`).

## Model-specific serve-args

See [MODELS.md](MODELS.md) for the full table of tested models and their required `--serve-args`. Reference this file when the user asks which models are supported or how to serve a specific model.

## Concurrency sweep

Use `--concurrency 4,8,16,32,64` to run at multiple concurrency levels in a single invocation. Produces one JSON per level (tagged `c4`, `c8`, etc.) and a consolidated comparison table. A single value (e.g. `--concurrency 8`) runs at that level only.

## Key metrics

TTFT (ms), TPOT (ms), tok/s, img/s, latency s/img, VRAM peak (MiB), prompt/completion token counts. Results saved as JSON to `./results/{backend}-v{version}-{model}-{gpu}-{tag}.json`.

## GPU detection

`CUDA_VISIBLE_DEVICES` is respected via `_nvidia_gpu_index()` -> `nvidia-smi --id={idx}`. When VRAM or GPU name looks wrong, check this env var first.
