# vlmbench on HuggingFace Jobs

Run VLM benchmarks across GPU types on HuggingFace infrastructure.
Results upload to a HF dataset repo for leaderboard generation.

## Prerequisites

- [HuggingFace Pro](https://huggingface.co/pro) account or org with Jobs access
- `HF_TOKEN` in `.env` file (or exported)

## Quick Start: Sweep a Model Across GPUs

```bash
# 1. Set your HF token
echo "HF_TOKEN=hf_xxx" > .env

# 2. Sweep Qwen3-VL-2B across 3 GPU types
uv run scripts/hf_jobs/sweep_gpus.py \
  --model Qwen/Qwen3-VL-2B-Instruct \
  --flavors t4-small,l4x1,a10g-small

# 3. Results land in: huggingface.co/datasets/vlm-run/vlmbench-results
```

## Run a Single GPU Job Directly

```bash
hf jobs uv run \
  --flavor a10g-small \
  --image vllm/vllm-openai:latest \
  scripts/hf_jobs/run_benchmark.py -- \
    --model Qwen/Qwen3-VL-2B-Instruct \
    --gpu-flavor a10g-small
```

## Adding a New Model to the Leaderboard

1. Pick the model ID (e.g. `Qwen/Qwen3-VL-8B-Instruct`)
2. Check if it needs special vLLM args (see `.claude/skills/vlmbench/MODELS.md`)
3. Run the sweep:
   ```bash
   uv run scripts/hf_jobs/sweep_gpus.py \
     --model Qwen/Qwen3-VL-8B-Instruct \
     --flavors t4-small,l4x1,a10g-small,a100-large \
     --serve-args '--mm-encoder-tp-mode "data"'
   ```
4. Results upload to `vlm-run/vlmbench-results` under `results/<gpu-flavor>/`

## Available GPU Flavors

| Flavor | GPU | VRAM |
|--------|-----|------|
| `t4-small` | T4 | 16 GB |
| `t4-medium` | T4 | 16 GB |
| `l4x1` | L4 | 24 GB |
| `l4x4` | 4x L4 | 96 GB |
| `a10g-small` | A10G | 24 GB |
| `a10g-large` | A10G | 24 GB |
| `a10g-largex2` | 2x A10G | 48 GB |
| `a10g-largex4` | 4x A10G | 96 GB |
| `a100-large` | A100 | 80 GB |

## Results Structure

```
vlm-run/vlmbench-results/
  results/
    t4-small/
      qwen3-vl-2b-instruct-20260208T143022.json
    l4x1/
      qwen3-vl-2b-instruct-20260208T143105.json
    a10g-small/
      qwen3-vl-2b-instruct-20260208T143048.json
```
