<div align="center">
<p align="center" style="width: 100%;">
    <img src="https://raw.githubusercontent.com/vlm-run/.github/refs/heads/main/profile/assets/vlm-black.svg" alt="VLM Run Logo" width="80" style="margin-bottom: -5px; color: #2e3138; vertical-align: middle; padding-right: 5px;"><br>
</p>
<h2>vlmbench</h2>
<p><b>Single-file, drop-in VLM benchmark CLI for your agents.</b></p>
<p align="center">
<a href="https://pypi.org/project/vlmbench/"><img alt="PyPI Version" src="https://badge.fury.io/py/vlmbench.svg"></a>
<a href="https://pypi.org/project/vlmbench/"><img alt="Python Versions" src="https://img.shields.io/pypi/pyversions/vlmbench"></a>
<a href="https://www.pepy.tech/projects/vlmbench"><img alt="PyPI Downloads" src="https://img.shields.io/pypi/dm/vlmbench"></a><br>
<a href="https://github.com/vlm-run/vlmbench/blob/main/LICENSE"><img alt="License" src="https://img.shields.io/github/license/vlm-run/vlmbench.svg"></a>
<a href="https://discord.gg/AMApC2UzVY"><img alt="Discord" src="https://img.shields.io/badge/discord-chat-purple?color=%235765F2&label=discord&logo=discord"></a>
<a href="https://twitter.com/vlmrun"><img alt="Twitter Follow" src="https://img.shields.io/twitter/follow/vlmrun.svg?style=social&logo=twitter"></a>
</p>
</div>

Benchmark any vision-language model on your own hardware with a single command. vlmbench auto-detects your platform, starts the right backend, and gives you reproducible results as JSON.

- [**Ollama**](https://ollama.com) on **macOS**: auto-starts, zero config
- [**vLLM**](https://docs.vllm.ai) on **Linux**: via Docker (`--gpus all`, auto-pulls) or native vLLM
- [**SGLang**](https://github.com/lmsysorg/sglang) on **Linux**: coming soon

<img alt="image" src="./assets/screenshot.png" style="width: 100%; height: auto;"/>

## Quick Start

No install needed — just run with [`uvx`](https://docs.astral.sh/uv/):

```bash
# Local images/PDFs (macOS Ollama)
uvx vlmbench run -m qwen3-vl:2b -i ./images/

# Linux + vLLM Docker (auto-starts with --gpus all)
uvx vlmbench run -m Qwen/Qwen3-VL-2B-Instruct -i ./images/

# HuggingFace dataset
uvx vlmbench run -m Qwen/Qwen3-VL-2B-Instruct \
  -d hf://vlm-run/FineVision-vlmbench-mini --max-samples 64

# Concurrency sweep
uvx vlmbench run -m Qwen/Qwen3-VL-8B-Instruct -i ./images/ \
  --concurrency 4,8,16,32,64

# Use a model profile (custom serve args + setup)
uvx vlmbench run --profile deepseek-ocr -i ./images/

# Cloud API
uvx vlmbench run -m Qwen/Qwen3-VL-2B-Instruct -i ./images/ \
  --base-url https://api.openai.com/v1 --api-key $OPENAI_API_KEY
```

Or install it: `pip install vlmbench`

## Example Run

```bash
uvx vlmbench run -m Qwen/Qwen3-VL-2B-Instruct \
  -d hf://vlm-run/FineVision-vlmbench-mini --max-samples 64 \
  --prompt "Describe this image in 80 words or less" \
  --concurrency 4,8,16 --backend vllm
```

```
╭─ Configuration ──────────────────────────────────────────────────────────────╮
│                                                                              │
│  model        Qwen/Qwen3-VL-2B-Instruct                                      │
│  revision     main                                                           │
│  backend      vLLM 0.11.2                                                    │
│  endpoint     http://localhost:8000/v1                                       │
│                                                                              │
│  gpu          NVIDIA RTX PRO 6000 Blackwell Workstation Edition              │
│  vram         97,887 MiB                                                     │
│  driver       580.126.09                                                     │
│                                                                              │
│  dataset      hf://vlm-run/FineVision-vlmbench-mini                          │
│  images       64 (mixed)                                                     │
│                                                                              │
│  max_tokens   2048                                                           │
│  runs         3                                                              │
│  concurrency  8                                                              │
│                                                                              │
│  monitor      tmux attach -t vlmbench-vllm                                   │
│                                                                              │
╰──────────────────────────────────────────────────────────────────────────────╯

╭─ Results ────────────────────────────────────────────────────────────────────╮
│                                                                              │
│  Metric                Value              p50        p95        p99          │
│  Throughput            13.33 img/s         —          —          —           │
│  Tokens/sec            1168 tok/s          —          —          —           │
│  Workers               8                   —          —          —           │
│  TTFT                  58 ms           51 ms     114 ms     140 ms           │
│  TPOT                  5.3 ms         5.0 ms     7.3 ms     7.4 ms           │
│  Latency (per worker)  0.54 s/img     0.46 s     0.92 s     1.36 s           │
│                                                                              │
│  Tokens (avg)          prompt 2,077  •  completion 88                        │
│  Token ranges          prompt 180–8,545  •  completion 55–190                │
│  Images                144  •  avg 964×867 (0.93 MP)                         │
│  Resolution            min 338×266  •  median 1024×768  •  max 2048×1755     │
│  VRAM peak             69.7 GB                                               │
│  Reliability           192/192 ok  •  14.4s total                            │
│                                                                              │
╰──────────────────────────────────────────────────────────────────────────────╯
```

## Leaderboard

Best peak throughput per model on NVIDIA RTX PRO 6000 Blackwell (vLLM v0.15.1, 39 runs across concurrency sweeps):

| # | Model | Best Tok/s | Workers | TTFT | TPOT |
|---|-------|-----------|---------|------|------|
| 1 | `lightonai/LightOnOCR-2-1B` | **2,439.8** | 32 | 1,439 ms | 22.1 ms |
| 2 | `Qwen/Qwen3-VL-2B-Instruct` | 2,409.3 | 64 | 440 ms | 14.3 ms |
| 3 | `PaddlePaddle/PaddleOCR-VL` | 2,341.9 | 64 | 6,385 ms | 49.0 ms |
| 4 | `deepseek-ai/DeepSeek-OCR` | 1,195.8 | 32 | 3,571 ms | 15.9 ms |
| 5 | `Qwen/Qwen3-VL-8B-Instruct` | 953.8 | 64 | 448 ms | 25.7 ms |

Compare your own results:

```bash
uvx vlmbench compare                       # auto-discovers ~/.vlmbench/benchmarks/
uvx vlmbench compare results/*.json        # or pass files explicitly
```

See [MODELS.md](.claude/skills/vlmbench/MODELS.md) for all tested models and their required `--serve-args`.

## Profiles

Some models need custom Docker images, extra pip installs, or special serve args. Profiles bundle all of this into a single YAML file — just pass `--profile` and vlmbench handles the rest.

```bash
uvx vlmbench profiles                                  # list available profiles
uvx vlmbench run --profile deepseek-ocr -i ./images/   # run with a profile
```

When you use `--profile`, it sets `--model`, `--prompt`, `--serve-args`, and (for Docker builds) the base image and setup commands. You can still override any flag explicitly.

| Profile | Model | Base Image | Custom Setup |
|---------|-------|------------|--------------|
| `glm-ocr` | `zai-org/GLM-OCR` | `vllm/vllm-openai:nightly` | vLLM nightly + transformers >= 5.1.0, MTP speculative decoding |
| `deepseek-ocr` | `deepseek-ai/DeepSeek-OCR` | `vllm/vllm-openai:v0.15.1` | Custom logits processor, no prefix caching |
| `paddleocr-vl` | `PaddlePaddle/PaddleOCR-VL` | `vllm/vllm-openai:v0.15.1` | Trust remote code, no prefix caching |
| `qwen3-vl-2b` | `Qwen/Qwen3-VL-2B-Instruct` | `vllm/vllm-openai:v0.15.1` | — |
| `qwen3-vl-8b` | `Qwen/Qwen3-VL-8B-Instruct` | `vllm/vllm-openai:v0.15.1` | — |

Profiles live in `vlmbench/profiles/*.yaml` and ship with the package. For local Docker workflows:

```bash
make build PROFILE=glm-ocr        # generates Dockerfile + docker build
make serve PROFILE=glm-ocr        # start server in tmux
make benchmark PROFILE=glm-ocr    # run benchmark against the server
```

## CLI Reference

| Flag | Default | Description |
|---|---|---|
| `--model` / `-m` | required | Model ID. Optional if `--profile` is set. |
| `--profile` | none | Model profile (e.g. `glm-ocr`). Sets model, prompt, serve-args. See `vlmbench profiles`. |
| `--input` / `-i` | sample URL | File, directory, or URL (images, PDFs, videos) |
| `--dataset` / `-d` | none | HuggingFace dataset (e.g. `vlm-run/FineVision-vlmbench-mini`) |
| `--base-url` | auto-detect | OpenAI-compatible base URL |
| `--api-key` | `no-key` | API key (also reads `OPENAI_API_KEY` env) |
| `--prompt` | `"Extract all text..."` | Prompt sent with each input |
| `--max-tokens` | `2048` | Max completion tokens |
| `--runs` | `3` | Timed runs per input |
| `--warmup` | `1` | Warmup runs (not recorded, fail-fast on errors) |
| `--concurrency` | `8` | Single value or comma-separated sweep (e.g. `4,8,16,32,64`) |
| `--max-samples` | all | Limit number of input samples (useful for dry-runs) |
| `--output-directory` | `~/.vlmbench/benchmarks/` | Output directory |
| `--tag` | none | Custom label (used in result filename and metadata) |
| `--upload` | off | Upload results to HuggingFace (requires `HF_TOKEN`) |
| `--upload-repo` | `vlm-run/vlmbench-results` | HuggingFace dataset repo for uploads |
| `--backend` | `auto` | `auto`, `ollama`, `vllm`, `vllm-openai:<tag>`, `sglang:<tag>` |
| `--serve/--no-serve` | `--serve` | Auto-start server if none detected |
| `--serve-args` | none | Extra args passed to server |
| `--quant` | `auto` | Quantization metadata: `fp16`, `bf16`, `q4_K_M`, etc. |
| `--revision` | `main` | Model revision metadata |

### Backends

| `--backend` | Resolves to | Serving |
|---|---|---|
| `auto` | `ollama` on macOS, `vllm-openai:latest` on Linux | Native / Docker |
| `ollama` | Ollama native | `ollama serve` in tmux |
| `vllm` | Native vLLM | `vllm serve` in tmux |
| `vllm-openai:latest` | `vllm/vllm-openai:latest` | `docker run --gpus all` |
| `vllm-openai:nightly` | `vllm/vllm-openai:nightly` | `docker run --gpus all` |
| `sglang:latest` | `lmsysorg/sglang:latest` | `docker run --gpus all` (coming soon) |

All Docker backends run with `--gpus all --ipc=host` and a deterministic container name for easy log access.

### Input Types

| Type | Extensions | Processing |
|---|---|---|
| Image | `.png`, `.jpg`, `.jpeg`, `.webp`, `.tiff`, `.bmp` | Base64 encode |
| PDF | `.pdf` | `pypdfium2` per-page -> base64 |
| Video | `.mp4`, `.mov`, `.avi`, `.mkv`, `.webm` | `ffmpeg` 1fps -> frames -> base64 |

Directories are processed recursively, sorted alphabetically.

### Output

Results are saved as JSON to `~/.vlmbench/benchmarks/` with model metadata, environment info, benchmark stats (TTFT, TPOT, throughput, latency percentiles), and raw per-run data. Each concurrency level produces a separate file.

Upload results to HuggingFace with `--upload`:

```bash
uvx vlmbench run -m Qwen/Qwen3-VL-2B-Instruct -d hf://vlm-run/FineVision-vlmbench-mini \
  --concurrency 4,8,16,32,64 --upload
```

Browse uploaded results at [`vlm-run/vlmbench-results`](https://huggingface.co/datasets/vlm-run/vlmbench-results).

## How It Works

When you run `vlmbench run`, here's what happens:

1. **Detects your platform** — macOS routes to Ollama, Linux to vLLM Docker
2. **Pulls the Docker image** — `docker pull vllm/vllm-openai:latest` (cached after first run)
3. **Starts the server in tmux** — `docker run --gpus all` in a named session (`vlmbench-vllm`)
4. **Launches a GPU monitor** — `nvitop` (Linux) or `macmon` (macOS) in a split pane
5. **Waits for the server** — polls `/v1/models` until ready (up to 600s)
6. **Runs warmup requests** — fail-fast validation before timed runs
7. **Benchmarks with concurrency** — streams completions via the OpenAI API, measures TTFT/TPOT/throughput
8. **Saves results as JSON** — one file per concurrency level in `~/.vlmbench/benchmarks/`

Attach to the live session anytime: `tmux attach -t vlmbench-vllm`

<details>
<summary><b>tmux session capture</b> — server logs + GPU monitor side by side</summary>

**Top pane — vLLM server logs:**

```
(APIServer pid=1) INFO 02-07 15:44:24 non-default args: {
  'model': 'lightonai/LightOnOCR-2-1B',
  'enable_prefix_caching': False,
  'limit_mm_per_prompt': {'image': 1},
  'mm_processor_cache_gb': 0.0
}
(APIServer pid=1) INFO 02-07 15:44:34 Resolved architecture: LightOnOCRForConditionalGeneration
(APIServer pid=1) INFO 02-07 15:44:34 Using max model len 16384
(EngineCore pid=272) INFO 02-07 15:44:44 Initializing a V1 LLM engine (v0.15.1) with config:
  model='lightonai/LightOnOCR-2-1B', dtype=torch.bfloat16, max_seq_len=16384,
  tensor_parallel_size=1, quantization=None
(EngineCore pid=272) INFO 02-07 15:45:41 Loading weights took 0.49 seconds
(EngineCore pid=272) INFO 02-07 15:45:42 Model loading took 1.88 GiB memory and 22.15 seconds
(EngineCore pid=272) INFO 02-07 15:46:11 Available KV cache memory: 77.94 GiB
(EngineCore pid=272) INFO 02-07 15:46:11 Maximum concurrency for 16,384 tokens per request: 44.53x
Capturing CUDA graphs (decode, FULL): 100% |██████████| 51/51
(APIServer pid=1) INFO Started server process [1]
(APIServer pid=1) INFO Application startup complete.
(APIServer pid=1) INFO 172.17.0.1 - "POST /v1/chat/completions HTTP/1.1" 200 OK
```

**Bottom pane — nvitop GPU monitor:**

```
NVITOP 1.6.2      Driver Version: 580.126.09      CUDA Driver Version: 13.0
╒═══════════════════════════════╤══════════════════════╤══════════════════════╕
│ GPU  Name        Persistence-M│ Bus-Id        Disp.A │ Volatile Uncorr. ECC │
│ Fan  Temp  Perf  Pwr:Usage/Cap│         Memory-Usage │ GPU-Util  Compute M. │
╞═══════════════════════════════╪══════════════════════╪══════════════════════╡
│   0  GeForce RTX 2080 Ti  Off │ 00000000:21:00.0 Off │                  N/A │
│ 27%   42C   P8     17W / 250W │  107.2MiB / 11264MiB │      0%      Default │
├───────────────────────────────┼──────────────────────┼──────────────────────┤
│   1  RTX PRO 6000         Off │ 00000000:4B:00.0 Off │                  N/A │
│ 30%   33C   P1     66W / 600W │  86.54GiB / 95.59GiB │      0%      Default │
╘═══════════════════════════════╧══════════════════════╧══════════════════════╛
  MEM: ███████████████████████████████████████████████████████████▏ 90.5%
  Load Average: 4.14  2.73  1.65
```

</details>

## Requirements

- **Python >= 3.11**, [uv](https://docs.astral.sh/uv/) recommended
- **Linux**: Docker + NVIDIA GPU support (or native vLLM via `uv pip install vllm`)
- **Monitoring**: `tmux`, `nvitop` (Linux) or `macmon` (macOS)
- **Optional**: `ffmpeg` (video frame extraction)
