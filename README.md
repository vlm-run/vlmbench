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

- **macOS** — [Ollama](https://ollama.com) (auto-starts, zero config)
- **Linux** — [vLLM](https://docs.vllm.ai) via Docker (`--gpus all`, auto-pulls) or native vLLM
- **SGLang** — coming soon

<img width="2468" height="1920" alt="image" src="https://raw.githubusercontent.com/vlm-run/vlmbench/main/assets/screenshot.png" />

## Quick Start

No install needed — just run with [`uvx`](https://docs.astral.sh/uv/):

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
│   Throughput   1664.8 tok/s   9.20 img/s                                     │
│   Latency        0.87 s/img  (p95: 3.55 s)                                   │
│   Tokens          270 prompt    181 completion (avg)                          │
│   Output       11222 total   181 tok/img   1664.8 out tok/s                  │
│   VRAM          5920 MiB peak                                                │
│   Reliability  186/186 ok, 0 retries                                         │
│                                                                              │
╰──────────────────────────────────────────────────────────────────────────────╯
  > Saved → results/lightonocr-2-1b-20260207T104621.json
```

## Behind the Scenes

When you run `uvx vlmbench run`, here's what happens automatically:

1. **Detects your platform** — macOS routes to Ollama, Linux to vLLM Docker
2. **Pulls the Docker image** — `docker pull vllm/vllm-openai:latest` (cached after first run)
3. **Starts the server in tmux** — `docker run --gpus all` in a named tmux session (`vlmbench-vllm`)
4. **Launches a GPU monitor** — `nvitop` (Linux) or `macmon` (macOS) in a split pane
5. **Waits for the server** — polls `/v1/models` until ready (up to 600s for large models)
6. **Runs warmup requests** — fail-fast validation before timed runs
7. **Benchmarks with concurrency** — streams completions via the OpenAI API, measures TTFT/TPOT/throughput
8. **Saves results as JSON** — one file per run in `./results/`, ready for `vlmbench compare`

Attach to the live session anytime with `tmux attach -t vlmbench-vllm`.

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

## Compare

```bash
uvx vlmbench compare results/*.json
```

```
╭───────────────────────────────┬──────────┬──────────┬─────────┬────────┬──────────────┬─────────────┬──────────┬────────────┬──────────────────────────────────────────────────────╮
│                               │     TTFT │     TPOT │         │        │ Duration (s) │ num_workers │     VRAM │            │                                                      │
│ Model                         │     (ms) │     (ms) │ Tok/s ↓ │  Img/s │              │             │          │ Backend    │ Hardware                                             │
├───────────────────────────────┼──────────┼──────────┼─────────┼────────┼──────────────┼─────────────┼──────────┼────────────┼──────────────────────────────────────────────────────┤
│ lightonai/LightOnOCR-2-1B     │      467 │      6.0 │  1664.8 │   9.20 │        162.4 │           8 │  5.78 GB │ vLLM 0.15.1│ NVIDIA RTX PRO 6000 Blackwell Workstation Edition    │
├───────────────────────────────┼──────────┼──────────┼─────────┼────────┼──────────────┼─────────────┼──────────┼────────────┼──────────────────────────────────────────────────────┤
│ rednote-hilab/dots.ocr        │     1424 │     10.2 │   477.6 │   7.76 │        190.8 │           8 │  9.42 GB │ vLLM 0.15.1│ NVIDIA RTX PRO 6000 Blackwell Workstation Edition    │
├───────────────────────────────┼──────────┼──────────┼─────────┼────────┼──────────────┼─────────────┼──────────┼────────────┼──────────────────────────────────────────────────────┤
│ Qwen/Qwen3-VL-8B-Instruct-FP8 │      698 │     17.2 │   461.6 │   6.40 │        232.0 │           8 │ 11.75 GB │ vLLM 0.15.1│ NVIDIA RTX PRO 6000 Blackwell Workstation Edition    │
├───────────────────────────────┼──────────┼──────────┼─────────┼────────┼──────────────┼─────────────┼──────────┼────────────┼──────────────────────────────────────────────────────┤
│ Qwen/Qwen3-VL-8B-Instruct     │      638 │     17.9 │   448.0 │   6.40 │        233.6 │           8 │ 17.41 GB │ vLLM 0.15.1│ NVIDIA RTX PRO 6000 Blackwell Workstation Edition    │
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

## Experiments

Benchmarking recipes that answer real questions about your VLM setup. Mix and match these patterns to build the experiment you need.

### Model shootout — which model is fastest?

```bash
# Same inputs, same concurrency, different models
vlmbench run -m lightonai/LightOnOCR-2-1B    -i ./docs/ --max-concurrency 8 --runs 3
vlmbench run -m rednote-hilab/dots.ocr        -i ./docs/ --max-concurrency 8 --runs 3
vlmbench run -m Qwen/Qwen3-VL-8B-Instruct    -i ./docs/ --max-concurrency 8 --runs 3
vlmbench run -m allenai/olmOCR-2-7B-1025-FP8  -i ./docs/ --max-concurrency 8 --runs 3

# Compare — sorted by tok/s
vlmbench compare results/*.json
```

When all runs used `--profile ocr --reference`, compare includes quality columns:

```
  compare (4 runs)

╭────────────────────────────────┬────────┬────────┬─────────┬────────┬─────────┬───────┬───────┬──────────┬────────────╮
│                                │   TTFT │   TPOT │         │        │   Tok/  │       │       │          │            │
│ Model                          │   (ms) │   (ms) │ Tok/s ↓ │  Img/s │   img   │  CER  │  WER  │     VRAM │ Backend    │
├────────────────────────────────┼────────┼────────┼─────────┼────────┼─────────┼───────┼───────┼──────────┼────────────┤
│ lightonai/LightOnOCR-2-1B      │    467 │    6.0 │  1664.8 │   9.20 │     181 │  2.1% │  4.8% │  5.78 GB │ vLLM 0.15  │
│ rednote-hilab/dots.ocr         │   1424 │   10.2 │   477.6 │   7.76 │     203 │  2.4% │  5.1% │  9.42 GB │ vLLM 0.15  │
│ Qwen/Qwen3-VL-8B-Instruct     │    698 │   17.2 │   461.6 │   6.40 │     289 │  1.4% │  3.2% │ 17.41 GB │ vLLM 0.15  │
│ allenai/olmOCR-2-7B-1025-FP8   │    812 │   14.8 │   412.3 │   5.74 │     312 │  1.6% │  3.5% │ 10.22 GB │ vLLM 0.15  │
╰────────────────────────────────┴────────┴────────┴─────────┴────────┴─────────┴───────┴───────┴──────────┴────────────╯

╭─ Summary ────────────────────────────────────────────────────────────────────╮
│  Runs       4 across 4 model(s)  total duration 818.8s                       │
│  Tok/s      1664.8 best   412.3 worst   754.1 avg                            │
│  Quality    1.4% best CER (Qwen3-VL-8B)   2.4% worst CER (dots.ocr)         │
│  Errors     0                                                                │
╰──────────────────────────────────────────────────────────── vlmbench v0.2.5 ─╯
```

### OCR quality profiling — fast, but is the output correct?

Use `--profile ocr` to measure **output token volume** alongside speed. Add `--reference` with ground-truth text files to score accuracy (CER / WER).

```bash
# Token-level output profiling (output tok/s, tokens per image, total output tokens)
vlmbench run -m lightonai/LightOnOCR-2-1B -i ./docs/ \
  --profile ocr --max-concurrency 8 --save-outputs

# Score against ground truth — filenames must match (invoice_001.png → invoice_001.txt)
vlmbench run -m lightonai/LightOnOCR-2-1B -i ./docs/ \
  --profile ocr --reference ./ground-truth/ --save-outputs

# Quality shootout — which model produces the best OCR?
vlmbench run -m Qwen/Qwen3-VL-8B-Instruct -i ./docs/ \
  --profile ocr --reference ./ground-truth/ --save-outputs
vlmbench run -m rednote-hilab/dots.ocr -i ./docs/ \
  --profile ocr --reference ./ground-truth/ --save-outputs
vlmbench compare results/*ocr*.json
```

The Results panel is followed by a quality panel when `--profile ocr --reference` is used:

```
╭─ Results ────────────────────────────────────────────────────────────────────╮
│                                                                              │
│   TTFT           467 ms    (p95: 1975 ms)                                    │
│   TPOT           6.0 ms    (p95: 6.2 ms)                                     │
│   Throughput   1664.8 tok/s   9.20 img/s                                     │
│   Latency        0.87 s/img  (p95: 3.55 s)                                   │
│   Tokens          270 prompt    181 completion (avg)                          │
│   Output       11222 total   181 tok/img   1664.8 out tok/s                  │
│   VRAM          5920 MiB peak                                                │
│   Reliability  186/186 ok, 0 retries                                         │
│                                                                              │
╰──────────────────────────────────────────────────────────────────────────────╯

╭─ OCR Quality (vs ground truth) ─────────────────────────────────────────────╮
│                                                                              │
│   CER            2.1%      (best: 0.4%, worst: 8.7%)                         │
│   WER            4.8%      (best: 1.2%, worst: 16.3%)                        │
│   Exact match   12/62      19.4%                                             │
│                                                                              │
│   Worst inputs                                                               │
│     scan_047.jpg     CER: 8.7%   WER: 16.3%   (handwritten, low contrast)   │
│     receipt_019.png  CER: 7.2%   WER: 14.8%   (crumpled, skewed)            │
│     form_003.tiff    CER: 6.1%   WER: 11.9%   (multi-column layout)         │
│                                                                              │
╰──────────────────────────────────────────────────────────────────────────────╯
  > Saved → results/lightonocr-2-1b-ocr-20260222T143012.json
  > Outputs → results/lightonocr-2-1b-ocr-20260222T143012-outputs/
```

### Markdown extraction — does it preserve table structure?

```bash
vlmbench run -m Qwen/Qwen3-VL-8B-Instruct -i ./financial-reports/ \
  --profile markdown --reference ./ground-truth/ --save-outputs \
  --prompt "Convert this document to markdown. Preserve all tables, headings, and formatting."

vlmbench run -m lightonai/LightOnOCR-2-1B -i ./financial-reports/ \
  --profile markdown --reference ./ground-truth/ --save-outputs

# Compare — output quality metrics side-by-side (CER, WER, structure fidelity)
vlmbench compare results/*markdown*.json
```

The `--profile markdown` panel scores structural element preservation:

```
╭─ Markdown Quality (vs ground truth) ────────────────────────────────────────╮
│                                                                              │
│   CER            1.8%      (best: 0.2%, worst: 6.1%)                         │
│   WER            3.9%      (best: 0.8%, worst: 12.4%)                        │
│                                                                              │
│   Structure                                                                  │
│     Headings     94.2%     preserved   (49/52)                               │
│     Tables       87.1%     preserved   (27/31)                               │
│     Lists        91.8%     preserved   (56/61)                               │
│     LaTeX        78.4%     preserved   (29/37)                               │
│                                                                              │
│   Worst inputs                                                               │
│     10k_page3.pdf    Tables: 2/5 preserved   (nested multi-level headers)    │
│     balance.pdf      LaTeX:  0/4 preserved   (inline equations dropped)      │
│                                                                              │
╰──────────────────────────────────────────────────────────────────────────────╯
  > Saved → results/qwen3-vl-8b-markdown-20260222T151230.json
  > Outputs → results/qwen3-vl-8b-markdown-20260222T151230-outputs/
```

### Concurrency sweep — what's the optimal batch size?

Auto-ramp concurrency (1, 2, 4, 8, ...) until throughput plateaus:

```bash
vlmbench run -m Qwen/Qwen3-VL-8B-Instruct -i ./docs/ --sweep concurrency
```

```
╭─ Sweep: concurrency ─── Qwen/Qwen3-VL-8B-Instruct ─────────────────────────────────────╮
│                                                                                          │
│   Workers   Tok/s         Img/s   TTFT p50   TTFT p95   Latency p50   VRAM               │
│   ────────────────────────────────────────────────────────────────────────────            │
│         1     58.2 ▏        0.82     198 ms      312 ms       1.22 s   5.8 GB             │
│         2    112.4 ▎        1.58     210 ms      345 ms       1.26 s   5.9 GB             │
│         4    210.8 ▍        2.96     245 ms      489 ms       1.35 s   6.1 GB             │
│       ★ 8    398.1 ▋        5.60     312 ms      721 ms       1.43 s   6.4 GB             │
│        16    448.0 ▊        6.30     689 ms     1842 ms       2.54 s   7.2 GB             │
│        32    451.2 ▊        6.34    1421 ms     3901 ms       5.04 s   8.1 GB             │
│                                                                                          │
│   ★ Optimal    8 workers — 89% of peak tok/s, 2.6× lower p95 TTFT                       │
│   ↑ Peak      32 workers — 451.2 tok/s (saturated, +0.7% over 16)                       │
│                                                                                          │
╰──────────────────────────────────────────────────────────────────────────────────────────╯
  > Saved → results/qwen3-vl-8b-sweep-concurrency-20260222T160045.json
```

### Resolution impact — does 4K input actually help?

Sweep input resolution to find the quality/speed sweet spot:

```bash
# Speed-only sweep
vlmbench run -m Qwen/Qwen3-VL-8B-Instruct -i ./docs/ --sweep resolution

# Quality + speed (with ground truth scoring)
vlmbench run -m Qwen/Qwen3-VL-8B-Instruct -i ./docs/ \
  --sweep resolution --profile ocr --reference ./ground-truth/

# Or test the model's native max resolution (no downscaling)
vlmbench run -m Qwen/Qwen3-VL-8B-Instruct -i ./highres-scans/ \
  --resolution max --tag native-max
```

Combined `--sweep resolution --profile ocr --reference` renders both speed and quality:

```
╭─ Sweep: resolution ─── Qwen/Qwen3-VL-8B-Instruct ──────────────────────────────────────╮
│                                                                                          │
│   Max size   Tok/s         Img/s   TTFT p50   Tok/img   CER     WER                     │
│   ────────────────────────────────────────────────────────────────────                    │
│     512 px   892.1 █▊       12.40      89 ms       72    8.2%   14.1%                    │
│    1024 px   461.3 █         6.42     198 ms      181    3.1%    6.8%                    │
│  ★ 2048 px   198.7 ▍        2.76     412 ms      342    1.9%    4.2%                    │
│    4096 px    62.4 ▏        0.87    1204 ms      891    1.8%    4.0%                    │
│                                                                                          │
│   ★ Sweet spot   2048px — best quality/speed ratio (1.9% CER at 198.7 tok/s)            │
│   ↓ Diminishing  4096px — 3.2× slower for only 0.1% CER improvement                    │
│                                                                                          │
╰──────────────────────────────────────────────────────────────────────────────────────────╯
  > Saved → results/qwen3-vl-8b-sweep-resolution-20260222T162130.json
```

### Quantization comparison — FP16 vs FP8, what do I actually lose?

```bash
vlmbench run -m Qwen/Qwen3-VL-8B-Instruct     -i ./docs/ --max-concurrency 8 --quant fp16
vlmbench run -m Qwen/Qwen3-VL-8B-Instruct-FP8 -i ./docs/ --max-concurrency 8 --quant fp8

vlmbench compare results/*qwen3-vl-8b*.json
```

```
  compare (2 runs)

╭──────────────────────────────────┬────────┬────────┬─────────┬────────┬─────────┬───────┬──────────╮
│ Model                            │   TTFT │   TPOT │         │        │   Tok/  │       │          │
│                                  │   (ms) │   (ms) │ Tok/s ↓ │  Img/s │   img   │ Quant │     VRAM │
├──────────────────────────────────┼────────┼────────┼─────────┼────────┼─────────┼───────┼──────────┤
│ Qwen/Qwen3-VL-8B-Instruct-FP8   │    612 │   14.1 │   461.6 │   6.40 │     289 │ fp8   │ 11.75 GB │
│ Qwen/Qwen3-VL-8B-Instruct       │    698 │   17.2 │   448.0 │   6.40 │     289 │ fp16  │ 17.41 GB │
╰──────────────────────────────────┴────────┴────────┴─────────┴────────┴─────────┴───────┴──────────╯

╭─ Summary ────────────────────────────────────────────────────────────────────╮
│  Runs       2 across 1 model(s)  total duration 465.6s                       │
│  Tok/s      461.6 best   448.0 worst   454.8 avg                             │
│  FP8 vs FP16   +3.0% tok/s   −32.5% VRAM   −12.3% TTFT                      │
│  Errors     0                                                                │
╰──────────────────────────────────────────────────────────── vlmbench v0.2.5 ─╯
```

### Backend comparison — vLLM vs Ollama on the same model

```bash
vlmbench run -m Qwen/Qwen3-VL-2B-Instruct -i ./docs/ \
  --backend vllm-openai:latest --max-concurrency 4

vlmbench run -m Qwen/Qwen3-VL-2B-Instruct -i ./docs/ \
  --backend vllm --max-concurrency 4

vlmbench run -m qwen3-vl:2b -i ./docs/ \
  --backend ollama --max-concurrency 4

vlmbench compare results/*qwen3-vl-2b*.json
```

### Multi-GPU scaling — does tensor parallel actually help?

```bash
CUDA_VISIBLE_DEVICES=0 \
  vlmbench run -m Qwen/Qwen3-VL-8B-Instruct -i ./docs/ \
  --max-concurrency 8 --tag 1gpu

CUDA_VISIBLE_DEVICES=0,1 \
  vlmbench run -m Qwen/Qwen3-VL-8B-Instruct -i ./docs/ \
  --max-concurrency 8 --serve-args "--tensor-parallel-size 2" --tag 2gpu

CUDA_VISIBLE_DEVICES=0,1,2,3 \
  vlmbench run -m Qwen/Qwen3-VL-8B-Instruct -i ./docs/ \
  --max-concurrency 8 --serve-args "--tensor-parallel-size 4" --tag 4gpu

vlmbench compare results/*qwen3-vl-8b*.json
```

```
  compare (3 runs)

╭────────────────────────────────┬────────┬────────┬─────────┬────────┬─────────┬──────────┬───────╮
│ Model                          │   TTFT │   TPOT │         │        │   Tok/  │          │       │
│                                │   (ms) │   (ms) │ Tok/s ↓ │  Img/s │   img   │     VRAM │ Tag   │
├────────────────────────────────┼────────┼────────┼─────────┼────────┼─────────┼──────────┼───────┤
│ Qwen/Qwen3-VL-8B-Instruct     │    389 │    8.1 │   812.4 │  11.30 │     289 │ 12.2 GB  │ 4gpu  │
│                                │    498 │   12.4 │   598.2 │   8.32 │     289 │ 11.8 GB  │ 2gpu  │
│                                │    698 │   17.2 │   448.0 │   6.40 │     289 │ 17.4 GB  │ 1gpu  │
╰────────────────────────────────┴────────┴────────┴─────────┴────────┴─────────┴──────────┴───────╯

╭─ Summary ────────────────────────────────────────────────────────────────────╮
│  Runs       3 across 1 model(s)  total duration 698.2s                       │
│  Tok/s      812.4 best   448.0 worst   619.5 avg                             │
│  Scaling    1→2 GPU: +33.5% tok/s   2→4 GPU: +35.8% tok/s                   │
│  Errors     0                                                                │
╰──────────────────────────────────────────────────────────── vlmbench v0.2.5 ─╯
```

### Cloud vs local — is the API faster than self-hosting?

```bash
# Local vLLM
vlmbench run -m Qwen/Qwen3-VL-8B-Instruct -i ./docs/ \
  --max-concurrency 8 --tag local

# Together AI
vlmbench run -m Qwen/Qwen3-VL-8B-Instruct -i ./docs/ \
  --max-concurrency 8 --tag together \
  --base-url https://api.together.xyz/v1 --api-key $TOGETHER_API_KEY

vlmbench compare results/*qwen3-vl-8b*.json
```

### PDF pipeline — how does page count affect throughput?

```bash
vlmbench run -m lightonai/LightOnOCR-2-1B -i ./invoices-1page/ \
  --max-concurrency 8 --tag 1page
vlmbench run -m lightonai/LightOnOCR-2-1B -i ./contracts-10page/ \
  --max-concurrency 8 --tag 10page
vlmbench run -m lightonai/LightOnOCR-2-1B -i ./reports-50page/ \
  --max-concurrency 8 --tag 50page

vlmbench compare results/*lightonocr*.json
```

```
  compare (3 runs)

╭──────────────────────────────┬────────┬────────┬─────────┬────────┬─────────┬──────────┬────────╮
│ Model                        │   TTFT │   TPOT │         │        │   Tok/  │          │        │
│                              │   (ms) │   (ms) │ Tok/s ↓ │  Img/s │   img   │     VRAM │ Tag    │
├──────────────────────────────┼────────┼────────┼─────────┼────────┼─────────┼──────────┼────────┤
│ lightonai/LightOnOCR-2-1B    │    312 │    5.8 │  1721.4 │  10.12 │     170 │  5.6 GB  │ 1page  │
│                              │    467 │    6.0 │  1664.8 │   9.20 │     181 │  5.9 GB  │ 10page │
│                              │    891 │    6.2 │  1598.2 │   8.84 │     181 │  6.4 GB  │ 50page │
╰──────────────────────────────┴────────┴────────┴─────────┴────────┴─────────┴──────────┴────────╯

╭─ Summary ────────────────────────────────────────────────────────────────────╮
│  Runs       3 across 1 model(s)  total duration 486.2s                       │
│  Tok/s      1721.4 best   1598.2 worst   1661.5 avg                          │
│  Scaling    1pg → 50pg: −7.2% tok/s   +185% TTFT   +14% VRAM                │
│  Errors     0                                                                │
╰──────────────────────────────────────────────────────────── vlmbench v0.2.5 ─╯
```

### HuggingFace dataset evaluation — standard benchmarks

```bash
vlmbench run -m Qwen/Qwen3-VL-8B-Instruct \
  -i hf://datasets/vidore/docvqa_test_subsampled \
  --max-concurrency 4 --runs 1
```

### Full matrix — answer everything at once

Define a benchmark matrix in TOML — model × backend × concurrency — and run it all with one command:

```bash
vlmbench init --preset ocr-shootout > bench.toml
```

```toml
# bench.toml
[matrix]
models = [
  "lightonai/LightOnOCR-2-1B",
  "Qwen/Qwen3-VL-8B-Instruct",
  "Qwen/Qwen3-VL-8B-Instruct-FP8",
  "rednote-hilab/dots.ocr",
]
concurrency = [1, 4, 8]
backends = ["vllm-openai:latest"]

[inputs]
path = "./docs/"
profile = "ocr"
reference = "./ground-truth/"
runs = 3

[output]
save = "./results/"
format = "json"
```

```bash
vlmbench matrix bench.toml
```

```
  matrix bench.toml — 4 models × 3 concurrency = 12 runs

  ✓  lightonai/LightOnOCR-2-1B        concurrency=1   1m 02s
  ✓  lightonai/LightOnOCR-2-1B        concurrency=4   0m 48s
  ✓  lightonai/LightOnOCR-2-1B        concurrency=8   0m 41s
  ✓  Qwen/Qwen3-VL-8B-Instruct       concurrency=1   3m 12s
  ✓  Qwen/Qwen3-VL-8B-Instruct       concurrency=4   1m 54s
  ●  Qwen/Qwen3-VL-8B-Instruct       concurrency=8   running...
  ·  Qwen/Qwen3-VL-8B-Instruct-FP8   concurrency=1
  ·  Qwen/Qwen3-VL-8B-Instruct-FP8   concurrency=4
  ·  Qwen/Qwen3-VL-8B-Instruct-FP8   concurrency=8
  ·  rednote-hilab/dots.ocr           concurrency=1
  ·  rednote-hilab/dots.ocr           concurrency=4
  ·  rednote-hilab/dots.ocr           concurrency=8

  Progress: 6/12 runs complete   elapsed 8m 37s   est. remaining ~9m
```

After all runs complete, view the full comparison and launch the dashboard:

```bash
vlmbench compare results/*.json
vlmbench dashboard results/ --open
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
| PDF | `.pdf` | `pypdfium2` per-page -> base64 |
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
- ffmpeg (video input) — optional
