# Roadmap

## Backends

- [ ] SGLang backend — currently stubbed, wire up Docker-based `lmsysorg/sglang` serving
- [ ] Ollama structured output — support guided decoding / JSON mode on Ollama
- [ ] Backend-specific tuning presets — auto-apply known-good `--serve-args` per model/backend pair
- [ ] TensorRT-LLM backend via Triton Inference Server

## Saturated throughput

- [ ] `--sweep-concurrency` — auto-ramp concurrency (1, 2, 4, 8, …) until throughput plateaus
- [ ] Report saturation point: max tok/s, concurrency at saturation, latency-at-saturation
- [ ] Goodput metric — throughput excluding errored / timed-out requests
- [ ] Queue-depth vs latency curve in results JSON

## HTML dashboard

- [ ] `vlmbench dashboard` — single self-contained HTML file (inline JS/CSS, no deps)
- [ ] Reads `results/*.json`, renders:
  - Throughput bar chart (tok/s by model × backend)
  - Latency percentile plot (p50/p95/p99 TTFT & TPOT)
  - VRAM usage comparison
  - Saturation curve (concurrency vs throughput) when sweep data present
- [ ] Filterable by model, backend, quantization, GPU
- [ ] `--open` flag to launch in default browser

## HuggingFace jobs + uv scripts

- [ ] `hfjobs/` directory with self-contained `uv run` scripts (PEP 723 metadata)
- [ ] One script per profiling task: single-model bench, sweep, compare, matrix
- [ ] `huggingface_hub` integration — push results to HF dataset repo
- [ ] `vlmbench run --push-to-hub <org>/<dataset>` — upload result JSON after each run
- [ ] CI template for scheduled HF Jobs (nightly benchmarks on new model releases)

## VLM compatibility matrix

- [ ] Machine-readable `models.toml` — full catalog of tested VLMs:
  - Model name, param count, quantizations (fp16, fp8, awq, gptq, gguf variants)
  - Supported backends (vLLM, Ollama, SGLang) with per-backend serve-args
  - Input capabilities: single image, multi-image, PDF, video
  - Max image resolution tested, max context length
  - Known issues / version constraints
- [ ] `vlmbench models` — pretty-print the matrix with filters (`--backend vllm`, `--capability multi-image`)
- [ ] `vlmbench models --json` — dump for programmatic use

## Benchmarking features

- [ ] Multi-image benchmarks — send N images per request, measure scaling
- [ ] Resolution sweep — benchmark same model at 512, 1024, 2048, 4096px
- [ ] PDF page-count sweep — measure throughput vs document length
- [ ] Prompt-length sensitivity — vary system/user prompt size, measure impact on TTFT
- [ ] Structured output benchmarks — JSON mode / guided decoding throughput
- [ ] Streaming vs non-streaming comparison
- [ ] KV cache hit rate tracking (for backends that report it)
- [ ] Multi-GPU scaling — benchmark same model across 1, 2, 4 GPUs with tensor parallel

## Developer experience

- [ ] `vlmbench init` — scaffold a `bench.toml` config for repeatable benchmark suites
- [ ] Config file support — define model × backend × concurrency matrix in TOML, run all at once
- [ ] `vlmbench diff <a.json> <b.json>` — side-by-side delta with % change and significance
- [ ] Machine-readable output — `--format csv` and `--format jsonl` for pipeline integration
- [ ] GitHub Actions template for PR-gated perf regression checks
