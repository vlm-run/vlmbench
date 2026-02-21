# Roadmap

## Backends

- [ ] SGLang backend — currently stubbed, wire up Docker-based `lmsysorg/sglang` serving
- [ ] Ollama structured output — support guided decoding / JSON mode on Ollama
- [ ] Backend-specific tuning presets — auto-apply known-good `--serve-args` per model/backend pair
- [ ] TensorRT-LLM backend via Triton Inference Server

## Sweep mode

- [ ] `--sweep concurrency` — auto-ramp concurrency (1, 2, 4, 8, …) until throughput plateaus
- [ ] `--sweep resolution` — benchmark same model at 512, 1024, 2048, 4096px in one run
- [ ] `--sweep max-tokens` — vary output length (256, 512, 1024, 2048, 4096) to measure scaling
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

## OCR / markdown profiling

- [ ] `--profile ocr` and `--profile markdown` modes — evaluate output quality, not just speed
- [ ] Token-level output metrics: total output tokens, tokens per image, output tok/s
- [ ] `--save-outputs` — persist raw model output text alongside metrics JSON
- [ ] Character-level accuracy — CER/WER against ground-truth when `--reference` provided
- [ ] Markdown structure fidelity — heading, table, list, and LaTeX preservation scores
- [ ] Side-by-side output diff in `vlmbench compare` for qualitative inspection
- [ ] Combined sweep + profile: `--sweep resolution --profile ocr --reference ./gt/` to find the quality/speed sweet spot

## Max image resolution support

- [ ] Per-model max resolution detection — query model config or `models.toml` for native resolution cap
- [ ] `--resolution max` flag — skip downscaling entirely, send at model's native max
- [ ] Resolution metadata in results JSON — actual input resolution per image after any resizing
- [ ] Warn when input exceeds model's max resolution (risk of silent truncation or OOM)

## Benchmarking features

- [ ] Multi-image benchmarks — send N images per request, measure scaling
- [ ] PDF page-count sweep — measure throughput vs document length
- [ ] Prompt-length sensitivity — vary system/user prompt size, measure impact on TTFT
- [ ] Structured output benchmarks — JSON mode / guided decoding throughput
- [ ] Streaming vs non-streaming comparison
- [ ] KV cache hit rate tracking (for backends that report it)
- [ ] Multi-GPU scaling — benchmark same model across 1, 2, 4 GPUs with tensor parallel

## Developer experience

- [ ] `vlmbench init` — scaffold a `bench.toml` config for repeatable benchmark suites
- [ ] `vlmbench init --preset ocr-shootout` — pre-built presets for common experiment types
- [ ] `vlmbench matrix bench.toml` — run full model × backend × concurrency matrix from TOML config
- [ ] `vlmbench diff <a.json> <b.json>` — side-by-side delta with % change and significance
- [ ] Machine-readable output — `--format csv` and `--format jsonl` for pipeline integration
- [ ] `--tag` enrichment — auto-tag with GPU name, quant, concurrency for easier filtering
- [ ] GitHub Actions template for PR-gated perf regression checks
