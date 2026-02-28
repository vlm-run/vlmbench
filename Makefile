default: help

help:
	@echo "Usage: make <target>"
	@echo ""
	@echo "Targets:"
	@echo "  clean          Remove build artifacts"
	@echo "  lint           Run pre-commit hooks"
	@echo "  test           Run tests"
	@echo "  dist           Build distribution"
	@echo ""
	@echo "Server:"
	@echo "  serve          Start a vLLM Docker server for testing"
	@echo "  serve-stop     Stop and remove the vllm-serve container"
	@echo ""
	@echo "Benchmarking:"
	@echo "  benchmark      Run benchmark locally using local branch (uv run vlmbench)"
	@echo "  uv-benchmark   Run benchmark via run_benchmark.sh (uvx, mimics HF Jobs)"
	@echo "  hf-benchmark   Submit single HF Jobs benchmark"
	@echo "  hf-sweep       Submit benchmarks across multiple GPU flavors"
	@echo ""
	@echo "Examples:"
	@echo "  make serve MODEL=Qwen/Qwen3-VL-2B-Instruct"
	@echo "  make benchmark MODEL=Qwen/Qwen3-VL-2B-Instruct"
	@echo "  make benchmark MODEL=Qwen/Qwen3-VL-2B-Instruct BENCHMARK_ARGS='--dataset vlm-run/FineVision-vlmbench-mini --max-concurrency 16'"
	@echo "  make hf-benchmark MODEL=Qwen/Qwen3-VL-2B-Instruct FLAVOR=l4x1 HF_NAMESPACE=vlm-run"
	@echo "  make hf-sweep MODEL=Qwen/Qwen3-VL-2B-Instruct FLAVORS='t4-small l4x1 a10g-small'"

clean: clean-build clean-pyc

clean-build:
	rm -rf build/ dist/ *.egg-info .eggs/ site/
	find . -name '*.egg-info' -exec rm -rf {} +
	find . -name '*.egg' -delete

clean-pyc:
	find . -name '*.pyc' -delete
	find . -name '*.pyo' -delete
	find . -name '__pycache__' -type d -exec rm -rf {} +

lint:
	pre-commit run --all-files

test:
	pytest -sv tests

dist: clean
	python -m build --sdist --wheel

# ── Serve settings ────────────────────────────────────────────────────
VLLM_IMAGE ?= vllm/vllm-openai:v0.15.1
SERVE_PORT ?= 8000
SERVE_ARGS ?=
GPU_DEVICE ?= 1
# LD_FIX: prepend host driver libs so the container's stale CUDA compat
#         libs (e.g. 575.x) don't shadow the host driver (e.g. 580.x).
LD_FIX     := /usr/lib/x86_64-linux-gnu:/usr/local/nvidia/lib64:/usr/local/cuda/lib64
HF_CACHE   ?= $(HOME)/.cache/huggingface
VLLM_CACHE ?= $(HOME)/.cache/vllm

# ── Benchmarking settings ────────────────────────────────────────────
BENCHMARK_ARGS        ?=
BENCHMARK_JOB_TIMEOUT ?= 45m
HF_NAMESPACE          ?= vlm-run
REPO_ID               ?= vlm-run/vlmbench-results
FLAVORS               ?= a100-large
CONCURRENCY           ?= 8

# ── Serve (vLLM Docker for testing) ──────────────────────────────────
# Usage: make serve MODEL=Qwen/Qwen3-VL-2B-Instruct
#        make serve MODEL=Qwen/Qwen3-VL-2B-Instruct VLLM_IMAGE=<custom-image>
#        make serve SERVE_PORT=8100 GPU_DEVICE=0
serve:
ifndef MODEL
	$(error MODEL is required. Example: make serve MODEL=Qwen/Qwen3-VL-2B-Instruct)
endif
	@echo "Starting $(VLLM_IMAGE) on :$(SERVE_PORT) (GPU $(GPU_DEVICE))..."
	docker run -d --name vllm-serve \
		--runtime nvidia --gpus '"device=$(GPU_DEVICE)"' \
		-e LD_LIBRARY_PATH=$(LD_FIX) \
		-e CUDA_DEVICE_ORDER=PCI_BUS_ID \
		-v $(HF_CACHE):/root/.cache/huggingface \
		-v $(VLLM_CACHE):/root/.cache/vllm \
		-p $(SERVE_PORT):8000 --ipc=host \
		$(VLLM_IMAGE) \
		--model $(MODEL) $(SERVE_ARGS)
	@echo "Container 'vllm-serve' started. Logs: docker logs -f vllm-serve"

serve-stop:
	docker rm -f vllm-serve 2>/dev/null || true

# ── Benchmarking ─────────────────────────────────────────────────────
# Run benchmark locally using local branch (uv run vlmbench).
# All vlmbench args go in BENCHMARK_ARGS. See `uv run vlmbench run --help`.
#
# Examples:
#   make benchmark MODEL=Qwen/Qwen3-VL-2B-Instruct
#   make benchmark MODEL=Qwen/Qwen3-VL-2B-Instruct BENCHMARK_ARGS="--dataset vlm-run/FineVision-vlmbench-mini"
#   make benchmark MODEL=Qwen/Qwen3-VL-2B-Instruct BENCHMARK_ARGS="--max-concurrency 16 --runs 5"
#   make benchmark MODEL=Qwen/Qwen3-VL-2B-Instruct BENCHMARK_ARGS="--input ./images --save ./results --tag my-run"
#   make benchmark MODEL=Qwen/Qwen3-VL-2B-Instruct BENCHMARK_ARGS="--base-url http://localhost:8100/v1 --no-serve"
#   make benchmark MODEL=Qwen/Qwen3-VL-2B-Instruct BENCHMARK_ARGS="--backend vllm-openai:latest --serve"
benchmark:
ifndef MODEL
	$(error MODEL is required. Example: make benchmark MODEL=Qwen/Qwen3-VL-2B-Instruct)
endif
	uv run vlmbench run --model $(MODEL) $(BENCHMARK_ARGS)

# Run benchmark via run_benchmark.sh (uvx from PyPI, mimics HF Jobs)
# Usage: make uv-benchmark MODEL=Qwen/Qwen3-VL-2B-Instruct
#        make uv-benchmark MODEL=Qwen/Qwen3-VL-2B-Instruct BENCHMARK_ARGS="--dataset vlm-run/FineVision-vlmbench-mini"
uv-benchmark:
ifndef MODEL
	$(error MODEL is required. Example: make uv-benchmark MODEL=Qwen/Qwen3-VL-2B-Instruct)
endif
	bash scripts/run_benchmark.sh --model $(MODEL) --no-upload $(BENCHMARK_ARGS)

# Submit benchmark to HF Jobs
# Usage: make hf-benchmark FLAVOR=l4x1 MODEL=Qwen/Qwen3-VL-2B-Instruct
#        make hf-benchmark FLAVOR=l4x1 MODEL=Qwen/Qwen3-VL-2B-Instruct BENCHMARK_ARGS="--dataset vlm-run/FineVision-vlmbench-mini"
hf-benchmark:
ifndef FLAVOR
	$(error FLAVOR is required. Example: make hf-benchmark FLAVOR=l4x1)
endif
ifndef MODEL
	$(error MODEL is required. Example: make hf-benchmark FLAVOR=l4x1 MODEL=Qwen/Qwen3-VL-2B-Instruct)
endif
	uvx hf jobs run \
		--namespace $(HF_NAMESPACE) \
		--flavor $(FLAVOR) \
		--secrets HF_TOKEN \
		--timeout $(BENCHMARK_JOB_TIMEOUT) \
		-- $(VLLM_IMAGE) \
		bash -c 'pip install -q vlmbench==0.3.4 \
			&& vllm serve $(MODEL) --max-model-len 8192 & \
			echo "Starting vLLM server in background..." \
			&& for i in $$(seq 1 120); do \
				curl -sf http://localhost:8000/health > /dev/null 2>&1 && break; \
				echo "  Waiting for vLLM... ($${i}/120)"; sleep 5; \
			done \
			&& curl -sf http://localhost:8000/health > /dev/null 2>&1 \
			&& echo "vLLM server ready!" \
			&& vlmbench run --model $(MODEL) --no-serve --base-url http://localhost:8000/v1 --max-concurrency $(CONCURRENCY) --tag c$(CONCURRENCY) $(BENCHMARK_ARGS) \
			|| echo "ERROR: vLLM server failed to start"'

# Sweep across multiple GPU flavors
# Usage: make hf-sweep MODEL=Qwen/Qwen3-VL-2B-Instruct FLAVORS="t4-small l4x1 a10g-small"
hf-sweep:
ifndef MODEL
	$(error MODEL is required. Example: make hf-sweep MODEL=Qwen/Qwen3-VL-2B-Instruct)
endif
	@echo "Launching $(words $(FLAVORS)) jobs for $(MODEL)..."
	@for flavor in $(FLAVORS); do \
		echo "  -> $$flavor"; \
		$(MAKE) --no-print-directory hf-benchmark MODEL=$(MODEL) FLAVOR=$$flavor BENCHMARK_ARGS='$(BENCHMARK_ARGS)' & \
	done; \
	wait
	@echo "All jobs submitted."

.PHONY: default help clean clean-build clean-pyc lint test dist serve serve-stop benchmark uv-benchmark hf-benchmark hf-sweep
