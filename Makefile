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
	@echo "Benchmarking:"
	@echo "  benchmark      Run benchmark locally (requires vllm + GPU)"
	@echo "  hf-benchmark   Submit single HF Jobs benchmark"
	@echo "  hf-sweep       Submit benchmarks across multiple GPU flavors"
	@echo ""
	@echo "Examples:"
	@echo "  make benchmark MODEL=Qwen/Qwen3-VL-2B-Instruct"
	@echo "  make hf-benchmark MODEL=Qwen/Qwen3-VL-2B-Instruct FLAVOR=l4x1"
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

# ── HF Jobs Benchmarking ─────────────────────────────────────────────
HF_TIMEOUT ?= 45m
HF_IMAGE   ?= vllm/vllm-openai:latest
REPO_ID    ?= vlm-run/vlmbench-results
INPUT      ?= https://storage.googleapis.com/vlm-data-public-prod/hub/examples/image.caption/car.jpg
SERVE_ARGS ?=
FLAVORS    ?= t4-small l4x1 a10g-small

# Run benchmark locally (requires vllm installed, NVIDIA GPU)
# Usage: make benchmark MODEL=Qwen/Qwen3-VL-2B-Instruct
benchmark:
ifndef MODEL
	$(error MODEL is required. Example: make benchmark MODEL=Qwen/Qwen3-VL-2B-Instruct)
endif
	bash scripts/run_benchmark.sh \
		--model $(MODEL) \
		--no-upload \
		$(if $(SERVE_ARGS),--serve-args '$(SERVE_ARGS)',) \
		$(if $(INPUT),--input $(INPUT),) \
		$(if $(DATASET),--dataset $(DATASET),)

# Submit simple benchmark to HF Jobs (uses inline uv script)
# Usage: make hf-benchmark FLAVOR=l4x1
hf-benchmark:
ifndef FLAVOR
	$(error FLAVOR is required. Example: make hf-benchmark FLAVOR=l4x1)
endif
	uvx hf jobs run \
		--flavor $(FLAVOR) \
		--secrets HF_TOKEN \
		--timeout $(HF_TIMEOUT) \
		uv run scripts/simple_benchmark.py

# Sweep across multiple GPU flavors
# Usage: make hf-sweep MODEL=Qwen/Qwen3-VL-2B-Instruct FLAVORS="t4-small l4x1 a10g-small"
hf-sweep:
ifndef MODEL
	$(error MODEL is required. Example: make hf-sweep MODEL=Qwen/Qwen3-VL-2B-Instruct)
endif
	@echo "Launching $(words $(FLAVORS)) jobs for $(MODEL)..."
	@for flavor in $(FLAVORS); do \
		echo "  -> $$flavor"; \
		$(MAKE) --no-print-directory hf-benchmark MODEL=$(MODEL) FLAVOR=$$flavor SERVE_ARGS='$(SERVE_ARGS)' & \
	done; \
	wait
	@echo "All jobs submitted."

.PHONY: default help clean clean-build clean-pyc lint test dist benchmark hf-benchmark hf-sweep
