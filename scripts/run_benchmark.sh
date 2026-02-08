#!/usr/bin/env bash
# Run vlmbench on a GPU and optionally upload results.
#
# Local:    ./scripts/run_benchmark.sh --model Qwen/Qwen3-VL-2B-Instruct --input ./images --no-upload
# HF Jobs:  make hf-benchmark MODEL=Qwen/Qwen3-VL-2B-Instruct FLAVOR=l4x1
set -euo pipefail

# ── Defaults ─────────────────────────────────────────────────────────
MODEL=""
GPU_FLAVOR=""
REPO_ID="vlm-run/vlmbench-results"
SERVE_ARGS=""
SAVE_DIR="/tmp/vlmbench-results"
INPUT=""
NO_UPLOAD=false

# ── Parse args ───────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case $1 in
        --model)        MODEL="$2";      shift 2 ;;
        --gpu-flavor)   GPU_FLAVOR="$2"; shift 2 ;;
        --repo-id)      REPO_ID="$2";    shift 2 ;;
        --serve-args)   SERVE_ARGS="$2"; shift 2 ;;
        --input)        INPUT="$2";      shift 2 ;;
        --save)         SAVE_DIR="$2";   shift 2 ;;
        --no-upload)    NO_UPLOAD=true;  shift   ;;
        --help|-h)
            echo "Usage: $0 --model MODEL --input PATH [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --model        Model ID (required, e.g. Qwen/Qwen3-VL-2B-Instruct)"
            echo "  --input        Input image/dir (required)"
            echo "  --gpu-flavor   GPU flavor tag for results (e.g. l4x1, a10g-small)"
            echo "  --repo-id      HF dataset repo for upload (default: vlm-run/vlmbench-results)"
            echo "  --serve-args   Extra vLLM serve args (e.g. '--max-model-len 4096')"
            echo "  --save         Output directory (default: /tmp/vlmbench-results)"
            echo "  --no-upload    Skip uploading results to HF"
            exit 0
            ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

[[ -z "$MODEL" ]] && echo "Error: --model is required" && exit 1
[[ -z "$INPUT" ]] && echo "Error: --input is required" && exit 1

# ── Run vlmbench ─────────────────────────────────────────────────────
mkdir -p "$SAVE_DIR"

BENCH_ARGS=(--model "$MODEL" --input "$INPUT" --serve --save "$SAVE_DIR")
[[ -n "$GPU_FLAVOR" ]]  && BENCH_ARGS+=(--tag "$GPU_FLAVOR")
[[ -n "$SERVE_ARGS" ]]  && BENCH_ARGS+=(--serve-args "$SERVE_ARGS")

echo "Running: vlmbench run ${BENCH_ARGS[*]}"
vlmbench run "${BENCH_ARGS[@]}"

# ── Upload results ───────────────────────────────────────────────────
if [[ "$NO_UPLOAD" == false ]]; then
    RESULT_FILE=$(ls -t "$SAVE_DIR"/*.json 2>/dev/null | head -1)
    if [[ -n "$RESULT_FILE" ]]; then
        FILENAME=$(basename "$RESULT_FILE")
        UPLOAD_PATH="results/${GPU_FLAVOR:-local}/${FILENAME}"
        echo "Uploading $FILENAME -> $REPO_ID/$UPLOAD_PATH"
        huggingface-cli upload "$REPO_ID" "$RESULT_FILE" "$UPLOAD_PATH" --repo-type dataset
    else
        echo "WARNING: No result JSON files found in $SAVE_DIR"
    fi
fi

echo "Done."
