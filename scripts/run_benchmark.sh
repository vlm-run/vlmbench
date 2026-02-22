#!/usr/bin/env bash
# Run vlmbench on a GPU and optionally upload results.
#
# Local:    ./scripts/run_benchmark.sh --model Qwen/Qwen3-VL-2B-Instruct --input ./images
# Dataset:  ./scripts/run_benchmark.sh --model Qwen/Qwen3-VL-2B-Instruct --dataset hf://vlm-run/FineVision-vlmbench-mini
# HF Jobs:  make hf-benchmark MODEL=Qwen/Qwen3-VL-2B-Instruct FLAVOR=l4x1
set -euo pipefail

# ── Defaults ─────────────────────────────────────────────────────────
REPO_ID="vlm-run/vlmbench-results"
SAVE_DIR="/tmp/vlmbench-results"
UPLOAD=false
BENCH_ARGS=()

# ── Parse args ───────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case $1 in
        --upload)       UPLOAD=true;     shift   ;;
        --no-upload)    UPLOAD=false;    shift   ;;
        --repo-id)      REPO_ID="$2";    shift 2 ;;
        --save)         SAVE_DIR="$2";   BENCH_ARGS+=(--save "$SAVE_DIR"); shift 2 ;;
        --help|-h)
            echo "Usage: $0 [--upload] [--repo-id REPO] -- <vlmbench args>"
            echo ""
            echo "Wrapper options:"
            echo "  --upload       Upload results to HF dataset repo"
            echo "  --no-upload    Don't upload results (default)"
            echo "  --repo-id      HF dataset repo for upload (default: vlm-run/vlmbench-results)"
            echo "  --save         Output directory (default: /tmp/vlmbench-results)"
            echo ""
            echo "All other args are passed through to vlmbench run."
            exit 0
            ;;
        *)              BENCH_ARGS+=("$1"); shift ;;
    esac
done

# ── Run vlmbench ─────────────────────────────────────────────────────
mkdir -p "$SAVE_DIR"

# Add --save if not already specified
if [[ ! " ${BENCH_ARGS[*]} " =~ " --save " ]]; then
    BENCH_ARGS+=(--save "$SAVE_DIR")
fi

VLMBENCH_PKG="vlmbench @ git+https://github.com/vlm-run/vlmbench.git@spillai/hf-jobs-gpu-sweep"
echo "Running: uvx --from '$VLMBENCH_PKG' vlmbench run ${BENCH_ARGS[*]}"
uvx --from "$VLMBENCH_PKG" vlmbench run "${BENCH_ARGS[@]}"

# ── Upload results ───────────────────────────────────────────────────
if [[ "$UPLOAD" == true ]]; then
    RESULT_FILE=$(ls -t "$SAVE_DIR"/*.json 2>/dev/null | head -1)
    if [[ -n "$RESULT_FILE" ]]; then
        FILENAME=$(basename "$RESULT_FILE")
        # Extract tag from filename or use 'local'
        TAG=$(echo "$FILENAME" | grep -oP '(?<=_)[^_]+(?=_\d{8})' || echo "local")
        UPLOAD_PATH="results/${TAG}/${FILENAME}"
        echo "Uploading $FILENAME -> $REPO_ID/$UPLOAD_PATH"
        huggingface-cli upload "$REPO_ID" "$RESULT_FILE" "$UPLOAD_PATH" --repo-type dataset
    else
        echo "WARNING: No result JSON files found in $SAVE_DIR"
    fi
fi

echo "Done."
