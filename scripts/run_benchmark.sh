#!/usr/bin/env bash
# Upload vlmbench result JSONs to a HuggingFace dataset repo.
#
# Usage: ./scripts/upload_results.sh results/*.json
#        ./scripts/upload_results.sh --repo-id my-org/my-results results/*.json
set -euo pipefail

REPO_ID="vlm-run/vlmbench-results"

# ── Parse args ───────────────────────────────────────────────────────
FILES=()
while [[ $# -gt 0 ]]; do
    case $1 in
        --repo-id)  REPO_ID="$2"; shift 2 ;;
        --help|-h)
            echo "Usage: $0 [--repo-id REPO] <json-files...>"
            echo ""
            echo "Upload vlmbench result JSON files to a HuggingFace dataset repo."
            echo ""
            echo "Options:"
            echo "  --repo-id    HF dataset repo (default: vlm-run/vlmbench-results)"
            exit 0
            ;;
        *)          FILES+=("$1"); shift ;;
    esac
done

if [[ ${#FILES[@]} -eq 0 ]]; then
    echo "ERROR: No JSON files specified. Usage: $0 results/*.json"
    exit 1
fi

# ── Upload ───────────────────────────────────────────────────────────
for f in "${FILES[@]}"; do
    if [[ ! -f "$f" ]]; then
        echo "WARNING: File not found: $f"
        continue
    fi
    FILENAME=$(basename "$f")
    UPLOAD_PATH="results/${FILENAME}"
    echo "Uploading $FILENAME -> $REPO_ID/$UPLOAD_PATH"
    python3 -c "import sys; from huggingface_hub import HfApi; HfApi().upload_file(path_or_fileobj=sys.argv[1], path_in_repo=sys.argv[2], repo_id=sys.argv[3], repo_type='dataset')" \
        "$f" "$UPLOAD_PATH" "$REPO_ID"
done

echo "Done."
