# `vlmbench eval` — the full surface on this branch

Every flag below exists today. `-p` defaults to `vlmrun`, so it is omitted unless
the example is about providers.

## Discovery — what can I point at?

```bash
vlmbench eval --list                      # evals, providers, models, prompts (builtin + any -c)
vlmbench eval --list-models -p vlmrun     # the ids this provider actually serves
vlmbench eval --list-models -p openrouter # OpenRouter, filtered to image-input models
vlmbench eval -d parsebench --dry-run -m qwen/qwen3.8-27b   # matrix + request count, spends nothing
```

## One model, one suite

```bash
vlmbench eval -d olmocr-bench -m zai-org/glm-ocr
vlmbench eval -d olmocr-bench -m zai-org/glm-ocr --limit 10 --stratify
vlmbench eval -d olmocr-bench -m zai-org/glm-ocr --limit 5%
```

`--limit` takes `N`, `N%` or a fraction. `--stratify` spreads the budget across
splits round-robin instead of taking the first N. Any limited run is branded
**PARTIAL** in every output and is not comparable to published numbers.

## The four built-in suites

```bash
vlmbench eval -d parsebench        -m rednote-hilab/dots.mocr --limit 10 --stratify
vlmbench eval -d olmocr-bench      -m rednote-hilab/dots.mocr --limit 10 --stratify
vlmbench eval -d omni-parse-bench  -m rednote-hilab/dots.mocr --limit 10 --stratify
vlmbench eval -d omnidocbench      -m rednote-hilab/dots.mocr --limit 10 --stratify
```

Short ids are a convenience. The full dataset id is always accepted, and is the
explicit form worth writing down in a script:

```bash
vlmbench eval -d hf://datasets/allenai/olmOCR-bench -m zai-org/glm-ocr --limit 10
```

## Prompts

```bash
vlmbench eval -d parsebench -m qwen/qwen3.8-27b --prompt markdown
vlmbench eval -d parsebench -m qwen/qwen3.8-27b --prompt markdown-tuned
vlmbench eval -d parsebench -m qwen/qwen3.8-27b --prompt prompts/ocr.py:Anchored
```

`--prompt` applies to a single `-m` model. A prompt is a registry name or a
`path.py:ClassName`. Models whose entry sets `takes_prompt=false` (the dedicated
OCR models) ignore it — the suite's own default prompt is used for them.

## Several rows at once — a run file

```python
# runs/ocr.py
from vlmbench import Run

RUNS = [
    Run(model="rednote-hilab/dots.mocr"),
    Run(model="zai-org/glm-ocr"),
    Run(model="qwen/qwen3.8-27b", prompt="markdown"),
    Run(model="qwen/qwen3.8-27b", prompt="markdown-tuned"),
    Run(provider="openrouter", model="qwen/qwen3-vl-8b-instruct", prompt="markdown-tuned"),
]

SMOKE = [Run(model="zai-org/glm-ocr")]
```

```bash
vlmbench eval -d parsebench -m runs/ocr.py --limit 10 --stratify
vlmbench eval -d parsebench -m runs/ocr.py:SMOKE --limit 10
```

Rows run in the order written and nothing multiplies — a row exists because
someone wrote that line. A `Run` may override the provider for itself.

## Providers

```bash
vlmbench eval -d parsebench -m qwen/qwen3-vl-8b-instruct -p openrouter --limit 10
vlmbench eval -d parsebench -m <openai-model-id> -p openai --limit 10
vlmbench eval -d parsebench -m qwen2.5vl:7b -p ollama --limit 10      # local, cost reported as —
vlmbench eval -d parsebench -m Qwen/Qwen3-VL-8B -p vllm --limit 10    # local, cost reported as —
```

Local backends never get an imputed cost: `$/1K` prints `—` rather than a
guessed GPU COGS.

## Request mechanics

```bash
vlmbench eval -d parsebench -m qwen/qwen3.8-27b --extra-body '{"top_p": 0.8}'
vlmbench eval -d parsebench -m zai-org/glm-ocr --concurrency 4
```

`--extra-body` is merged into every request body (last wins), on top of whatever
the model's catalog entry already sets. `--concurrency` caps in-flight requests;
it defaults to the provider's own cap, which is what keeps a run under the
provider's rate limit.

## Caching

```bash
vlmbench eval -d parsebench -m zai-org/glm-ocr --limit 10               # cache hit = free
vlmbench eval -d parsebench -m zai-org/glm-ocr --limit 10 --no-cache    # re-request everything
vlmbench eval -d parsebench -m zai-org/glm-ocr --limit 10 --score-only  # re-score, zero network
```

The cache key fingerprints provider, model id, rendered prompt text,
`extra_body` and the sample — so editing a prompt misses the cache by
construction, and a scorer fix can be re-applied to old output for free with
`--score-only`.

## Bringing your own catalog

```bash
vlmbench eval -d parsebench -m my-model -c catalog/fireworks.py
vlmbench eval -d parsebench -m my-model -c catalog/           # a whole directory
vlmbench eval -d catalog/invoices.py -m zai-org/glm-ocr       # your own eval, as -d
vlmbench eval -d parsebench -m my-model -c catalog/a.py -c catalog/b.py:MyModel
```

`-c` is repeatable and takes a file, a directory, or `file.py:ClassName`.
Nothing is installed — these are plain Python files loaded by import.

## Reporting

```bash
vlmbench report                                      # every run under ~/.vlmbench/evals
vlmbench report -o board.html                        # self-contained HTML leaderboard
vlmbench report runs/*/result.json -o board.html
vlmbench report -o board.md --format md
vlmbench report -o board.csv --format csv
vlmbench report -o viewer.html --viewer              # empty page; drop result.json files onto it
vlmbench report -o board.html --title "OCR board" --mixed
```

`--mixed` is required to put PARTIAL and full runs in one table, so the two
never silently blend. `--viewer` emits a page with no data baked in: it imports
`result.json` files client-side, which is how a result from another machine gets
looked at without a server.

## Output layout

```
~/.vlmbench/evals/<suite>-<YYMMDD>-<hash>/
  result.json          # scores, totals, provenance
  outputs/<provider>--<model>--<prompt>/<sample>.md   # exactly what the model returned
```

Keeping raw output next to the scores is what makes a surprising number
checkable — every example above can be audited by reading the `.md` the model
actually produced.
