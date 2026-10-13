# Evals

Accuracy benchmarking for OCR and document VLMs. `vlmbench run` measures speed; `vlmbench eval` measures whether the output is *right*.

```bash
uvx vlmbench eval -d olmocr-bench -m zai-org/glm-ocr --limit 10
```

## Concepts

| Concept | What it is | Where it lives |
|---|---|---|
| **Eval** | A dataset plus how to score it — `samples()` and `score()` | `vlmbench/evals/*.py` or your own `.py` |
| **Sample** | One page: media, split, ground truth, prompt context | yielded by `Eval.samples()` |
| **Prompt** | Model-agnostic prompt text, named per row | a prompt file |
| **Run** | One declared row: model, optional provider and prompt | a run file |
| **Model** | Request mechanics for one model id — carries no prompt | `vlmbench/models.py` or a catalog file |
| **Provider** | Where requests go, plus optional hooks | `vlmbench/providers.py` or a catalog file |
| **Verdict** | `(column, value)` — `bool` counts a test, `float` averages, `None` is unsupported | yielded by `Eval.score()` |

- A **row** on the leaderboard is one `Run`: a `(provider, model, prompt)` triple.
- A **column** is either boolean (a pass rate) or continuous (a mean), never both.
- `None` means *not measured*, and is reported separately from a score of zero.

```mermaid
flowchart LR
  subgraph Catalog["Catalog — builtin + any .py you point at"]
    E[Eval]
    P[Prompt]
    M[Model]
    V[Provider]
  end

  E -->|samples| S[Sample]
  S --> R{{Runner}}
  P --> R
  M --> R
  V -->|AsyncOpenAI client| R
  R -->|Prediction| E
  E -->|score → Verdict| A[Aggregate]
  A --> J[result.json]
  J --> B[report → HTML / MD / CSV]
```

### Run flow

```mermaid
flowchart TD
  CLI["vlmbench eval -d … -m … -c …"] --> L[Load catalog<br/>builtin → ~/.vlmbench/catalog → -c]
  L --> X[One subject per Run<br/>provider · model · prompt]
  X --> S["Samples (--limit, --stratify)"]

  S --> MC{media in LRU?}
  MC -->|no| MD{rendered on disk?}
  MD -->|no| HF[HF download + render page]
  HF --> CORP[(~/.vlmbench/corpora)]
  MD -->|yes| CORP
  CORP --> URI[data URI]
  MC -->|yes| URI

  URI --> RC{response cached?<br/>provider·model·prompt·extra_body·sample}
  RC -->|hit| PRED[Prediction]
  RC -->|miss| API[AsyncOpenAI<br/>one client per provider]
  API --> CACHE[(~/.vlmbench/cache)]
  CACHE --> PRED

  PRED --> SC["Eval.score → (column, value)"]
  SC --> AG[Aggregate per column<br/>bool → pass rate · float → mean · None → unsupported]
  AG --> RJ[(result.json + outputs/)]
  RJ --> REP[report → HTML / MD / CSV]
```

## Suites

| `-d` | Dataset | Scale | Columns |
|---|---|---|---|
| `parsebench` | `hf://datasets/llamaindex/ParseBench` | 1,509 pages | content, noise, text, order, tables, layout |
| `olmocr-bench` | `hf://datasets/allenai/olmOCR-bench` | 1,403 pages | present, absent, order, tables, baseline |
| `omni-parse-bench` | `hf://datasets/datalab-to/omni_parse_bench` | 2,937 pages | present, order, repeat, tables |
| `omnidocbench` | `hf://datasets/opendatalab/OmniDocBench` | 1,651 pages | text, order, tables |

- Short name or full `hf://` ref both work; the full ref is always what gets recorded.
- Pin a revision with `@`: `-d parsebench@3f9ab12`. Unpinned runs record the run date (`YY-MM-DD`) as the revision.
- Not implemented yet, reported as `unsupported` rather than scored: olmOCR `math`, ParseBench charts and formatting, omni-parse `layout_kind`, OmniDocBench `equations`.
- OmniDocBench is annotation-based rather than rule-based: each block's text must appear, in reading order. Headers, footers and page numbers are excluded, since models are routinely asked to drop them.

## Progressive examples

### 1 — One model on a suite

```bash
vlmbench eval -d olmocr-bench -m zai-org/glm-ocr --limit 10 --stratify
```

- `-p` defaults to `vlmrun` (the gateway), so the provider can be left off.
- `--limit 10`, `--limit 1%` or `--limit 0.5%` for a fast signal; `--stratify` spreads it across splits.
- Any limited run is branded **PARTIAL** everywhere and is not comparable to published numbers.
- Responses are cached on disk by a fingerprint of provider, model, prompt text, `extra_body` and sample — so a re-run costs nothing and an edited prompt misses the cache.
- `--no-cache` re-requests everything; `--score-only` re-scores cached output with no network at all.

### 2 — Several models, mixed providers

```python
# runs/ocr.py
from prompts.ocr import Anchored

from vlmbench import Run

RUNS = [
    Run(model="zai-org/glm-ocr"),
    Run(model="paddlepaddle/paddleocr-vl-1.6"),
    Run(model="qwen/qwen3.8-27b", prompt=Anchored),
    Run(provider="openrouter", model="qwen/qwen3-vl-8b-instruct", prompt=Anchored),
]

SMOKE = [Run(model="zai-org/glm-ocr")]
```

```bash
vlmbench eval -d olmocr-bench -m runs/ocr.py --limit 10
vlmbench eval -d olmocr-bench -m runs/ocr.py:SMOKE --limit 10
```

- A run file declares `RUNS`; `-m file.py:NAME` picks any other list in it.
- Rows run in the order written. Nothing multiplies — a row exists because someone wrote that line.

- `-m` is one model id, or a run file. `-p` picks the provider for the whole run; a `Run` may override it.
- `vlmbench eval --list-models -p vlmrun` prints the ids that provider serves.
- `--dry-run` prints the matrix and request count before spending anything.

### 3 — Custom prompts

```python
# prompts/ocr.py
from vlmbench import Prompt

class Anchored(Prompt):
    text = "Raw text layer:\n---\n{anchor_text}\n---\nReturn clean markdown."

class Plain(Prompt):
    text = "Return the page as clean markdown, preserving reading order."
```

```bash
vlmbench eval -d olmocr-bench -m qwen/qwen3.8-27b --prompt prompts/ocr.py:Anchored --limit 10
```

- `--prompt` takes a registered name or a `path.py:Name` reference, and applies to a single `-m` model. A run file carries its own prompts instead.
- Prompts name no model, so the same object runs on any provider — the recorded `sha256` proves two rows used identical text.
- Omit the prompt to use the eval's default. Naming one for a `takes_prompt = False` model is an error.
- `{placeholders}` interpolate from `Sample.context`; escape literal braces as `{{`.

Three prompts ship built in, as a measured baseline for what wording is worth:

| Name | Tokens | Text |
| ---- | -----: | ---- |
| `markdown` | 6 | `Convert this page to markdown.` |
| `markdown-tuned` | 62 | one clause per column the suites score weakest |
| `markdown-lean` | 27 | tuned, minus the clauses that cost or did nothing |

```bash
vlmbench eval -d parsebench -m qwen/qwen3.8-27b --prompt markdown
vlmbench eval -d parsebench -m qwen/qwen3.8-27b --prompt markdown-lean
```

Token counts are measured against the model's own tokenizer by diffing
`usage.prompt_tokens`, not estimated.

**What they are worth, measured.** On `qwen/qwen3.8-27b` over all four suites
at `--limit 10 --stratify`, no prompt reliably beat the 6-token `markdown`:

| Suite | `markdown` | `markdown-tuned` | `markdown-lean` |
| ----- | ---------: | ---------------: | --------------: |
| parsebench | 29.5 | 30.4 | 26.9 |
| olmocr-bench | 25.0 | 20.8 | 25.0 |
| omni-parse-bench | 0.0 | 0.0 | 0.0 |
| omnidocbench | 19.3 | 29.5 | 18.4 |

Read the test counts before reading the spread. At `--limit 10` a column can
rest on very few tests — parsebench `tables` n=3, omnidocbench `tables` n=6 —
so several points is one test flipping. `markdown-tuned` and `markdown-lean`
carry the same table clause and scored 29.5 and 18.4 on omnidocbench, which
settles it: that gap is noise. Only parsebench `text` (n=1709), `order`
(n=259) and omnidocbench `text` (n=138) have the power to resolve anything,
and across those the three prompts sit within a few points with no consistent
ordering.

One effect did replicate, because it has a mechanism: `markdown-tuned` tells
the model to keep headers, footers and page numbers, which fights
olmOCR-bench's `headers_footers` split, whose `absent` rules check those are
*excluded*. It scored 83.3 on `absent` against 100.0 for both other prompts.

The honest reading is that prompt wording is not this model's bottleneck.
Comparing prompts seriously needs a much larger `--limit` than 10.

### Does a bigger prompt help? (30 pages, paired)

`markdown-faithful` (57 tokens) and `markdown-rich` (173 tokens) were run
against the 6-token `markdown` on the same 30 stratified pages, so page
difficulty cancels. Aggregate overall:

| Suite | `markdown` | `markdown-faithful` | `markdown-rich` |
| ----- | ---------: | ------------------: | --------------: |
| parsebench | 33.3 | 30.5 | 33.8 |
| omnidocbench | 18.6 | 19.9 | 23.3 |

That table flatters the long prompt. `overall` is an unweighted mean of
column scores, so omnidocbench `tables` (6 pages) weighs as much as `text`
(27 pages), and omnidocbench's +4.7 is almost entirely that one column.

Pairing per page and pooling both suites gives the honest estimate:

| Column | pages | `markdown-faithful` Δ | `markdown-rich` Δ |
| ------ | ----: | --------------------: | ----------------: |
| text | 36 | +2.2 [-1.0, +5.3] | -0.9 [-4.0, +2.3] |
| order | 46 | -1.1 [-3.2, +1.0] | -0.2 [-2.7, +2.3] |
| tables | 16 | -3.8 [-17.2, +9.7] | +6.8 [-2.2, +15.8] |

Every 95% interval spans zero: **a 173-token prompt is not distinguishable
from a 6-token one** on this model. The one directional hint is `tables`,
positive for `markdown-rich` on both suites independently (+13.7 and +7.1),
but on 16 pages it is unresolved.

Longer prompts do have one measured cost: parsebench `noise`, which is a
precision column, fell 68.0 -> 60.7 and 61.9. More instruction produces more
output, and more output that ground truth does not support.

For this model, decoding mattered more than any prompt: turning on
`repetition_penalty` moved parsebench 24.0 -> 30.4, larger than every prompt
effect above. Model choice mattered more still — the dedicated OCR models
score 45-90 on these same pages.

### 4 — Custom provider

```python
# catalog/fireworks.py
from vlmbench import Provider

class Fireworks(Provider):
    name = "fireworks"
    base_url = "https://api.fireworks.ai/inference/v1"
    api_key_env = "FIREWORKS_API_KEY"
    concurrency = 4
```

```bash
vlmbench eval -d olmocr-bench -c catalog/fireworks.py -p fireworks -m qwen/qwen3-vl-8b --limit 10
```

Optional hooks, each with a working OpenAI-compatible default:

| Hook | Default | Override when |
|---|---|---|
| `request_kwargs(subject, sample)` | `{}` | the provider needs routing or vendor params |
| `cost_usd(usage)` | reads `usage.cost` | cost arrives somewhere else |
| `list_models(client)` | `GET /models` | `-m all` should be filtered |

### 5 — Custom model mechanics

```python
# catalog/models.py
from vlmbench import Model

class MyOcr(Model):
    model = "acme/my-ocr-1b"
    takes_prompt = False          # image alone; the prompt is ignored
    extra_body = {"method": "markdown"}
```

- Only needed for models with quirks. A bare `-m acme/my-ocr-1b` works with defaults.
- Model ids match **normalized** (lowercase, `_ : .` folded), so provider casing differences don't matter.
- Merge order, later wins: `Provider.request_kwargs()` → `Model.extra_body` → `--extra-body`.

### 6 — Your own eval

```python
# catalog/invoices.py
import json

from vlmbench import Eval, Sample

class AcmeInvoices(Eval):
    name = "acme-invoices"
    dataset = "hf://datasets/acme/invoices-bench"
    prompt = "Extract invoice_number, date and total as JSON."

    def samples(self):
        for row in self.hf_rows():
            yield Sample(id=row["id"], media=row["image"], truth=row["gold_fields"])

    def score(self, sample, output):
        got = json.loads(output)
        for field, expected in sample.truth.items():
            yield field, str(got.get(field, "")).strip() == str(expected).strip()
```

```bash
vlmbench eval -d catalog/invoices.py -m zai-org/glm-ocr --limit 10
```

- Yielded column names become leaderboard columns — the report needs no changes.
- `Sample.media` may be a data URI, an http(s) URL, `text:...`, or `path/in/repo.pdf#2` which is downloaded, rendered and cached.

### 7 — Report

```bash
vlmbench report ~/.vlmbench/evals/*/result.json -o board.html   # baked, and imports more
vlmbench report ~/.vlmbench/evals/*/result.json --format md     # for a PR body
vlmbench report --viewer -o viewer.html                         # empty page, drop files in
```

- The HTML page renders from embedded `result.json` documents, so you can **drop more result files onto it** to add rows. Nothing leaves the browser.
- Re-dropping the same run is ignored (deduped on `run_id`); click any column header to sort; a Suite column appears once more than one suite is loaded.
- Repeat runs of the same suite + provider + model + prompt collapse into one row: the newest is shown, older ones sit behind a **+N older** toggle. Every row carries the date it was evaluated.
- `--viewer` emits the same page with no runs baked in — keep one around and feed it whatever you produce.

- Bands: **≥ 85** strong, **≥ 70** adequate, **< 70** weak — per-eval via `thresholds`.
- Cost is reported as **$/1K pages**.
- Partial and full runs will not share a table without `--mixed`.

## Loading catalogs

| Source | Loaded |
|---|---|
| builtin | always |
| `~/.vlmbench/catalog/*.py` | always |
| `$VLMBENCH_CATALOG` | always (`:`-separated) |
| `-c <file \| dir \| file.py:Class>` | per run, repeatable |

Files load by **ordinary import** — `runs/ocr.py` is imported as `runs.ocr`, with the working directory on `sys.path`. So `from prompts.ocr import Anchored` inside a run file works, tracebacks are normal, and a class imported twice is the same class.

- One file may define any mix of `Eval`, `Model`, `Prompt` and `Provider`; subclassing registers it.
- Registration makes something *available*, never *selected* — `-d`, `-m` and `-p` decide what actually runs.
- A user entry shadows a builtin of the same name, and says so in the run header.
- `vlmbench eval --list` prints everything loaded with its origin.

## Output

```
~/.vlmbench/evals/<suite>-<id>/
  result.json          totals, per-row scores, perf, cost, resolved prompt + sha256, revision
  outputs/<row>/*.md   raw model output for this run
~/.vlmbench/cache/<fingerprint>.md    shared response cache across runs
~/.vlmbench/corpora/<suite>/          rendered pages
```

- Every run prints samples, total cost, total request time, p50 latency, cache hits and errors.
- Cost comes from the provider's `usage.cost`; local serving reports `—`, never an imputed number.
- Secrets are redacted before anything is written.

## Not implemented

- `--judge` raises `NotImplementedError`.
- olmOCR `math`, ParseBench charts and formatting, omni `layout_kind` report `unsupported` instead of scoring.
