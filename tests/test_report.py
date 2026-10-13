"""Leaderboard rendering tests: bands, formats, and the partial/full guard."""

import json

import pytest

from vlmbench.report import Row, band, column_order, load_rows, to_csv, to_html, to_markdown


def make_row(**kwargs) -> Row:
    base = dict(
        provider="vlmrun",
        model="zai-org/glm-ocr",
        prompt="—",
        overall=86.9,
        columns={
            "content": {"score": 91.4, "tests": 0, "passed": 0, "samples": 5, "unsupported": 0},
            "charts": {"score": None, "tests": 0, "passed": 0, "samples": 0, "unsupported": 44},
        },
        usd_per_1k=2.10,
        eval_name="parsebench",
    )
    return Row(**{**base, **kwargs})


@pytest.mark.parametrize(
    ("score", "expected"),
    [(100.0, "good"), (85.0, "good"), (84.9, "warning"), (70.0, "warning"), (69.9, "critical"), (None, "none")],
)
def test_bands_use_the_eval_thresholds(score, expected):
    assert band(score, 85.0, 70.0) == expected


def test_bands_follow_a_custom_threshold():
    assert band(88.0, good=95.0, fair=85.0) == "warning"


def test_known_columns_lead_then_catalog_columns_follow():
    row = make_row(columns={"zz_custom": {"score": 1.0}, "tables": {"score": 2.0}, "content": {"score": 3.0}})
    assert column_order([row]) == ["content", "tables", "zz_custom"]


def test_rows_sort_by_overall_with_unscored_last():
    rows = [make_row(overall=50.0), make_row(overall=None), make_row(overall=90.0)]
    payload = {
        "kind": "eval",
        "eval": {"name": "x", "thresholds": {"good": 85, "fair": 70}},
        "limit": {"pages": 5, "of": 5, "partial": False},
        "rows": [
            {"provider": "p", "model": "m", "scores": {"overall": r.overall, "columns": {}}, "cost": {}} for r in rows
        ],
    }
    assert [r.overall for r in load_rows_from(payload)] == [90.0, 50.0, None]


def load_rows_from(payload, tmp=None):
    import tempfile
    from pathlib import Path

    path = Path(tempfile.mkdtemp()) / "result.json"
    path.write_text(json.dumps(payload))
    return load_rows([path])


def test_markdown_and_csv_round_trip_the_columns():
    rows = [make_row()]
    markdown = to_markdown(rows)
    assert "| Content | Charts | $/1K pages |" in markdown
    assert "**86.9**" in markdown
    csv_text = to_csv(rows)
    assert csv_text.splitlines()[0] == "Provider,Model,Prompt,Overall,Content,Charts,Cost_Usd_Per_1k_Pages"
    assert "86.90" in csv_text


def test_partial_runs_are_branded_in_every_format():
    rows = [make_row(partial=True, pages=5, of_pages=1509)]
    assert "PARTIAL" in to_markdown(rows)
    assert "Partial run" in to_html(rows)


def test_html_is_self_contained_and_carries_both_themes():
    page = to_html([make_row()])
    assert "<script src=" not in page and "cdn" not in page.lower()
    assert "prefers-color-scheme: dark" in page
    assert 'data-theme="dark"' in page
    assert "#0ca30c" in page and "#d03b3b" in page


def test_unscored_column_renders_as_a_dash_not_zero():
    page = to_html([make_row()])
    assert ">—<" in page
    assert "width:0.0%" not in page


def test_missing_cost_never_renders_as_zero():
    page = to_html([make_row(usd_per_1k=None)])
    assert "provider reported no cost" in page


def test_single_provider_and_promptless_columns_are_dropped():
    page = to_html([make_row()])
    assert "<th>Provider</th>" not in page
    assert "<th>Prompt</th>" not in page
    assert "| Model | Overall |" in to_markdown([make_row()]).splitlines()[0]


def test_columns_return_when_they_distinguish_rows():
    rows = [make_row(), make_row(provider="openrouter", prompt="Anchored")]
    page = to_html(rows)
    assert "<th>Provider</th>" in page
    assert "<th>Prompt</th>" in page


def test_csv_keeps_every_column_for_machine_readers():
    header = to_csv([make_row()]).splitlines()[0]
    assert header.startswith("Provider,Model,Prompt,Overall")
