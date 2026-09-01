"""
Runs every query in test_queries.json through the full pipeline, then scores
each report against rubric.md using an LLM judge, and saves timestamped
results to eval/results/ (gitignored — commit summary numbers to the README
manually once a baseline is established, not the raw run files).

This is the before/after evidence generator: run it, change a prompt or the
retrieval, run it again, compare aggregates. That delta ("faithfulness 3.1 ->
4.2 after grounding the News Analyst") is the resume-defensible metric.

Run:
    python -m eval.run_eval
    python -m eval.run_eval --limit 3        # smoke test on first 3 queries
"""
import argparse
import json
import re
import time
from datetime import datetime
from pathlib import Path

from src.config import get_judge_llm
from src.crew import run_research

EVAL_DIR = Path(__file__).resolve().parent
RESULTS_DIR = EVAL_DIR / "results"
RUBRIC_PATH = EVAL_DIR / "rubric.md"

# Categories from test_queries.json where "the company isn't in the store" is
# the correct behaviour, so the Graceful Failure criterion applies.
NOT_IN_STORE_CATEGORIES = {"not_in_store"}

JUDGE_SYSTEM = (
    "You are a strict evaluation judge for financial research reports. You "
    "score only what is in the report text against the rubric. You never "
    "reward fluent writing that isn't grounded. Respond with JSON only."
)

JUDGE_TEMPLATE = """\
Score the REPORT below against this rubric.

RUBRIC:
{rubric}

USER QUERY:
{query}

EXPECTED TICKER BEHAVIOUR: {ticker_expectation}

REPORT:
\"\"\"
{report}
\"\"\"

Return ONLY this JSON object, no prose:
{{
  "faithfulness": <int 1-5>,
  "relevance": <int 1-5>,
  "ticker_accuracy_pass": <true|false>,
  "graceful_failure_pass": <true|false|null>,
  "disclaimer_present_pass": <true|false>,
  "judge_notes": "<one or two sentences>"
}}
graceful_failure_pass must be null unless the expected behaviour is that the
company is NOT in the coverage universe.

Hard rules — apply before scoring:
- If the report analyses a DIFFERENT company than the user asked about, set
  faithfulness=1 and relevance=1.
- faithfulness is about grounding, not fluency. A well-written report that
  asserts figures or qualitative claims with no visible tool output / cited
  passage is at most 3. Reserve 5 for reports where every claim is traceable.
- If the report says a required data point was unavailable, that does NOT
  lower faithfulness — acknowledging a gap is faithful behaviour.
"""


def load_test_queries() -> list[dict]:
    with open(EVAL_DIR / "test_queries.json", encoding="utf-8") as f:
        return json.load(f)


def _extract_json(text: str) -> dict:
    """Judge models sometimes wrap JSON in prose or ```json fences. Grab the
    first {...} block and parse that."""
    fenced = re.search(r"\{.*\}", text, re.DOTALL)
    if not fenced:
        raise ValueError(f"No JSON object in judge response: {text[:200]!r}")
    return json.loads(fenced.group(0))


# Language the crew is prompted to emit when a company isn't in the ticker
# store (see src/crew.py resolve_task / src/tools/ticker_resolver.py).
_GRACEFUL_MARKERS = ("coverage universe", "ticker_not_found", "not in the coverage",
                     "outside finsight", "not covered")


def graceful_failure_ok(report: str) -> bool:
    """Deterministic check that the pipeline refused rather than fabricated an
    analysis for an out-of-universe company. Not left to the judge — the
    baseline showed the judge returns null / faithfulness=5 even when the
    report analyses an entirely different company (q15 -> Hero MotoCorp)."""
    low = report.lower()
    return any(m in low for m in _GRACEFUL_MARKERS)


def judge_report(llm, rubric: str, query: dict, report: str) -> dict:
    not_in_store = query.get("category") in NOT_IN_STORE_CATEGORIES
    expectation = (
        "The company is NOT in the coverage universe — the system should say so "
        "and not fabricate a ticker or analysis."
        if not_in_store
        else "The company IS in the coverage universe — the correct '.NS' ticker "
        "should be resolved and analysed."
    )
    prompt = JUDGE_TEMPLATE.format(
        rubric=rubric,
        query=query["query"],
        ticker_expectation=expectation,
        report=report,
    )
    raw = llm.call([
        {"role": "system", "content": JUDGE_SYSTEM},
        {"role": "user", "content": prompt},
    ])
    scores = _extract_json(raw if isinstance(raw, str) else str(raw))
    if not_in_store:
        # Override the judge with the deterministic check for both the
        # safety criteria — the judge is unreliable here.
        ok = graceful_failure_ok(report)
        scores["graceful_failure_pass"] = ok
        scores["ticker_accuracy_pass"] = ok
    else:
        scores["graceful_failure_pass"] = None
    return scores


def _run_research_with_retry(query: str, attempts: int = 3) -> str:
    """Groq's free tier intermittently returns an empty completion ("Invalid
    response from LLM call - None or empty"), which crashed q14 in the baseline.
    That's transient — retry with backoff before giving up on the query."""
    for i in range(attempts):
        try:
            return run_research(query)
        except Exception as exc:
            if i == attempts - 1 or "None or empty" not in str(exc):
                raise
            print(f"  .. transient LLM error, retry {i + 1}/{attempts - 1}: {exc}")
            time.sleep(15 * (i + 1))


def _aggregate(results: list[dict]) -> dict:
    def mean(vals):
        vals = [v for v in vals if isinstance(v, (int, float))]
        return round(sum(vals) / len(vals), 2) if vals else None

    def pass_rate(key):
        vals = [
            r["scores"].get(key)
            for r in results
            if r["scores"].get(key) is not None
        ]
        return round(100 * sum(bool(v) for v in vals) / len(vals), 1) if vals else None

    return {
        "n_queries": len(results),
        "mean_faithfulness": mean(r["scores"].get("faithfulness") for r in results),
        "mean_relevance": mean(r["scores"].get("relevance") for r in results),
        "ticker_accuracy_pass_pct": pass_rate("ticker_accuracy_pass"),
        "graceful_failure_pass_pct": pass_rate("graceful_failure_pass"),
        "disclaimer_present_pass_pct": pass_rate("disclaimer_present_pass"),
    }


def run(limit: int | None = None):
    RESULTS_DIR.mkdir(exist_ok=True)
    rubric = RUBRIC_PATH.read_text(encoding="utf-8")
    llm = get_judge_llm()
    queries = load_test_queries()
    if limit:
        queries = queries[:limit]

    results = []
    for q in queries:
        print(f"[{q['id']}] {q['query']}")
        try:
            report = _run_research_with_retry(q["query"])
            scores = judge_report(llm, rubric, q, report)
        except Exception as exc:  # keep the run going; log the failure
            print(f"  !! {type(exc).__name__}: {exc}")
            report, scores = f"ERROR: {exc}", {
                "faithfulness": None, "relevance": None,
                "ticker_accuracy_pass": None, "graceful_failure_pass": None,
                "disclaimer_present_pass": None, "judge_notes": f"pipeline error: {exc}",
            }
        results.append({**q, "report": report, "scores": scores})
        print(f"  -> {scores}")

    agg = _aggregate(results)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = RESULTS_DIR / f"{stamp}.json"
    out_path.write_text(
        json.dumps({"aggregate": agg, "results": results}, indent=2), encoding="utf-8"
    )

    print("\n=== AGGREGATE ===")
    for k, v in agg.items():
        print(f"  {k}: {v}")
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    run(limit=args.limit)
