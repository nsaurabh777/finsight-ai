"""Pretty-print the per-query breakdown of the latest (or a named) eval run.

    python -m scripts.inspect_eval                 # newest run in eval/results/
    python -m scripts.inspect_eval 20260901_163601 # a specific run
"""
import json
import sys
from pathlib import Path

RESULTS_DIR = Path(__file__).resolve().parent.parent / "eval" / "results"


def main() -> None:
    if len(sys.argv) > 1:
        path = RESULTS_DIR / f"{sys.argv[1].removesuffix('.json')}.json"
    else:
        runs = sorted(RESULTS_DIR.glob("*.json"))
        if not runs:
            sys.exit(f"no eval runs in {RESULTS_DIR}")
        path = runs[-1]

    data = json.loads(path.read_text(encoding="utf-8"))
    print(f"# {path.name}\n")

    for r in data["results"]:
        s = r["scores"]
        print(
            f"[{r['id']}] {r.get('category')}\n"
            f"  ticker={s.get('ticker_accuracy_pass')}  "
            f"graceful={s.get('graceful_failure_pass')}  "
            f"faith={s.get('faithfulness')}  rel={s.get('relevance')}  "
            f"disclaimer={s.get('disclaimer_present_pass')}"
        )
        notes = s.get("judge_notes")
        if notes:
            print(f"  notes: {notes}")
        if str(r.get("report", "")).startswith("ERROR:"):
            print(f"  {r['report'][:200]}")
        print()

    print("=== AGGREGATE ===")
    for k, v in data["aggregate"].items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
