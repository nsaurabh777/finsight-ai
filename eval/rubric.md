# FinSight AI — Eval Rubric

Used by `run_eval.py`'s LLM-as-judge to score each generated report. Mirrors
the LLM-as-judge pattern used to take the intent-classification Jaccard Index
from 0.87 to 0.93 professionally — applied here to a personal project so the
resume claim is backed by an actual harness, not just the phrase "RAG."

## Criteria (score each 1–5 unless noted as pass/fail)

### 1. Faithfulness (1–5)
Does every factual/qualitative claim in the report trace to either a tool
output (numbers) or a retrieved passage (qualitative claims), rather than
being asserted from the model's general knowledge?
- 5: every claim is traceable
- 3: mostly traceable, one or two unsupported minor claims
- 1: report makes claims with no basis in retrieved/tool content

### 2. Relevance (1–5)
Does the report actually answer what the user asked, at the right depth
for their apparent expertise level (see queries q12/q13 — persona-aware)?
- 5: directly and completely answers the query
- 3: answers the general topic but misses the specific angle asked
- 1: generic report, ignores the specific question

### 3. Ticker Accuracy (pass/fail)
Did the Stock Researcher resolve the *correct* ticker for the company named?
Fail if wrong ticker OR if it silently proceeded on a low-confidence guess
instead of flagging "not found" (see queries q14/q15).

### 4. Graceful Failure (pass/fail, only scored for q14/q15-style queries)
When the company isn't in the ticker store, does the system say so clearly
instead of hallucinating a plausible-looking ticker and running analysis on
a wrong or nonexistent security?

### 5. Disclaimer Present (pass/fail)
Is the exact non-financial-advice disclaimer present in the output?

## Scoring output

`run_eval.py` should produce, per run: per-query scores across all criteria,
an aggregate (mean Faithfulness, mean Relevance, % pass on Ticker Accuracy /
Graceful Failure / Disclaimer), saved with a timestamp so successive runs
can be compared — this before/after comparison is what turns "I built RAG
grounding" into "grounding raised faithfulness from X to Y," a real
resume-defensible metric.
