# LLM Extraction Evaluation Framework

**Python · OpenAI API · Langfuse · Pydantic · pytest**

A small, reproducible evaluation framework for benchmarking LLM-based information extraction from unstructured order descriptions. It compares zero-shot and few-shot prompts against labeled examples, validates structured predictions, checks literal evidence in source text, and categorizes extraction errors.

## Project scope

The task is to extract three fields from a description of a **completed purchase**:

- `customer_name`: the buyer, not the seller
- `product`: the purchased item
- `amount`: the final amount paid

If no completed purchase is confirmed (for example, a cancelled, failed, pending, or merely proposed transaction), the expected result is `null` for all three fields. For completed purchases, missing individual fields are `null`.

Example input:

> Rahul bought an iPhone 15 for Rs. 65000.

Expected output:

```json
{
  "customer_name": "Rahul",
  "product": "iPhone 15",
  "amount": 65000
}
```

## How it works

```text
40 labeled examples (datasets/eval.json)
          |
          v
Python experiment runner (src/runner.py)
          |
          +--> Zero-shot prompt
          +--> Few-shot prompt
          |
          v
OpenAI structured extraction (src/extractor.py, src/schema.py)
          |
          v
Metrics and error categories (src/metrics.py)
          |
          +--> Langfuse observations and scores (when tracing is enabled)
          +--> Per-document CSV reports
          |
          v
Comparison and error summary (src/report.py)
```

## Repository layout

```text
llm-extraction-evaluation/
├── datasets/
│   └── eval.json
├── src/
│   ├── __init__.py
│   ├── schema.py
│   ├── extractor.py
│   ├── metrics.py
│   ├── runner.py
│   └── report.py
├── tests/
│   └── test_metrics.py
├── reports/
├── .env.example
├── .gitignore
├── requirements.txt
└── README.md
```

## Setup

Requires Python 3.11 or newer. On Ubuntu/Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
cp .env.example .env
```

Edit `.env` with your private credentials:

```dotenv
OPENAI_API_KEY=replace_me
OPENAI_MODEL=gpt-4o-mini
LANGFUSE_PUBLIC_KEY=replace_me
LANGFUSE_SECRET_KEY=replace_me
LANGFUSE_BASE_URL=https://cloud.langfuse.com
```

Use the Langfuse base URL corresponding to your project region. OpenAI API requests may incur charges. **Never commit `.env` or share credentials.**

## Run automated tests

```bash
python -m pytest -v
```

**Observed result:** 12 tests passed (November 19, 2026). These tests cover local metric, validation, error categorization, and reporting behavior; they do not independently validate live API availability.

## Run evaluations

```bash
python -m src.runner --approach zero_shot
python -m src.runner --approach few_shot
python -m src.report
```

Omit Langfuse tracing for a local report-only run by adding `--no-trace` to either runner command.

Generated outputs:

- `reports/zero_shot.csv`: predictions and document-level metrics
- `reports/few_shot.csv`: predictions and document-level metrics
- `reports/comparison.csv`: aggregate and per-field metrics
- `reports/error_summary.csv`: error category counts

These generated CSVs are ignored by Git. The experiment results summarized below are recorded in the README so the repository can be understood without uploading traces or raw reports.

## Evaluation metrics

| Metric | Definition |
|---|---|
| Field accuracy | Proportion of the three labeled fields that match the prediction after case/whitespace normalization |
| Document exact match | Proportion of documents where all three fields match |
| Per-field accuracy | Accuracy reported separately for customer name, product, and amount |
| Schema validity | Fraction of predictions passing strict Pydantic validation |
| Literal grounding | Fraction of non-null predicted values found using a simple literal/numeric evidence check in the source |
| Mean latency | Mean wall-clock extraction/evaluation time per document |
| Error categories | Missing, unexpected (`unsupported_*`), or incorrect field values; invalid schema; possible ungrounded values; request/parse failures |

Literal grounding is **not** semantic entailment: a value can appear in the source and still be incorrect for the task, as when a cancelled transaction mentions a buyer, product, and price. The all-null prediction has literal grounding score 1 by convention, so it must be interpreted together with correctness metrics.

Structured outputs are requested using Pydantic, so 100% schema validity on successful runs does not demonstrate that unconstrained prompting would produce well-formed JSON.

## Experiments and observed results

Both prompts were evaluated on the same **40-example synthetic, labeled development set**. During error analysis, two zero-shot documents and one few-shot document initially failed exact matching. In particular, documents `order_019` and `order_024` exposed false-positive extraction for transactions that should have had null fields. The common instructions were then strengthened to explicitly require evidence of a completed purchase.

| Metric | Zero-shot baseline | Zero-shot revised | Few-shot baseline | Few-shot revised |
|---|---:|---:|---:|---:|
| Field accuracy | 95.83% | 100.00% | 97.50% | 100.00% |
| Document exact match | 95.00% | 100.00% | 97.50% | 100.00% |
| Schema validity | 100.00% | 100.00% | 100.00% | 100.00% |
| Literal grounding | 100.00% | 100.00% | 100.00% | 100.00% |
| Mean latency | 3.271 s | 4.137 s | 2.942 s | 4.974 s |

Before prompt refinement, zero-shot field accuracy by field was 95.0% (`customer_name`), 95.0% (`product`), and 97.5% (`amount`); few-shot accuracy was 97.5% on each field. The baseline error report recorded five unexpected fields for zero-shot and three for few-shot. Those counts refer to incorrect fields rather than independently proven hallucinations.

**Interpretation:** The revised instructions improved results on the *same* development dataset used to diagnose the failures. Therefore the 100% revised scores should not be represented as held-out generalization performance. Latency also varied, and these single-run timings cannot establish a consistent latency advantage.

## Langfuse observability

When tracing is enabled, the runner records a parent `evaluate-document` observation and a nested `extract-order` generation observation for each document. The project has been verified to send observations to Langfuse. Trace metadata enables inspection of extraction input/output and timing; a field-accuracy score is associated with the trace. No external tool-calling workflow is implemented in this project.

Do not send confidential documents to Langfuse without appropriate permissions and handling rules.

## Limitations

- The 40 documents are synthetic and were used for development and prompt refinement, not as a blind held-out test.
- The literal grounding check is a lightweight heuristic; it does not establish semantic source support.
- Prompt variants use schema-constrained structured output, which makes schema-validity comparisons less informative.
- Latency is based on individual experiment runs and can fluctuate with network and service conditions.
- This is an evaluation prototype, not a production-grade document-processing system.

## Technologies

- **Python:** evaluation runner and reporting
- **OpenAI API:** structured LLM extraction
- **Pydantic:** schema definition and validation
- **Langfuse:** tracing and score inspection
- **pytest:** automated metric and regression tests

## Future work (outside this project scope)

An independently labeled held-out benchmark, more robust evidence-span grounding, and repeated timing experiments would enable stronger performance claims.
