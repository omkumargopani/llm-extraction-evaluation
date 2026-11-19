import csv
from collections import Counter
from pathlib import Path

FIELDS = ("customer_name", "product", "amount")

def load_rows(path):
    with open(path, newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))

def average(rows, field):
    return sum((1.0 if row[field] == "True" else 0.0) if row[field] in ("True", "False") else float(row[field]) for row in rows) / len(rows)

def summarize(rows):
    if not rows:
        raise ValueError("No rows to summarize")
    failures = Counter()
    for row in rows:
        failures.update(category for category in row["errors"].split("|") if category)
    return {
        "documents": len(rows),
        "field_accuracy": average(rows, "accuracy"),
        "exact_match": average(rows, "exact_document_match"),
        "schema_validity": average(rows, "schema_valid"),
        "literal_grounding": average(rows, "grounding"),
        "mean_latency_seconds": average(rows, "latency_seconds"),
        **{f"{field}_accuracy": average(rows, f"correct_{field}") for field in FIELDS},
    }, failures

def main():
    paths = [Path("reports/zero_shot.csv"), Path("reports/few_shot.csv")]
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing experiment reports: " + ", ".join(missing))
    summaries = []
    error_rows = []
    ids = []
    for path in paths:
        rows = load_rows(path)
        ids.append({row["id"] for row in rows})
        summary, errors = summarize(rows)
        summaries.append({"approach": path.stem, **summary})
        error_rows.extend({"approach": path.stem, "category": name, "count": count} for name, count in sorted(errors.items()))
    if ids[0] != ids[1]:
        raise ValueError("Experiments must evaluate the same document IDs")
    output = Path("reports")
    with (output / "comparison.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)
    with (output / "error_summary.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["approach", "category", "count"])
        writer.writeheader()
        writer.writerows(error_rows)
    for summary in summaries:
        print(f"{summary['approach']}: field_accuracy={summary['field_accuracy']:.3f}, exact_match={summary['exact_match']:.3f}, validity={summary['schema_validity']:.3f}, grounding={summary['literal_grounding']:.3f}, latency={summary['mean_latency_seconds']:.3f}s")
    print("saved: reports/comparison.csv")
    print("saved: reports/error_summary.csv")

if __name__ == "__main__":
    main()
