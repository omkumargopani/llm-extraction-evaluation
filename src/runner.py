import argparse
import csv
import json
import os
import time
from pathlib import Path
from dotenv import load_dotenv
from src.extractor import extract_order
from src.metrics import FIELDS, field_accuracy, field_results, schema_valid, grounding_score, error_categories

load_dotenv()

def run(dataset_path: str, approach: str, trace: bool = True) -> list[dict]:
    records = json.loads(Path(dataset_path).read_text(encoding="utf-8"))
    if not records:
        raise ValueError("Evaluation dataset is empty")
    if len({doc["id"] for doc in records}) != len(records):
        raise ValueError("Dataset IDs must be unique")
    langfuse = None
    if trace:
        from langfuse import get_client
        langfuse = get_client()
    results = []
    for doc in records:
        start = time.perf_counter()
        prediction = {}
        error = ""
        try:
            if langfuse is None:
                prediction = extract_order(doc["text"], approach)
            else:
                with langfuse.start_as_current_observation(name="evaluate-document", as_type="span", input={"id": doc["id"], "text": doc["text"], "approach": approach}) as span:
                    with langfuse.start_as_current_observation(name="extract-order", as_type="generation", model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"), input=doc["text"]) as generation:
                        prediction = extract_order(doc["text"], approach)
                        generation.update(output=prediction)
                    score = field_accuracy(doc["expected"], prediction)
                    span.score_trace(name="field_accuracy", value=score, data_type="NUMERIC")
                    span.update(output=prediction)
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
        latency = round(time.perf_counter() - start, 4)
        valid = schema_valid(prediction)
        accuracy = field_accuracy(doc["expected"], prediction) if valid and not error else 0.0
        grounding = grounding_score(doc["text"], prediction) if valid else 0.0
        categories = error_categories(doc["expected"], prediction, valid, grounding)
        if error:
            categories.append("request_or_parse_failure")
        matches = field_results(doc["expected"], prediction)
        results.append({
            "id": doc["id"], "approach": approach, "accuracy": accuracy,
            "schema_valid": valid, "grounding": grounding, "latency_seconds": latency,
            "exact_document_match": all(matches.values()) and valid and not error,
            **{f"correct_{field}": matches[field] and valid and not error for field in FIELDS},
            "errors": "|".join(categories), "exception": error,
            "expected": json.dumps(doc["expected"], ensure_ascii=False),
            "prediction": json.dumps(prediction, ensure_ascii=False)
        })
    if langfuse is not None:
        langfuse.flush()
    return results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="datasets/eval.json")
    parser.add_argument("--approach", choices=["zero_shot", "few_shot"], default="zero_shot")
    parser.add_argument("--no-trace", action="store_true")
    args = parser.parse_args()
    results = run(args.dataset, args.approach, trace=not args.no_trace)
    output = Path("reports") / f"{args.approach}.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    for metric in ("accuracy", "grounding", "latency_seconds"):
        value = sum(row[metric] for row in results) / len(results)
        print(f"{metric}: {value:.3f}")
    print(f"schema_valid_rate: {sum(row['schema_valid'] for row in results) / len(results):.3f}")
    print(f"document_exact_match_rate: {sum(row['exact_document_match'] for row in results) / len(results):.3f}")
    print(f"saved: {output}")

if __name__ == "__main__":
    main()
