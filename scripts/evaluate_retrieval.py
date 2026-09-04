import argparse
import json
import sys
from pathlib import Path
from time import perf_counter
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.evaluation import evaluate_retrieved_items, summarize_metrics
from app.services import retrieve


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate EvidenceRAG retrieval quality.")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("eval/retrieval_eval.example.jsonl"),
        help="JSONL file with question and relevant target labels.",
    )
    parser.add_argument("--k", type=int, default=5, help="Number of retrieved chunks to evaluate.")
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON output.")
    args = parser.parse_args()

    cases = load_cases(args.dataset)
    results = [evaluate_case(case, args.k) for case in cases]
    metrics = [result["metrics"] for result in results]
    summary = summarize_metrics(metrics)
    summary["mean_latency_ms"] = (
        sum(result["latency_ms"] for result in results) / len(results) if results else 0.0
    )

    report = {"dataset": str(args.dataset), "summary": summary, "cases": results}
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print_report(report)
    return 0


def load_cases(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Evaluation dataset was not found: {path}")

    cases: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        case = json.loads(stripped)
        if not case.get("question"):
            raise ValueError(f"Line {line_number} is missing a question")
        if not case.get("relevant"):
            raise ValueError(f"Line {line_number} is missing relevant labels")
        cases.append(case)
    return cases


def evaluate_case(case: dict[str, Any], k: int) -> dict[str, Any]:
    started = perf_counter()
    evidence = retrieve(case["question"], k)
    latency_ms = (perf_counter() - started) * 1000
    metrics = evaluate_retrieved_items(evidence, case["relevant"], k)
    return {
        "question": case["question"],
        "metrics": metrics,
        "latency_ms": round(latency_ms, 2),
        "retrieved": [
            {
                "chunk_id": row["id"],
                "document_id": row["document_id"],
                "document": row["document"],
                "page": row["page"],
                "score": row["score"],
            }
            for row in evidence
        ],
    }


def print_report(report: dict[str, Any]) -> None:
    print(f"Dataset: {report['dataset']}")
    print()
    for index, result in enumerate(report["cases"], start=1):
        metrics = result["metrics"]
        print(f"{index}. {result['question']}")
        print(
            "   "
            f"P@{metrics['k']}: {metrics['precision_at_k']:.3f} | "
            f"R@{metrics['k']}: {metrics['recall_at_k']:.3f} | "
            f"MRR: {metrics['mrr']:.3f} | "
            f"nDCG@{metrics['k']}: {metrics['ndcg_at_k']:.3f} | "
            f"latency: {result['latency_ms']:.2f} ms"
        )
    summary = report["summary"]
    print()
    print("Summary")
    print(f"Cases: {summary['cases']}")
    print(f"Mean Precision@K: {summary['mean_precision_at_k']:.3f}")
    print(f"Mean Recall@K: {summary['mean_recall_at_k']:.3f}")
    print(f"Mean MRR: {summary['mean_mrr']:.3f}")
    print(f"Mean nDCG@K: {summary['mean_ndcg_at_k']:.3f}")
    print(f"Mean latency: {summary['mean_latency_ms']:.2f} ms")


if __name__ == "__main__":
    raise SystemExit(main())
