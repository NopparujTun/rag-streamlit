"""
RAG Evaluation Pipeline (Faithfulness & Accuracy).

Runs every question in the evaluation dataset through the full RAG pipeline
(hybrid search → LLM generation → faithfulness & accuracy check) and reports 
aggregated metrics broken down by difficulty level.

Usage:
    python scripts/run_eval.py                              # default paths
    python scripts/run_eval.py --dataset custom.json        # custom dataset
    python scripts/run_eval.py --output results/out.json    # custom output
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

# Ensure the project root is on sys.path so ``src.*`` imports resolve
# regardless of working directory.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv

load_dotenv(PROJECT_ROOT / ".env")

import yaml

from src.rag.embeddings import get_embedding_model
from src.rag.evaluator import evaluate_faithfulness, evaluate_accuracy
from src.rag.generator import generate_answer
from src.rag.vectorstore import load_hybrid_store, perform_hybrid_search, get_reranker

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULT_DATASET = PROJECT_ROOT / "eval_dataset.json"
DEFAULT_OUTPUT = PROJECT_ROOT / "eval_results.json"
DEFAULT_CONFIG = PROJECT_ROOT / "config.yaml"
SEARCH_TOP_K = 3
DIFFICULTY_LEVELS = ("easy", "medium", "hard")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_config(path: Path) -> dict[str, Any]:
    """Load YAML configuration from *path*."""
    with open(path, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def load_dataset(path: Path) -> list[dict[str, Any]]:
    """Load a JSON evaluation dataset from *path*."""
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def save_results(results: list[dict[str, Any]], path: Path) -> None:
    """Persist evaluation results to a JSON file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(results, fh, ensure_ascii=False, indent=2)


def print_summary(results: list[dict[str, Any]]) -> None:
    """Print a formatted summary table of evaluation results."""
    total = len(results)
    if total == 0:
        logger.warning("No results to summarise.")
        return

    print()
    for level in DIFFICULTY_LEVELS:
        subset = [r for r in results if r["difficulty"] == level]
        if not subset:
            continue
        faith = sum(r["faithfulness"] for r in subset) / len(subset)
        acc = sum(r["accuracy"] for r in subset) / len(subset)
        avg_lat = sum(r["latency"] for r in subset) / len(subset)
        print(
            f"[{level.upper():6}] "
            f"Faithfulness: {faith:.1%} | "
            f"Accuracy: {acc:.1%} | "
            f"Avg latency: {avg_lat:.2f}s | "
            f"n={len(subset)}"
        )

    avg_latency = sum(r["latency"] for r in results) / total
    p95_index = min(int(total * 0.95), total - 1)
    p95 = sorted(r["latency"] for r in results)[p95_index]
    overall_faith = sum(r["faithfulness"] for r in results) / total
    overall_acc = sum(r["accuracy"] for r in results) / total

    print(f"\n{'=' * 60}")
    print(f"Overall Faithfulness : {overall_faith:.1%}")
    print(f"Overall Accuracy     : {overall_acc:.1%}")
    print(f"Avg Latency          : {avg_latency:.2f}s")
    print(f"P95 Latency          : {p95:.2f}s")
    print(f"Total questions      : {total}")


# ---------------------------------------------------------------------------
# Core pipeline
# ---------------------------------------------------------------------------

def evaluate_single_question(
    item: dict[str, Any],
    vectorstore,
    bm25_retriever,
    reranker_model,
) -> dict[str, Any]:
    """Run retrieval → generation → faithfulness & accuracy check for one question."""
    start = time.time()

    answer, retrieved, _ = generate_answer(
        query=item["question"],
        vectorstore=vectorstore,
        bm25_retriever=bm25_retriever,
        chat_history="",
        reranker=reranker_model,
    )
    latency = time.time() - start

    # Evaluate Faithfulness
    faith_verdict = evaluate_faithfulness(
        context_docs=retrieved,
        generated_answer=answer,
    )
    faith_passed = faith_verdict == "PASS"

    # Evaluate Accuracy 
    ground_truth = item.get("ground_truth", "")
    acc_verdict = evaluate_accuracy(
        question=item["question"],
        ground_truth=ground_truth,
        answer=answer,
    )
    acc_passed = acc_verdict == "PASS"

    return {
        "id": item["id"],
        "difficulty": item["difficulty"],
        "section": item.get("relevant_section", "N/A"),
        "latency": round(latency, 2),
        "faithfulness": faith_passed,
        "accuracy": acc_passed,
        "faith_verdict": faith_verdict,
        "acc_verdict": acc_verdict,
        "answer": answer,
    }


def run_evaluation(
    config_path: Path,
    dataset_path: Path,
    output_path: Path,
) -> list[dict[str, Any]]:
    """Execute the full RAG evaluation pipeline and return results."""
    config = load_config(config_path)

    logger.info("Loading embedding model: %s", config["embedding"]["model_name"])
    embedding_model = get_embedding_model(
        model_name=config["embedding"]["model_name"],
        device=config["embedding"]["device"],
    )

    logger.info("Connecting to vector store: %s", config["vector_db"]["index_name"])
    vectorstore, bm25_retriever = load_hybrid_store(
        embedding_model=embedding_model,
        persist_dir=config["vector_db"]["persist_directory"],
        index_name=config["vector_db"]["index_name"],
    )

    logger.info("Initializing Cross-Encoder Re-ranker...")
    reranker_model = get_reranker()

    dataset = load_dataset(dataset_path)
    logger.info("Loaded %d questions from %s", len(dataset), dataset_path.name)

    results: list[dict[str, Any]] = []
    for item in dataset:
        logger.info("Running %s...", item["id"])

        result = evaluate_single_question(item, vectorstore, bm25_retriever, reranker_model)
        
        faith_status = "PASS" if result["faithfulness"] else "FAIL"
        acc_status = "PASS" if result["accuracy"] else "FAIL"
        
        logger.info(
            "  %s → F:%s | A:%s (%.1fs)", 
            item["id"], faith_status, acc_status, result["latency"]
        )
        results.append(result)

    save_results(results, output_path)
    logger.info("Saved results to %s", output_path)
    print_summary(results)

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run RAG evaluation pipeline (Faithfulness & Accuracy).",
    )
    parser.add_argument(
        "--config", type=Path, default=DEFAULT_CONFIG,
        help="Path to config.yaml (default: %(default)s)",
    )
    parser.add_argument(
        "--dataset", type=Path, default=DEFAULT_DATASET,
        help="Path to eval_dataset.json (default: %(default)s)",
    )
    parser.add_argument(
        "--output", type=Path, default=DEFAULT_OUTPUT,
        help="Output path for results JSON (default: %(default)s)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_evaluation(
        config_path=args.config,
        dataset_path=args.dataset,
        output_path=args.output,
    )