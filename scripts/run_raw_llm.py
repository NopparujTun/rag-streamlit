"""
Raw LLM Baseline Evaluation (No-RAG).

Runs every question in the evaluation dataset through the OpenTyphoon LLM
WITHOUT any document retrieval. This establishes a baseline to prove
the value of the RAG pipeline. Measures only Accuracy.

Usage:
    python scripts/run_raw_llm.py
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any

# Ensure the project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")

from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from src.rag.evaluator import evaluate_accuracy

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEFAULT_DATASET = PROJECT_ROOT / "eval_dataset.json"
DEFAULT_OUTPUT = PROJECT_ROOT / "raw_llm_results.json"
DIFFICULTY_LEVELS = ("easy", "medium", "hard")

_TYPHOON_MODEL = "typhoon-v2.5-30b-a3b-instruct"
_TYPHOON_BASE_URL = "https://api.opentyphoon.ai/v1"

# ---------------------------------------------------------------------------
# LLM Setup
# ---------------------------------------------------------------------------

def get_raw_llm_chain():
    """Create a simple prompt-to-LLM chain without RAG context."""
    llm = ChatOpenAI(
        model=_TYPHOON_MODEL,
        api_key=os.environ.get("TYPHOON_API_KEY"),
        base_url=_TYPHOON_BASE_URL,
        temperature=0.1, # ใช้ Temp ต่ำเพื่อให้คำตอบนิ่งที่สุด
        max_tokens=1000,
    )
    
    # Prompt โล่งๆ ให้โมเดลใช้ความรู้ตัวเองตอบ
    template = """คุณคือ AI ผู้ช่วยที่มีความรู้กว้างขวาง จงตอบคำถามต่อไปนี้อย่างกระชับและถูกต้องที่สุด:

คำถาม: {query}
คำตอบ:"""
    
    prompt = PromptTemplate.from_template(template)
    return prompt | llm

# ---------------------------------------------------------------------------
# Core pipeline
# ---------------------------------------------------------------------------

def load_dataset(path: Path) -> list[dict[str, Any]]:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)

def print_summary(results: list[dict[str, Any]]) -> None:
    total = len(results)
    if total == 0:
        return

    print()
    for level in DIFFICULTY_LEVELS:
        subset = [r for r in results if r["difficulty"] == level]
        if not subset:
            continue
        acc = sum(r["accuracy"] for r in subset) / len(subset)
        avg_lat = sum(r["latency"] for r in subset) / len(subset)
        print(
            f"[{level.upper():6}] "
            f"Accuracy: {acc:.1%} | "
            f"Avg latency: {avg_lat:.2f}s | "
            f"n={len(subset)}"
        )

    avg_latency = sum(r["latency"] for r in results) / total
    overall_acc = sum(r["accuracy"] for r in results) / total

    print(f"\n{'=' * 45}")
    print(f"Overall Accuracy (NO RAG) : {overall_acc:.1%}")
    print(f"Avg Latency               : {avg_latency:.2f}s")
    print(f"Total questions           : {total}")
    print(f"{'=' * 45}\n")

def run_raw_evaluation(dataset_path: Path, output_path: Path):
    logger.info("Initializing Raw LLM Chain...")
    chain = get_raw_llm_chain()

    dataset = load_dataset(dataset_path)
    logger.info("Loaded %d questions from %s", len(dataset), dataset_path.name)

    results = []
    for item in dataset:
        logger.info("Running %s...", item["id"])
        start = time.time()

        # 1. Generate Answer (No RAG)
        try:
            response = chain.invoke({"query": item["question"]})
            answer = response.content
        except Exception as e:
            logger.error("LLM Error on %s: %s", item["id"], e)
            answer = f"ERROR: {e}"

        latency = time.time() - start

        # 2. Evaluate Accuracy against Ground Truth
        ground_truth = item.get("ground_truth", "")
        acc_verdict = evaluate_accuracy(
            question=item["question"],
            ground_truth=ground_truth,
            answer=answer,
        )
        acc_passed = acc_verdict == "PASS"

        logger.info("  %s → A:%s (%.1fs)", item["id"], "PASS" if acc_passed else "FAIL", latency)

        results.append({
            "id": item["id"],
            "difficulty": item["difficulty"],
            "latency": round(latency, 2),
            "accuracy": acc_passed,
            "acc_verdict": acc_verdict,
            "answer": answer,
        })

    # Save Results
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(results, fh, ensure_ascii=False, indent=2)
    
    logger.info("Saved raw baseline results to %s", output_path)
    print_summary(results)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Raw LLM Evaluation (Baseline).")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    
    run_raw_evaluation(args.dataset, args.output)