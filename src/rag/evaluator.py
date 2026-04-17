"""
Evaluation module for RAG answer quality assessment.

Provides two evaluation strategies:
- Faithfulness: checks if an answer is grounded in the provided context documents.
- Accuracy: checks if an answer matches expected ground truth (for baseline comparisons).

Both evaluators use the Typhoon API as an LLM judge with structured JSON output.
"""

import json
import logging
import os
import re
import time
from typing import Optional

import requests

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

TYPHOON_API_URL = "https://api.opentyphoon.ai/v1/chat/completions"
TYPHOON_MODEL = "typhoon-v2.5-30b-a3b-instruct"
TYPHOON_TEMPERATURE = 0.0
TYPHOON_MAX_TOKENS = 4000
TYPHOON_TIMEOUT_SECONDS = 90

MAX_PROMPT_LENGTH = 6_000
MAX_ANSWER_LENGTH = 500
MAX_CONTEXT_LENGTH = 3_000

MAX_RETRIES = 3
RATE_LIMIT_WAIT_SECONDS = 20


# ---------------------------------------------------------------------------
# Typhoon API client
# ---------------------------------------------------------------------------

def _call_typhoon_api(prompt: str) -> str:
    """Send a prompt to the Typhoon chat completions API and return the response text.

    Handles prompt truncation, rate-limit retries, and HTTP error codes.

    Args:
        prompt: The full prompt string to send.

    Returns:
        The assistant's response content, or an ``ERROR_*`` prefixed string on failure.
    """
    api_key = os.environ.get("TYPHOON_API_KEY")
    if not api_key:
        return "ERROR_CONFIG: TYPHOON_API_KEY environment variable is not set"

    # Truncate overly long prompts to stay within token limits.
    if len(prompt) > MAX_PROMPT_LENGTH:
        half = MAX_PROMPT_LENGTH // 2
        prompt = prompt[:half] + "\n...[truncated]...\n" + prompt[-half:]

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": TYPHOON_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": TYPHOON_TEMPERATURE,
        "max_tokens": TYPHOON_MAX_TOKENS,
    }

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = requests.post(
                TYPHOON_API_URL,
                headers=headers,
                json=payload,
                timeout=TYPHOON_TIMEOUT_SECONDS,
            )

            if response.status_code == 200:
                return response.json()["choices"][0]["message"]["content"]

            if response.status_code == 429:
                logger.warning(
                    "Rate-limited by Typhoon API (attempt %d/%d). Waiting %ds...",
                    attempt, MAX_RETRIES, RATE_LIMIT_WAIT_SECONDS,
                )
                time.sleep(RATE_LIMIT_WAIT_SECONDS)
                continue

            return f"ERROR_API: {response.status_code} - {response.text}"

        except requests.RequestException as exc:
            logger.error("Typhoon API request failed: %s", exc)
            return f"ERROR_REQ: {exc}"

    return "ERROR: Max retries exceeded"


# ---------------------------------------------------------------------------
# Verdict parsing
# ---------------------------------------------------------------------------

def _parse_verdict(content: str) -> str:
    """Extract a PASS / FAIL verdict from an LLM response.

    Attempts JSON parsing first, then falls back to keyword matching.

    Args:
        content: Raw LLM response text.

    Returns:
        ``"PASS"``, ``"FAIL"``, or the original ``ERROR_*`` string.
    """
    print(f"\n[DEBUG LLM]: {content}\n")
    if content.startswith("ERROR"):
        return content

    # Strip markdown code-fences the model sometimes adds.
    cleaned = re.sub(r"```json|```", "", content).strip()

    try:
        data = json.loads(cleaned)
        return data.get("verdict", "FAIL").upper()
    except (json.JSONDecodeError, AttributeError):
        pass

    # Fallback: simple keyword search.
    if "PASS" in content.upper():
        return "PASS"
    return "FAIL"


# ---------------------------------------------------------------------------
# Public evaluation functions
# ---------------------------------------------------------------------------

def evaluate_accuracy(question: str, ground_truth: str, answer: str) -> str:
    """Judge whether *answer* is factually consistent with *ground_truth*.

    Intended for baseline (no-RAG) evaluation where there is no retrieval
    context — we simply compare the model's answer against a known-correct answer.

    Args:
        question: The original question.
        ground_truth: The expected/correct answer.
        answer: The model-generated answer to evaluate.

    Returns:
        ``"PASS"`` if the answer is substantively correct, ``"FAIL"`` otherwise.
    """
    truncated_answer = answer[:MAX_ANSWER_LENGTH]

    prompt = (
        "คุณคือระบบตรวจคำตอบระดับผู้เชี่ยวชาญ\n"
        "ตัดสินว่า 'คำตอบจากระบบ' มีเนื้อหาถูกต้องตาม 'เฉลย' หรือไม่ "
        "(เน้นสาระสำคัญ ไม่ต้องตรงทุกคำ)\n\n"
        f"โจทย์: {question}\n"
        f"เฉลย: {ground_truth}\n"
        f"คำตอบ: {truncated_answer}\n\n"
        'ตอบเป็น JSON เท่านั้น: {"verdict": "PASS"} หรือ {"verdict": "FAIL"}'
    )
    return _parse_verdict(_call_typhoon_api(prompt))


def evaluate_faithfulness(
    context_docs: list,
    generated_answer: str,
) -> str:
    """Judge whether *generated_answer* is fully grounded in *context_docs*.

    Used during RAG evaluation to detect hallucination — the answer should
    contain only information present in the retrieved documents.

    Args:
        context_docs: List of LangChain ``Document`` objects used as context.
        generated_answer: The RAG-generated answer to evaluate.

    Returns:
        ``"PASS"`` if faithful, ``"FAIL"`` if hallucination detected.
    """
    if not context_docs:
        return "FAIL"

    context_text = "\n\n".join(
        doc.page_content for doc in context_docs
    )[:MAX_CONTEXT_LENGTH]

    truncated_answer = generated_answer[:MAX_ANSWER_LENGTH]

    prompt = (
        "คุณคือระบบตรวจสอบความถูกต้อง "
        "ตรวจสอบว่า 'คำตอบ' สอดคล้องและมาจาก 'ข้อมูลอ้างอิง' หรือไม่\n"
        "**เงื่อนไขสำคัญ**: หาก 'คำตอบ' ระบุความหมายในทำนองว่า 'ขออภัย ไม่พบข้อมูล', 'ไม่มีข้อมูลในเอกสาร', หรือปฏิเสธการตอบเนื่องจากข้อมูลไม่พอ ให้คุณตัดสินว่าสอดคล้องและตอบ PASS ทันที\n\n"
        f"อ้างอิง: {context_text}\n"
        f"คำตอบ: {truncated_answer}\n\n"
        'ตอบเป็น JSON เท่านั้น: {"verdict": "PASS"} หรือ {"verdict": "FAIL"}'
    )
    return _parse_verdict(_call_typhoon_api(prompt))