"""RAG answer generator.

Sends a user query plus retrieved context to the OpenTyphoon LLM and
returns a grounded answer.  Includes automatic retry with exponential
back-off for transient API failures.
"""

import logging
import os
import time
from typing import List, Optional, Tuple

from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# LLM configuration
# ---------------------------------------------------------------------------

_TYPHOON_MODEL = "typhoon-v2.5-30b-a3b-instruct"
_TYPHOON_BASE_URL = "https://api.opentyphoon.ai/v1"
_TYPHOON_TEMPERATURE = 0.1
_TYPHOON_MAX_TOKENS = 10000
_MAX_RETRIES = 3

_NO_DOCS_MESSAGE = (
    "ขออภัย ไม่พบข้อมูลที่เกี่ยวข้องในฐานข้อมูลเอกสารปัจจุบัน "
    "กรุณาลองปรับคำค้นหา"
)
_API_ERROR_MESSAGE = (
    "⚠️ ขออภัย ระบบประมวลผลหนาแน่น (API Error) กรุณาลองใหม่อีกครั้ง"
)

_PROMPT_TEMPLATE = """\
คุณคือ AI ผู้ช่วยองค์กรที่มีความเชี่ยวชาญ จงตอบคำถามโดยใช้ข้อมูลจาก Context ที่ให้มาเท่านั้น
หากคำตอบไม่มีใน Context ให้ตอบว่า "ขออภัย ไม่พบข้อมูลในเอกสารอ้างอิง" ห้ามแต่งข้อมูลขึ้นมาเอง

[ประวัติการสนทนาก่อนหน้า]
{chat_history}

[ข้อมูลอ้างอิง (Context)]
{context}

คำถามปัจจุบัน: {query}
คำตอบ:
"""


# ---------------------------------------------------------------------------
# LLM factory
# ---------------------------------------------------------------------------

def _get_llm() -> ChatOpenAI:
    """Create a ChatOpenAI client configured for the Typhoon API.

    Returns:
        A ``ChatOpenAI`` instance connected to OpenTyphoon.
    """
    return ChatOpenAI(
        model=_TYPHOON_MODEL,
        api_key=os.environ.get("TYPHOON_API_KEY"),
        base_url=_TYPHOON_BASE_URL,
        temperature=_TYPHOON_TEMPERATURE,
        max_tokens=_TYPHOON_MAX_TOKENS,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_answer(
    query: str,
    retrieved_docs: List[Document],
    chat_history: str = "",
) -> Tuple[str, List[Document]]:
    """Generate a grounded answer using retrieved documents as context.

    If *retrieved_docs* is empty the function returns a "no data" message
    immediately without making an API call.

    Args:
        query: The user's question.
        retrieved_docs: LangChain ``Document`` objects from hybrid search.
        chat_history: Prior conversation turns formatted as a string.

    Returns:
        A ``(answer_text, source_documents)`` tuple.  On failure the answer
        contains an error message and source list is empty.
    """
    if not retrieved_docs:
        return _NO_DOCS_MESSAGE, []

    llm = _get_llm()
    context_text = "\n\n".join(doc.page_content for doc in retrieved_docs)

    prompt = PromptTemplate.from_template(_PROMPT_TEMPLATE)
    chain = prompt | llm

    for attempt in range(_MAX_RETRIES):
        try:
            response = chain.invoke({
                "context": context_text,
                "query": query,
                "chat_history": chat_history,
            })
            return response.content, retrieved_docs

        except Exception as exc:
            logger.warning(
                "LLM API error (attempt %d/%d): %s",
                attempt + 1, _MAX_RETRIES, exc,
            )
            if attempt == _MAX_RETRIES - 1:
                logger.error("All %d LLM retries exhausted", _MAX_RETRIES)
                return _API_ERROR_MESSAGE, []
            time.sleep(2 ** attempt)

    # Unreachable, but satisfies type checkers.
    return _API_ERROR_MESSAGE, []