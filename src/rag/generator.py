"""RAG answer generator with Agentic capabilities.

Uses a LangChain AgentExecutor to autonomously decide when to search
the knowledge base, allowing for multi-step reasoning and dynamic query
formulation.
"""

import logging
import os
import time
from typing import Any, List, Optional, Tuple

from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

from src.rag.vectorstore import HybridRetriever

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

API_ERROR_MESSAGE = "⚠️ ขออภัย ระบบประมวลผลหนาแน่น (API Error) กรุณาลองใหม่อีกครั้ง"


class RAGAgent:
    """Agentic RAG assistant that uses a HybridRetriever to answer questions."""

    def __init__(
        self,
        retriever: HybridRetriever,
        model_name: str = "typhoon-v2.5-30b-a3b-instruct",
        temperature: float = 0.1,
        max_retries: int = 3,
    ):
        self.retriever = retriever
        self.max_retries = max_retries
        self.llm = ChatOpenAI(
            model=model_name,
            api_key=os.environ.get("TYPHOON_API_KEY"),
            base_url="https://api.opentyphoon.ai/v1",
            temperature=temperature,
            max_tokens=10000,
        )
        self.retrieved_docs: List[Document] = []

    def _get_tools(self) -> List[Any]:
        """Define tools for the agent."""

        @tool
        def search_knowledge_base(search_query: str) -> str:
            """Search the enterprise knowledge base for relevant documents. 
            Always use this to answer questions about the company or the uploaded documents."""
            docs = self.retriever.search(search_query)
            self.retrieved_docs.extend(docs)
            if not docs:
                return "No relevant documents found for this query."
            return "\n\n".join(doc.page_content for doc in docs)

        return [search_knowledge_base]

    def _get_prompt(self) -> ChatPromptTemplate:
        """Construct the agent prompt."""
        return ChatPromptTemplate.from_messages([
            ("system", (
                "คุณคือ AI ผู้ช่วยองค์กรที่มีความเชี่ยวชาญ "
                "จงตอบคำถามโดยใช้ข้อมูลจากเครื่องมือค้นหาข้อมูล (search_knowledge_base) "
                "หากจำเป็นต้องค้นหาข้อมูลเพิ่มเติม สามารถค้นหาได้หลายครั้ง "
                "หากไม่พบข้อมูลในเอกสารอ้างอิง ให้ตอบว่า 'ขออภัย ไม่พบข้อมูลในเอกสารอ้างอิง' "
                "ห้ามแต่งข้อมูลขึ้นมาเอง"
            )),
            ("user", "ประวัติการสนทนา:\n{chat_history}\n\nคำถามปัจจุบัน: {input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])

    def generate(
        self, 
        query: str, 
        chat_history: str = ""
    ) -> Tuple[str, List[Document], List[Tuple[Any, str]]]:
        """Generate a grounded answer for the given query."""
        self.retrieved_docs = []  # Reset for new generation
        
        tools = self._get_tools()
        prompt = self._get_prompt()
        agent = create_tool_calling_agent(self.llm, tools, prompt)
        
        executor = AgentExecutor(
            agent=agent,
            tools=tools,
            return_intermediate_steps=True,
            verbose=True
        )

        for attempt in range(self.max_retries):
            try:
                response = executor.invoke({
                    "input": query,
                    "chat_history": chat_history,
                })
                return response["output"], self.retrieved_docs, response.get("intermediate_steps", [])
            except Exception as exc:
                logger.warning("Agent execution error (attempt %d/%d): %s", attempt + 1, self.max_retries, exc)
                if attempt == self.max_retries - 1:
                    logger.error("All %d retries exhausted", self.max_retries)
                    return API_ERROR_MESSAGE, [], []
                time.sleep(2 ** attempt)

        return API_ERROR_MESSAGE, [], []


def generate_answer(
    query: str,
    vectorstore,
    bm25_retriever,
    chat_history: str = "",
    reranker=None,
) -> Tuple[str, List[Document], List[Tuple[Any, str]]]:
    """Legacy wrapper for generate_answer to maintain backward compatibility."""
    retriever = HybridRetriever(vectorstore, bm25_retriever, reranker)
    agent = RAGAgent(retriever)
    return agent.generate(query, chat_history)
