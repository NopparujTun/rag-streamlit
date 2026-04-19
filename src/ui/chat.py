"""Chat UI components for the Streamlit application.

Provides functions to initialise, render, and manage the conversational
interface, including source-document display and evaluation metrics.
"""

from typing import Any, List, Optional, Tuple

import streamlit as st
from langchain_core.documents import Document

from src.utils.helpers import clean_markdown


# ---------------------------------------------------------------------------
# Chat state management
# ---------------------------------------------------------------------------

def init_chat_history() -> None:
    """Initialise the chat message history in Streamlit session state.

    Sets a default greeting message from the assistant if no prior
    conversation exists.
    """
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": "สวัสดีครับ มีอะไรให้ผมช่วยค้นหาในเอกสารขององค์กรไหมครับ?",
            }
        ]


def render_chat_history() -> None:
    """Render all messages in the session-state history to the chat UI."""
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            if "steps" in msg and msg["steps"]:
                display_agent_steps(msg["steps"])
            st.write(msg["content"])


def add_user_message(prompt: str) -> None:
    """Append a user message to the history and render it immediately.

    Args:
        prompt: The user's input text.
    """
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)


# ---------------------------------------------------------------------------
# Source display
# ---------------------------------------------------------------------------

def display_agent_steps(steps: List[Tuple[Any, str]]) -> None:
    """Render the intermediate reasoning steps of the agent."""
    if not steps:
        return
    with st.expander("🤔 ดูความคิดของ AI (Agent Thinking)", expanded=False):
        for i, (action, observation) in enumerate(steps):
            st.markdown(f"**Step {i+1}:** `Tool: {action.tool}`")
            if hasattr(action, "tool_input"):
                st.markdown(f"**Query:** `{action.tool_input}`")
            # handle if observation is long
            obs_str = str(observation)
            if len(obs_str) > 200:
                obs_str = obs_str[:200] + "..."
            st.caption(f"**Result:** {obs_str}")
            if i < len(steps) - 1:
                st.divider()

def display_sources(sources: List[Document]) -> None:
    """Render retrieved source documents in a collapsible expander.

    Each document is displayed with its page number and cleaned content.
    Horizontal dividers separate consecutive sources.

    Args:
        sources: LangChain ``Document`` objects used as retrieval context.
    """
    if not sources:
        return

    with st.expander("📚 ดูแหล่งอ้างอิง (Source Context)", expanded=False):
        for i, doc in enumerate(sources):
            page = doc.metadata.get("page", "Unknown")
            clean_content = clean_markdown(doc.page_content)
            st.markdown(f"**[หน้า {page}]**\n{clean_content}")

            if i < len(sources) - 1:
                st.divider()


# ---------------------------------------------------------------------------
# Evaluation metrics
# ---------------------------------------------------------------------------

def render_evaluation_metrics(response_time: float, eval_result: str) -> None:
    """Display response latency and faithfulness verdict as compact captions.

    Args:
        response_time: Time taken for the LLM to produce the answer (seconds).
        eval_result: Verdict string from the evaluator (contains ``"PASS"``,
                     ``"FAIL"``, or an error prefix).
    """
    col1, col2 = st.columns([1, 3])
    with col1:
        st.caption(f"⏱️ {response_time:.2f} วินาที")
    with col2:
        if "PASS" in eval_result:
            st.caption("✅ **Fact-Check:** PASS")
        elif "FAIL" in eval_result:
            st.caption("⚠️ **Fact-Check:** FAIL")
        else:
            st.caption("🔍 **Fact-Check:** N/A")