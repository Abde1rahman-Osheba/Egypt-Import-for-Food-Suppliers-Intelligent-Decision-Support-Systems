"""
Optional Ollama (local LLM) integration for section-level narratives.
Uses HTTP API only — run `ollama serve` and pull a model (e.g. `ollama pull llama3.2`).
"""

from __future__ import annotations

import os
from typing import Optional

import streamlit as st

try:
    import requests
except ImportError:
    requests = None

DEFAULT_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
DEFAULT_MODEL = os.environ.get("OLLAMA_MODEL", "llama3.2")
SYSTEM_PROMPT = (
    "You are an expert assistant for Egyptian food importers and intelligent decision support systems. "
    "Be concise, use markdown (short bullets), stay factual to the numbers provided, and suggest "
    "practical next checks. If the user writes Arabic, reply in Arabic or bilingual as appropriate."
)


def _url(base: str, path: str) -> str:
    return base.rstrip("/") + path


def ollama_ping(base_url: str, timeout: int = 5) -> tuple[bool, str]:
    if requests is None:
        return False, "Python package `requests` is not installed."
    try:
        r = requests.get(_url(base_url, "/api/tags"), timeout=timeout)
        r.raise_for_status()
        return True, "Ollama reachable."
    except Exception as e:
        return False, f"Cannot reach Ollama at {base_url}: {e}"


def ollama_chat(
    base_url: str,
    model: str,
    user_message: str,
    timeout: int = 120,
) -> tuple[Optional[str], Optional[str]]:
    """
    Non-streaming chat. Returns (assistant_text, error_string).
    """
    if requests is None:
        return None, "Install `requests` to use Ollama."
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        "stream": False,
    }
    try:
        r = requests.post(_url(base_url, "/api/chat"), json=payload, timeout=timeout)
        r.raise_for_status()
        data = r.json()
        msg = data.get("message") or {}
        text = msg.get("content")
        if text:
            return str(text).strip(), None
        return None, "Empty response from Ollama."
    except Exception as e:
        return None, str(e)


def init_ollama_session_state() -> None:
    """Ollama is always on for explainers; URL/model persist in session state."""
    st.session_state.ollama_enabled = True
    if "ollama_base" not in st.session_state:
        st.session_state.ollama_base = DEFAULT_BASE_URL
    if "ollama_model" not in st.session_state:
        st.session_state.ollama_model = DEFAULT_MODEL


def render_ollama_sidebar() -> None:
    """Widgets placed inside an existing `with st.sidebar:` block."""
    init_ollama_session_state()
    st.markdown("---")
    st.markdown("### Ollama (local LLM)")
    st.caption(
        "Section explainers and route advisor use Ollama when reachable. Requires `ollama serve` and a pulled model."
    )
    st.session_state.ollama_base = st.text_input(
        "Ollama base URL",
        value=st.session_state.ollama_base,
    )
    st.session_state.ollama_model = st.text_input(
        "Model",
        value=st.session_state.ollama_model,
        help="Example: llama3.2, mistral, qwen2.5",
    )
    if st.button("Test connection", key="ollama_ping_btn"):
        ok, msg = ollama_ping(st.session_state.ollama_base)
        if ok:
            st.success(msg)
        else:
            st.error(msg)


def section_ollama_explainer(
    section_id: str,
    page_title: str,
    bullets: list[str],
    extra_context: str = "",
) -> None:
    """
    Per-page expander: build a prompt from bullets + optional user question; call Ollama on demand.
    """
    init_ollama_session_state()
    with st.expander(page_title, expanded=False):
        user_q = st.text_area(
            "Your question (optional)",
            key=f"ollama_q_{section_id}",
            height=68,
            placeholder="e.g. What should procurement do if NLP conflict rises but prices are flat?",
        )
        c1, c2 = st.columns(2)
        with c1:
            go = st.button("Generate explanation", key=f"ollama_go_{section_id}", type="primary")
        with c2:
            if st.button("Clear cached reply", key=f"ollama_clr_{section_id}"):
                st.session_state.pop(f"ollama_txt_{section_id}", None)
                st.rerun()

        cache_key = f"ollama_txt_{section_id}"
        if go:
            facts = "\n".join(f"- {b}" for b in bullets if b)
            block = f"Context / facts:\n{facts}\n"
            if extra_context.strip():
                block += f"\nAdditional context:\n{extra_context.strip()}\n"
            prompt = (
                f"Dashboard section: **{page_title}**\n\n{block}\n"
                f"Task: {user_q.strip() or 'Summarize this view for an import risk manager: what it means, key drivers, and 2–3 concrete follow-ups.'}"
            )
            with st.spinner(f"Calling Ollama ({st.session_state.ollama_model})…"):
                text, err = ollama_chat(
                    st.session_state.ollama_base,
                    st.session_state.ollama_model,
                    prompt,
                )
            if err:
                st.error(f"Ollama error: {err}")
            elif text:
                st.session_state[cache_key] = text

        if cache_key in st.session_state:
            st.markdown(st.session_state[cache_key])
