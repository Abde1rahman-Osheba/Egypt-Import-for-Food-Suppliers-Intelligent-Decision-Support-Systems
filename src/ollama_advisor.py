"""Optional Ollama advisor using /api/generate; never crashes if offline."""

from __future__ import annotations

from typing import Any, Optional, Tuple

try:
    import requests
except ImportError:
    requests = None

from src.config import OLLAMA_BASE_URL, OLLAMA_MODEL


def ollama_generate_text(
    prompt: str,
    model: Optional[str] = None,
    base_url: Optional[str] = None,
    timeout: int = 90,
) -> Tuple[Optional[str], Optional[str]]:
    if requests is None:
        return None, "requests not installed"
    base = (base_url or OLLAMA_BASE_URL).rstrip("/")
    payload = {"model": model or OLLAMA_MODEL, "prompt": prompt, "stream": False}
    try:
        r = requests.post(f"{base}/api/generate", json=payload, timeout=timeout)
        r.raise_for_status()
        data = r.json()
        text = data.get("response")
        if text:
            return str(text).strip(), None
        return None, "Empty Ollama response"
    except Exception as e:
        return None, str(e)


def compact_maritime_context(
    selected_port_id: str,
    selected_ship_id: str,
    context: dict[str, Any],
    vessels_summary: str,
    ports_summary: str,
    zones_summary: str,
    alerts_summary: str,
) -> str:
    return (
        "MARITIME DSS CONTEXT (structured, do not invent numbers):\n"
        f"- Executive: unified_risk={context.get('unified_risk')}, pss={context.get('pss')}, "
        f"log={context.get('log')}, geo={context.get('geo')}\n"
        f"- Selected port_id={selected_port_id or 'none'}\n"
        f"- Selected ship_id={selected_ship_id or 'none'}\n"
        f"- Vessels: {vessels_summary}\n"
        f"- Ports: {ports_summary}\n"
        f"- Risk zones: {zones_summary}\n"
        f"- Alerts: {alerts_summary}\n"
        "Answer in concise operational language; cite only fields above."
    )


def advisor_route_explanation(
    ship_name: str,
    origin: str,
    dest: str,
    cargo: str,
    route_risk: float,
    near_zone: bool,
    rec: str,
    alt_explain: str,
    ollama_on: bool,
    model: str,
    base_url: str,
) -> str:
    prompt = (
        "You advise Egypt Ministry of Transport / food import monitoring. "
        f"Vessel {ship_name} from {origin} to {dest}, cargo {cargo}. "
        f"Route risk score {route_risk:.0f}/100, near_zone={near_zone}, system rec={rec}. "
        f"Detour note: {alt_explain}. Give 3 bullets: situation, action, monitoring."
    )
    if not ollama_on:
        return "Ollama is not available. Showing rule-based recommendation instead. " + (
            f"Keep {rec}: route risk {route_risk:.0f}; conflict proximity {near_zone}. {alt_explain}"
        )
    text, err = ollama_generate_text(prompt, model=model, base_url=base_url)
    if err or not text:
        return (
            "Ollama is not available or returned an error. Showing rule-based recommendation instead. "
            f"{rec.upper()}: risk {route_risk:.0f}/100. {alt_explain}"
        )
    return text
