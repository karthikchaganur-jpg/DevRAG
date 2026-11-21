from __future__ import annotations

import os
from typing import Any


def configured_provider() -> str | None:
    provider = os.getenv("DEVRAG_PROVIDER", "auto").strip().lower()
    if provider == "none":
        return None
    if provider == "gemini":
        return "gemini" if os.getenv("GEMINI_API_KEY") else None
    if provider == "openai":
        return "openai" if os.getenv("OPENAI_API_KEY") else None
    if os.getenv("GEMINI_API_KEY"):
        return "gemini"
    if os.getenv("OPENAI_API_KEY"):
        return "openai"
    return None


def generate_answer(question: str, sources: list[dict[str, Any]]) -> str | None:
    """Use the configured hosted model, or return None for local fallback."""
    provider = configured_provider()
    if not provider:
        return None
    context = "\n\n".join(
        f"[{source['path']}:{source['start']}-{source['end']}]\n{source['text']}"
        for source in sources
    )
    system_instruction = (
        "Answer using only the supplied code excerpts. Cite each claim as [path:line]. "
        "If the excerpts do not establish something, say what is missing. Keep the answer concise."
    )
    try:
        if provider == "gemini":
            from google import genai
            from google.genai import types

            client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
            result = client.models.generate_content(
                model=os.getenv("DEVRAG_GEMINI_MODEL", "gemini-3.5-flash-lite"),
                contents=f"Question: {question}\n\nCode excerpts:\n{context}",
                config=types.GenerateContentConfig(
                    system_instruction=system_instruction,
                    temperature=0.2,
                ),
            )
            return result.text or None

        from openai import OpenAI

        result = OpenAI(api_key=os.environ["OPENAI_API_KEY"]).chat.completions.create(
            model=os.getenv("DEVRAG_OPENAI_MODEL", os.getenv("DEVRAG_MODEL", "gpt-4o-mini")),
            messages=[
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": f"Question: {question}\n\nCode excerpts:\n{context}"},
            ],
            temperature=0.2,
        )
        return result.choices[0].message.content or None
    except Exception:
        return None
