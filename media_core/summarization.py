# media_core/summarization.py

import json
from openai import OpenAI

_client = OpenAI()  # reads OPENAI_API_KEY


def summarize_text(text: str, model: str = "gpt-4o-mini") -> str:
    text = (text or "").strip()
    if not text:
        return ""
    prompt = (
        "Provide a clear, concise summary of the following content. "
        "Keep the original language when possible. If there are lists, preserve bullets or emojis. "
        "Aim for 3–6 short bullet points followed by a brief overall conclusion.\n\n"
        f"{text[:8000]}"
    )
    try:
        resp = _client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a concise technical summarizer."},
                {"role": "user", "content": prompt},
            ],
            max_tokens=300,
            temperature=0.3,
        )
        return (resp.choices[0].message.content or "").strip()
    except Exception as e:
        return f"(Summary unavailable: {e})"