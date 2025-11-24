# media_core/translation.py

import json
from openai import OpenAI

_client = OpenAI()  # reads OPENAI_API_KEY


def translate_texts_to_uz(texts: list[str], model: str = "gpt-4o-mini") -> list[str]:
    """
    Translate a list of short strings to Uzbek (uz). Returns a list of same length.
    Keeps punctuation; avoids adding extra notes. Empty inputs -> empty outputs.
    Batches to stay under token limits.
    """
    cleaned = [(t or "").strip() for t in texts]
    if not any(cleaned):
        return [""] * len(texts)

    out = [""] * len(cleaned)

    BATCH = 60
    for i in range(0, len(cleaned), BATCH):
        chunk = cleaned[i:i + BATCH]
        prompt = (
            "Translate the following lines to Uzbek (uz). "
            "Return ONLY a JSON array of strings, same order and length, no extra text.\n\n"
            + json.dumps(chunk, ensure_ascii=False)
        )
        try:
            resp = _client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a precise translator."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.2,
                max_tokens=1600,
            )
            content = (resp.choices[0].message.content or "").strip()
            import json as _json
            arr = _json.loads(content)
            if isinstance(arr, list):
                for j, v in enumerate(arr):
                    out[i + j] = (v or "").strip()
        except Exception:
            # graceful fallback: keep English if translation fails
            for j, v in enumerate(chunk):
                out[i + j] = v
    return out