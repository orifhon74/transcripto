# media_core/translation.py

from openai import OpenAI
import json

_client = OpenAI()

SUPPORTED_LANGS = {
    "uz": "Uzbek",
    "en": "English",
    "ru": "Russian",
    # add more as needed
}

def translate_texts(texts: list[str], target_lang: str, model: str = "gpt-4o-mini") -> list[str]:
    cleaned = [(t or "").strip() for t in texts]
    if not any(cleaned):
        return [""] * len(cleaned)

    if target_lang not in SUPPORTED_LANGS:
        # fallback: just echo original
        return cleaned

    out = [""] * len(cleaned)
    BATCH = 60

    for i in range(0, len(cleaned), BATCH):
        chunk = cleaned[i:i+BATCH]
        prompt = (
            f"Translate the following lines to {SUPPORTED_LANGS[target_lang]} "
            "(language code: " + target_lang + "). "
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
            arr = json.loads(content)
            if isinstance(arr, list):
                for j, v in enumerate(arr):
                    out[i + j] = (v or "").strip()
        except Exception:
            for j, v in enumerate(chunk):
                out[i + j] = v
    return out