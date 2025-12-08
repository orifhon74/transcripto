import json
from typing import List

from openai import OpenAI

_client = OpenAI()  # uses OPENAI_API_KEY from env


# Languages we explicitly support (can expand anytime)
SUPPORTED_LANGS = {
    "uz": "Uzbek",
    "ru": "Russian",
    "es": "Spanish",
    "fr": "French",
    "de": "German",
    "tr": "Turkish",
    # "ar": "Arabic",
    "zh": "Chinese (Simplified)",
    "ja": "Japanese",
    "ko": "Korean",
    "hi": "Hindi",
}


def get_supported_languages() -> dict:
    """
    Small helper if you ever want to show the list in the UI.
    Returns {code: human_name}.
    """
    return dict(SUPPORTED_LANGS)


def translate_texts(
    texts: List[str],
    target_lang: str,
    model: str = "gpt-4o-mini",
) -> List[str]:
    """
    Generic translation helper.

    - texts: list of strings
    - target_lang: language code from SUPPORTED_LANGS (e.g., "uz", "ru", "en")
    - returns list with same length
    - if target_lang unsupported, returns original texts as-is
    """
    cleaned = [(t or "").strip() for t in texts]
    if not any(cleaned):
        return [""] * len(cleaned)

    if target_lang not in SUPPORTED_LANGS:
        # Unknown language code -> no-op
        return cleaned

    out = [""] * len(cleaned)
    BATCH = 60  # keep chunk size reasonable

    for i in range(0, len(cleaned), BATCH):
        chunk = cleaned[i:i + BATCH]
        prompt = (
            f"Translate the following lines to {SUPPORTED_LANGS[target_lang]} "
            f"(language code: {target_lang}). "
            "Return ONLY a JSON array of strings, same order and length, no extra commentary.\n\n"
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
            else:
                # malformed -> fallback to originals
                for j, v in enumerate(chunk):
                    out[i + j] = v
        except Exception:
            # graceful fallback: keep originals if translation fails
            for j, v in enumerate(chunk):
                out[i + j] = v

    return out


# Backwards-compat helper so old code keeps working
def translate_texts_to_uz(texts: List[str]) -> List[str]:
    """
    Legacy helper: translate to Uzbek.
    Internally just calls translate_texts(texts, "uz").
    """
    return translate_texts(texts, "uz")