"""
Wrapper defensif untuk inspeksi ban lewat OpenRouter, dirancang untuk model
free-tier yang dukungannya terhadap structured output tidak seragam:

    - google/gemma-4-26b-a4b-it:free   (structured output resmi didukung)
    - google/gemma-4-31b-it:free       (tidak eksplisit didukung -> defensif)
    - nvidia/nemotron-nano-12b-v2-vl:free  (reasoning model -> bisa selipkan <think>)

Strategi:
1. Coba response_format=json_schema (strict).
2. Kalau provider menolak param itu (400 error) -> fallback ke json_object.
3. Kalau tetap gagal / hasil bukan JSON valid -> fallback ke prompt-based JSON,
   bersihkan reasoning trace (<think>...</think>), ekstrak blok JSON via regex.
4. Semua hasil, dari jalur manapun, WAJIB lolos validasi pydantic sebelum dipakai.
5. Kalau satu model gagal total (auth/rate-limit/parse gagal semua fallback),
   lanjut ke model berikutnya di daftar (cascade).
"""

import re
import json
import httpx
from typing import Literal, Optional
from pydantic import BaseModel, Field, ValidationError

from tyre_inspection_schema import TYRE_INSPECTION_SCHEMA, SYSTEM_PROMPT  # sesuaikan import SYSTEM_PROMPT

GEMINI_OPENAI_URL = "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions"

MODEL_CASCADE = [
    "gemini-3.5-flash"
]


# ---------- Skema validasi ketat di sisi client (pydantic) ----------
class KriteriaItem(BaseModel):
    status: Literal["aman", "bermasalah", "tidak_terlihat"]
    alasan: str


class TyreInspectionResult(BaseModel):
    is_tyre: bool
    status: Literal["good", "warning", "bad", "tidak_dapat_ditentukan", "invalid"]
    label: str
    kriteria: dict[str, KriteriaItem]
    confidence_percent: int = Field(ge=0, le=100)
    message: str
    recommendation: str


def _strip_reasoning_trace(text: str) -> str:
    """Buang <think>...</think> atau blok reasoning lain yang kadang
    diselipkan oleh model reasoning (mis. Nemotron) sebelum JSON final."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    return text.strip()


def _extract_json_block(text: str) -> Optional[str]:
    """Ekstrak blok {...} pertama yang valid dari teks bebas (fallback terakhir)."""
    text = _strip_reasoning_trace(text)
    # buang code fence ```json ... ``` kalau ada
    text = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.MULTILINE).strip()
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    return match.group(0) if match else None


async def _call_openrouter(client: httpx.AsyncClient, api_key: str, model: str,
                            image_data_url: str, response_format: Optional[dict]) -> dict:
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Analisis gambar ban berikut sesuai instruksi sistem. Jawab HANYA JSON."},
                    {"type": "image_url", "image_url": {"url": image_data_url}},
                ],
            },
        ],
        "temperature": 0.2,
    }
    if response_format:
        payload["response_format"] = response_format

    resp = await client.post(
        GEMINI_OPENAI_URL,
        json=payload,
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        timeout=25,
    )
    resp.raise_for_status()
    return resp.json()


async def inspect_tyre_with_cascade(api_key: str, image_data_url: str) -> TyreInspectionResult:
    last_error = None

    for model in MODEL_CASCADE:
        print(f"\\n[LLM CASCADE] Mencoba model: {model}")
        async with httpx.AsyncClient() as client:
            # --- Percobaan 1: strict json_schema ---
            try:
                print(f"  -> [1] Mencoba json_schema (strict) dengan {model}...")
                data = await _call_openrouter(client, api_key, model, image_data_url, TYRE_INSPECTION_SCHEMA)
                content = data["choices"][0]["message"]["content"]
                result = TyreInspectionResult.model_validate_json(content)
                print(f"  -> [1] SUKSES!")
                return result  # sukses, langsung pulang
            except Exception as e:
                last_error = f"[{model}] json_schema gagal: {e}"
                print(f"  -> [1] GAGAL: {e}")

            # --- Percobaan 2: json_object (lebih longgar, tanpa schema strict) ---
            try:
                print(f"  -> [2] Mencoba json_object (longgar) dengan {model}...")
                data = await _call_openrouter(
                    client, api_key, model, image_data_url, {"type": "json_object"}
                )
                content = data["choices"][0]["message"]["content"]
                content = _strip_reasoning_trace(content)
                result = TyreInspectionResult.model_validate_json(content)
                print(f"  -> [2] SUKSES!")
                return result
            except Exception as e:
                last_error = f"[{model}] json_object gagal: {e}"
                print(f"  -> [2] GAGAL: {e}")

            # --- Percobaan 3: prompt-based, ekstrak JSON manual ---
            try:
                print(f"  -> [3] Mencoba prompt-based (ekstrak regex) dengan {model}...")
                data = await _call_openrouter(client, api_key, model, image_data_url, None)
                content = data["choices"][0]["message"]["content"]
                json_block = _extract_json_block(content)
                if not json_block:
                    raise ValueError("Tidak ada blok JSON ditemukan di respons.")
                result = TyreInspectionResult.model_validate_json(json_block)
                print(f"  -> [3] SUKSES!")
                return result
            except Exception as e:
                last_error = f"[{model}] prompt-based gagal: {e}"
                print(f"  -> [3] GAGAL: {e}")
                continue  # lanjut ke model berikutnya di cascade

    raise RuntimeError(f"Semua model di cascade gagal. Error terakhir: {last_error}")
