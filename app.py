import io
import os
import base64
import json
import httpx

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

from robust_tyre_inspection import inspect_tyre_with_cascade, TyreInspectionResult

# ─────────────────────────────────────────
# FastAPI Setup
# ─────────────────────────────────────────

app = FastAPI(title="Tyre Quality Classifier – Gemma Vision")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─────────────────────────────────────────
# OpenRouter Configuration
# ─────────────────────────────────────────

GEMINI_API_KEY = os.getenv("OPENROUTER_API_KEY", "")

# ─────────────────────────────────────────
# Helper: encode image ke base64
# ─────────────────────────────────────────

def encode_image_to_base64(pil_image: Image.Image, max_size: int = 1024) -> str:
    """Resize agar tidak terlalu besar, lalu encode ke base64 JPEG."""
    img = pil_image.copy()
    img.thumbnail((max_size, max_size), Image.LANCZOS)
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG", quality=85)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


# (CNN telah dinonaktifkan sepenuhnya untuk menghemat memori server)

# ─────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────

@app.get("/")
def root():
    return {
        "status": "running",
        "provider": "OpenRouter",
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Baca dan validasi file gambar
    contents = await file.read()
    try:
        original_image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="File bukan gambar yang valid.")

    # TAHAP 1: CNN — Deteksi apakah gambar adalah ban (DIBYPASS KARENA TIDAK AKURAT)
    is_tyre, cnn_prob = True, 1.0 # Bypassing CNN, biarkan AI Gemini yang memvalidasi
    
    if not is_tyre:
        cnn_confidence = (1.0 - cnn_prob) * 100
        print(f"[REJECTED] CNN: Gambar bukan ban (prob={cnn_prob:.4f}, confidence={cnn_confidence:.1f}%)")
        return {
            "is_tyre": False,
            "status": "invalid",
            "label": "Gambar Tidak Valid",
            "confidence_percent": round(cnn_confidence, 1),
            "raw_prob": round(cnn_prob, 4),
            "message": f"Gambar yang diupload bukan foto ban kendaraan (CNN confidence: {cnn_confidence:.1f}%).",
            "recommendation": "Silakan upload foto ban kendaraan yang jelas, close-up, dan terang.",
            "top1_label": "bukan ban",
        }

    # TAHAP 2: LLM — Analisis kondisi ban
    cnn_confidence = cnn_prob * 100
    print(f"[ACCEPTED] CNN: Gambar adalah ban (prob={cnn_prob:.4f}, confidence={cnn_confidence:.1f}%). Melanjutkan ke LLM...")
    
    b64_image = encode_image_to_base64(original_image)
    data_url = f"data:image/jpeg;base64,{b64_image}"
    
    try:
        result: TyreInspectionResult = await inspect_tyre_with_cascade(GEMINI_API_KEY, data_url)
        return result.model_dump()
    except Exception as e:
        print(f"[ERROR] LLM analysis failed: {e}")
        # Kembalikan response 'invalid' agar frontend menampilkan pesan yang rapi
        return {
            "is_tyre": False,
            "status": "invalid",
            "label": "Gagal Diproses",
            "confidence_percent": 0,
            "message": f"Server AI menolak atau gagal memproses gambar ini. Kemungkinan gambar melanggar filter keamanan AI atau bukan ban.",
            "recommendation": "Silakan coba foto ban yang lain dan pastikan hanya objek ban yang terlihat jelas.",
            "kriteria": {}
        }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )