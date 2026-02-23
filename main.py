import os
import base64
from io import BytesIO

import torch
from PIL import Image
from transformers import LlavaForConditionalGeneration, AutoProcessor
from peft import PeftModel

from fastapi import FastAPI, HTTPException, Header
from pydantic import BaseModel
import asyncio

# -----------------------------
# 0) Security / Runtime Config
# -----------------------------
API_KEY = os.getenv("API_KEY")  # Runpod Env: API_KEY=...
REQUIRE_API_KEY = os.getenv("REQUIRE_API_KEY", "true").lower() in ("1", "true", "yes", "y")

# 데모 안전장치: 동시에 들어오는 요청 제한 (GPU OOM / latency 폭증 방지)
MAX_CONCURRENT = int(os.getenv("MAX_CONCURRENT", "1"))
_sema = asyncio.Semaphore(MAX_CONCURRENT)

# -----------------------------
# 1) Model Load
# -----------------------------
BASE_MODEL_ID = "llava-hf/llava-1.5-7b-hf"
ADAPTER_ID = "igeon510/youtube-fakevlm"

print("Starting to load model... This may take a few minutes.")

model = LlavaForConditionalGeneration.from_pretrained(
    BASE_MODEL_ID,
    torch_dtype=torch.float16,
    device_map="auto",
)
model = PeftModel.from_pretrained(model, ADAPTER_ID)
model.eval()

processor = AutoProcessor.from_pretrained(BASE_MODEL_ID)

print("Model loaded successfully!")

# -----------------------------
# 2) FastAPI
# -----------------------------
app = FastAPI()

class InferenceRequest(BaseModel):
    image_base64: str
    prompt: str

def _check_auth(x_api_key: str | None):
    """
    - REQUIRE_API_KEY=true 이고 API_KEY가 설정되어 있으면, 헤더 x-api-key를 강제
    - REQUIRE_API_KEY=false면 인증 생략 (로컬 디버깅용)
    """
    if not REQUIRE_API_KEY:
        return

    if not API_KEY:
        raise HTTPException(status_code=500, detail="Server misconfigured: API_KEY is not set")

    if not x_api_key or x_api_key != API_KEY:
        raise HTTPException(status_code=403, detail="Forbidden")

@app.post("/inference")
async def generate(
    request: InferenceRequest,
    x_api_key: str | None = Header(default=None),
):
    # 1) 인증
    _check_auth(x_api_key)

    # 2) 동시성 제한 (GPU 보호)
    async with _sema:
        try:
            # 이미지 디코딩
            image_bytes = base64.b64decode(request.image_base64)
            image = Image.open(BytesIO(image_bytes)).convert("RGB")

            # 전처리
            inputs = processor(
                text=request.prompt,
                images=image,
                return_tensors="pt",
            )

            # 텐서를 GPU로 이동
            inputs = {k: v.to(model.device) for k, v in inputs.items()}

            # 픽셀 값 dtype 정합성
            if "pixel_values" in inputs:
                inputs["pixel_values"] = inputs["pixel_values"].to(model.dtype)

            with torch.no_grad():
                output_ids = model.generate(
                    **inputs,
                    max_new_tokens=256,
                    do_sample=False,
                )

            full_text = processor.decode(output_ids[0], skip_special_tokens=True)
            answer = full_text.split("ASSISTANT:")[-1].strip()

            return {"result": answer}

        except Exception as e:
            print(f"Error during inference: {e}")
            raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def health():
    return {"status": "ready"}

@app.get("/healthz")
async def healthz():
    return {"ok": True, "model": BASE_MODEL_ID, "adapter": ADAPTER_ID, "max_concurrent": MAX_CONCURRENT}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)