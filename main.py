import base64
from io import BytesIO
import torch
from PIL import Image
from transformers import LlavaForConditionalGeneration, AutoProcessor
from peft import PeftModel
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# --- 1. 초기 설정 및 모델 로드 ---
BASE_MODEL_ID = "llava-hf/llava-1.5-7b-hf"
ADAPTER_ID = "igeon510/llava-1.5-7b-qlora"

print("Starting to load model... This may take a few minutes.")

# Base Model 로드
model = LlavaForConditionalGeneration.from_pretrained(
    BASE_MODEL_ID,
    torch_dtype=torch.float16,
    device_map="auto"
)

# Adapter 로드
model = PeftModel.from_pretrained(model, ADAPTER_ID)
model.eval()

# Processor 로드
processor = AutoProcessor.from_pretrained(BASE_MODEL_ID)

print("Model loaded successfully!")

# --- 2. FastAPI 설정 ---
app = FastAPI()

# 요청 데이터 구조 정의
class InferenceRequest(BaseModel):
    image_base64: str
    prompt: str

# --- 3. API 엔드포인트 ---
@app.post("/inference")
async def generate(request: InferenceRequest):
    try:
        # 이미지 디코딩
        image_bytes = base64.b64decode(request.image_base64)
        image = Image.open(BytesIO(image_bytes)).convert("RGB")

        # 전처리
        inputs = processor(
            text=request.prompt,
            images=image,
            return_tensors="pt"
        )
        
        # 텐서를 GPU로 이동
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        # 픽셀 값 타입 정합성 맞추기
        if "pixel_values" in inputs:
            inputs["pixel_values"] = inputs["pixel_values"].to(model.dtype)

        # 추론 수행
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=128,
                do_sample=False
            )

        # 결과 디코딩
        full_text = processor.decode(output_ids[0], skip_special_tokens=True)
        answer = full_text.split("ASSISTANT:")[-1].strip()

        return {"result": answer}

    except Exception as e:
        print(f"Error during inference: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# 건강 상태 체크용 엔드포인트
@app.get("/")
async def health():
    return {"status": "ready"}

# --- 4. 서버 실행 ---
if __name__ == "__main__":
    import uvicorn
    # 8000번 포트로 서버 시작
    uvicorn.run(app, host="0.0.0.0", port=8000)