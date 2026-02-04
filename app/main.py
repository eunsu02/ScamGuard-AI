from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from app.kobert_model_loader import predict_scam_kobert 
from app.model_loader import get_model 
from app.utils import process_youtube_video, get_youtube_text, split_text 
from torchvision import transforms
from PIL import Image
import torch
import os
import numpy as np
from transformers import BertTokenizer, BertForSequenceClassification

app = FastAPI(title="ScamGuard AI API")

# --- 딥페이크 모델 로드 ---
model, device = get_model("models/scamguard_model.pth")
# --- KoBERT 사기 자막 탐지 모델 로드 ---
KO_MODEL_PATH = "models/kobert_model"
ko_tokenizer = BertTokenizer.from_pretrained(KO_MODEL_PATH)
ko_model = BertForSequenceClassification.from_pretrained(KO_MODEL_PATH)
ko_model.to(device)
ko_model.eval()

@app.get("/")
def read_root():
    return {"message": "Scam Guard AI Server is Running!"}

# --- 딥페이크 전처리 설정 ---
transformer = transforms.Compose(
    [
        transforms.ToPILImage(),
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ]
)

# --- 딥페이크 탐지 엔드포인트 ---
@app.post("/deepfake")
async def predict_deepfake_from_url(url: str):
    # 1. 유튜브에서 얼굴 추출
    face_img = process_youtube_video(url)
    if face_img is None:
        raise HTTPException(
            status_code=400, detail="얼굴을 찾을 수 없거나 영상 처리에 실패했습니다."
        )

    # 2. 전처리 및 추론
    input_tensor = transformer(face_img).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(input_tensor)
        prob = torch.nn.functional.softmax(output, dim=1)[0][1].item()

    return {
        "url": url,
        "is_fake": prob > 0.5,
        "confidence": round(prob * 100, 2),
        "message": "🚨 딥페이크 의심" if prob > 0.5 else "✅ 정상 영상",
    }

# --- 자막 사기 탐지 엔드포인트 ---
@app.post(
    "/youtube-scam", 
    tags=["자막 분석"], 
    summary="유튜브 자막 사기 판별",
    description="KoBERT 모델을 사용하여 유튜브 자막을 문장 단위로 분석하고 사기 위험도를 판별합니다."
)
async def analyze_text_scam(
    url: str = Query(..., description="분석할 유튜브 영상 URL", example="https://www.youtube.com/watch?v=ANCwJT3E7ko")
):
    video_id = url.split("v=")[-1].split("&")[0]
    raw_text = get_youtube_text(video_id)
    if not raw_text:
        raise HTTPException(status_code=400, detail="자막을 가져올 수 없는 영상입니다.")

    chunks = split_text(raw_text)
    scam_results = []
    max_prob = 0.0 
    
    for sentence in chunks:
        prob = predict_scam_kobert(sentence)
        
        if prob > max_prob:
            max_prob = prob
        
        if prob >= 0.7:
            scam_results.append({
                "text": sentence,
                "scam_probability": f"{round(prob * 100, 2)}%"
            })

    # 3단계 상태 판별 로직 (위험, 주의, 안전)
    if max_prob >= 0.9:
        final_status = "🚨 위험"
    elif max_prob >= 0.7:
        final_status = "⚠️ 주의"
    else:
        final_status = "✅ 안전"

    return {
        "url": url,
        "total_sentences": len(chunks),
        "highest_probability": f"{round(max_prob * 100, 2)}%",
        "detected_scams": scam_results,
        "status": final_status
    }

# --- 배치 이미지 테스트 엔드포인트 ---
@app.get("/test-batch")
async def test_batch_images():
    # 1. 테스트 이미지 폴더 경로 설정
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    test_dir = os.path.join(base_dir, "test_images")

    if not os.path.exists(test_dir):
        return {"error": "test_images 폴더를 찾을 수 없습니다."}

    results = []
    # 2. 폴더 내 파일들 리스팅 (png, jpg, jpeg만 골라내기)
    image_files = [
        f for f in os.listdir(test_dir) if f.lower().endswith((".png", ".jpg", ".jpeg"))
    ]

    for filename in image_files:
        img_path = os.path.join(test_dir, filename)
        image = Image.open(img_path).convert("RGB")

        # 💡 PIL 이미지를 Numpy 배열로 변환해서 transformer에 전달
        image_np = np.array(image)
        input_tensor = transformer(image_np).unsqueeze(0).to(device)
        with torch.no_grad():
            output = model(input_tensor)
            prob = torch.nn.functional.softmax(output, dim=1)[0][1].item()

        confidence = round(prob * 100, 2)
        results.append(
            {
                "filename": filename,
                "is_fake": confidence > 50,
                "confidence": f"{confidence}%",
                "status": "🚨 딥페이크 의심" if confidence > 50 else "✅ 정상",
            }
        )

    # 4. 전체 결과 반환
    return {"total_count": len(results), "predictions": results}
