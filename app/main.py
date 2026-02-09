from fastapi import FastAPI, File, UploadFile, HTTPException, Query, Body, Depends, Request
from app.kobert_model_loader import predict_scam_kobert 
from app.model_loader import get_model 
from app.utils import process_youtube_video, get_youtube_text, split_text, extract_video_id, apply_keyword_bias
from schemas.script import YouTubeScamResponse, ScriptScamResponse
from torchvision import transforms
from PIL import Image
import torch
import os
import numpy as np
from transformers import BertTokenizer, BertForSequenceClassification
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from sqlalchemy.orm import Session
from .database import engine, Base, get_db
from .models import AnalysisHistory
from fastapi.templating import Jinja2Templates

app = FastAPI(title="ScamGuard AI API")

# 템플릿 설정
templates = Jinja2Templates(directory="template")
# 서버 시작 시 테이블 자동 생성
Base.metadata.create_all(bind=engine)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # 로컬 테스트용 전체 허용
    allow_methods=["*"],
    allow_headers=["*"],
)

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

# --- 최근 분석 기록 10개를 가져오는 엔드포인트 ---
@app.get("/recent-results")
async def get_recent_results(db: Session = Depends(get_db)):
    return db.query(AnalysisHistory).order_by(AnalysisHistory.created_at.desc()).limit(10).all()

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
async def predict_deepfake_from_url(url: str, db: Session = Depends(get_db)):
    # 1. 유튜브에서 얼굴 추출
    face_img = process_youtube_video(url)
    if face_img is None:
        return {
            "url": url,
            "is_fake": False,
            "confidence": 0.0,
            "message": "얼굴을 찾을 수 없습니다."
        }

    # 2. 전처리 및 추론
    input_tensor = transformer(face_img).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(input_tensor)
        prob = torch.nn.functional.softmax(output, dim=1)[0][1].item()

    result = {
        "url": url,
        "is_fake": prob > 0.5,
        "confidence": round(prob * 100, 2),
        "message": "🚨 딥페이크 의심" if prob > 0.5 else "✅ 정상 영상",
    }

    # 분석 완료 시 DB 자동 저장
    history = AnalysisHistory(
        url=url,
        video_id=extract_video_id(url),
        status=result["message"],
        deepfake_prob=result["confidence"],
        is_fake=1 if result["is_fake"] else 0
    )
    db.add(history)
    db.commit()

    return result


# --- 사기 탐지 공통 분석 로직 수행 함수 ---
def run_scam_analysis_logic(chunks):
    scam_results = []
    max_prob = 0.0 
    
    for sentence in chunks:
        # 1. 모델 기반 기초 확률 예측
        prob = predict_scam_kobert(sentence)
        # 2. 키워드 가중치 합산 및 상세 사유(reason) 추출
        prob, detected_info = apply_keyword_bias(sentence, prob)
        
        # 최고 확률값 갱신
        if prob > max_prob:
            max_prob = prob
        
        # 탐지 임계값(0.7) 초과 시 결과 리스트 추가
        if prob >= 0.7:
            scam_results.append({
                "text": sentence,
                "scam_probability": f"{round(prob * 100, 2)}%",
                "reason": detected_info # 탐지된 키워드 정보 포함
            })

    # 최종 위험 상태 판별 (위험, 주의, 안전)
    if max_prob >= 0.9:
        final_status = "🚨 위험"
    elif max_prob >= 0.7:
        final_status = "⚠️ 주의"
    else:
        final_status = "✅ 안전"
        
    return max_prob, scam_results, final_status

# --- 유튜브 영상 기반 사기 탐지 엔드포인트 ---
@app.post(
    "/youtube-scam", 
    response_model=YouTubeScamResponse,
    tags=["자막 분석"], 
    summary="유튜브 자막 사기 판별"
)
async def analyze_text_scam(
    url: str = Query(..., description="분석할 유튜브 영상 URL"),
    db: Session = Depends(get_db)
):
    # 비디오 ID 추출 및 자막 데이터 획득
    video_id = extract_video_id(url)
    if not video_id:
        raise HTTPException(status_code=400, detail="ID 추출 실패")

    raw_text = get_youtube_text(video_id)
    if not raw_text:
        raise HTTPException(status_code=400, detail="자막을 가져올 수 없는 영상입니다.")

    # 자막 분할 및 공통 분석 함수 호출
    chunks = split_text(raw_text)
    max_prob, scam_results, final_status = run_scam_analysis_logic(chunks)

    # 분석 완료 시 DB 자동 저장
    keywords_list = []
    for scam in scam_results:
        for r in scam['reason']:
            keywords_list.append(r['keyword'])

    history = AnalysisHistory(
        url=url,
        video_id=video_id,
        status=final_status,
        scam_prob=round(max_prob * 100, 2),
        keywords=list(set(keywords_list)) # 중복 제거 후 저장
    )
    db.add(history)
    db.commit()

    return {
        "url": url,
        "total_sentences": len(chunks),
        "highest_probability": f"{round(max_prob * 100, 2)}%",
        "detected_scams": scam_results,
        "status": final_status
    }

# --- 자막 기반 사기 탐지 엔드포인트 ---
@app.post(
    "/script-scam", 
    response_model=ScriptScamResponse,
    tags=["자막 분석"], 
    summary="텍스트 스크립트 사기 판별"
)
async def analyze_raw_script(
    script: str = Body(..., description="분석할 자막 또는 대본 텍스트", embed=True)
):
    if not script or len(script.strip()) < 10:
        raise HTTPException(status_code=400, detail="분석할 텍스트가 너무 짧거나 비어있습니다.")

    # 텍스트 분할 및 공통 분석 함수 호출
    chunks = split_text(script)
    max_prob, scam_results, final_status = run_scam_analysis_logic(chunks)

    return {
        "input_summary": script[:50] + "...", 
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

@app.get("/web-analysis", response_class=HTMLResponse)
async def get_web_page(request: Request, url: str = Query(None)):
    return templates.TemplateResponse("web_analysis.html", {"request": request, "url": url})