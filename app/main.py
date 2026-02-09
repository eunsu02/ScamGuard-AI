from fastapi import FastAPI, File, UploadFile, HTTPException, Query, Body
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

app = FastAPI(title="ScamGuard AI API")

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

    return {
        "url": url,
        "is_fake": prob > 0.5,
        "confidence": round(prob * 100, 2),
        "message": "🚨 딥페이크 의심" if prob > 0.5 else "✅ 정상 영상",
    }


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
    url: str = Query(..., description="분석할 유튜브 영상 URL")
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
async def get_web_page(url: str = Query(None, description="유튜브 URL")):
    return """
    <!DOCTYPE html>
    <html lang="ko">
    <head>
        <meta charset="UTF-8">
        <title>ScamGuard AI - Deepfake Lab</title>
        <style>
            body { margin: 0; padding: 0; font-family: -apple-system, system-ui, sans-serif; background-color: #ffffff; color: #1d1d1f; display: flex; align-items: center; justify-content: center; min-height: 100vh; }
            .container { width: 100%; max-width: 480px; padding: 20px; }
            h1 { font-size: 28px; font-weight: 700; margin: 0 0 10px 0; letter-spacing: -0.5px; }
            p { font-size: 15px; color: #86868b; margin-bottom: 40px; }
            #thumbContainer { width: 100%; aspect-ratio: 16/9; background: #f5f5f7; border-radius: 14px; margin-bottom: 24px; overflow: hidden; display: none; }
            #previewImg { width: 100%; height: 100%; object-fit: cover; }
            input { width: 100%; padding: 18px; box-sizing: border-box; border: none; background: #f5f5f7; border-radius: 12px; font-size: 15px; margin-bottom: 15px; }
            input:focus { background: #e8e8ed; outline: none; }
            button { width: 100%; padding: 18px; border: none; background: #000; color: #fff; border-radius: 12px; font-size: 15px; font-weight: 600; cursor: pointer; }
            button:hover { opacity: 0.8; }
            button:disabled { background: #d2d2d7; cursor: not-allowed; }
            #statusArea { margin-top: 40px; display: none; }
            .bar-bg { width: 100%; height: 2px; background: #f5f5f7; margin-bottom: 10px; }
            #bar { width: 0%; height: 100%; background: #000; transition: width 0.3s; }
            #log { font-size: 12px; text-align: center; color: #86868b; }
            #resultCard { margin-top: 30px; padding: 25px; border-radius: 15px; display: none; text-align: center; }
            .safe { background: #f5f5f7; color: #1d1d1f; }
            .danger { background: #fff2f2; color: #ff3b30; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Deepfake Lab</h1>
            <p>정밀 프레임 분석 대시보드</p>
            <div id="thumbContainer"><img id="previewImg" src=""></div>
            <input type="text" id="urlInput" placeholder="유튜브 링크를 입력하세요" oninput="updateThumb()">
            <button onclick="start()" id="btn">분석 시작</button>
            <div id="statusArea">
                <div class="bar-bg"><div id="bar"></div></div>
                <div id="log">READY</div>
                <div id="resultCard">
                    <div id="resTitle" style="font-size: 18px; font-weight: 700;"></div>
                    <div id="resConf" style="font-size: 13px; margin-top: 5px; opacity: 0.6;"></div>
                </div>
            </div>
        </div>
        <script>
            function updateThumb() {
                const url = document.getElementById('urlInput').value;
                const reg = /^.*((youtu.be\/)|(v\/)|(\/u\/\w\/)|(embed\/)|(watch\?))\??v?=?([^#&?]*).*/;
                const match = url.match(reg);
                const id = (match && match[7].length == 11) ? match[7] : false;
                const container = document.getElementById('thumbContainer');
                if(id) {
                    document.getElementById('previewImg').src = `https://img.youtube.com/vi/${id}/maxresdefault.jpg`;
                    container.style.display = 'block';
                } else { container.style.display = 'none'; }
            }
            window.onload = () => {
                const url = new URLSearchParams(window.location.search).get('url');
                if(url) { document.getElementById('urlInput').value = url; updateThumb(); }
            };
            async function start() {
                const url = document.getElementById('urlInput').value;
                const btn = document.getElementById('btn');
                const bar = document.getElementById('bar');
                const log = document.getElementById('log');
                const resCard = document.getElementById('resultCard');
                btn.disabled = true;
                document.getElementById('statusArea').style.display = 'block';
                resCard.style.display = 'none';
                let p = 0;
                const inv = setInterval(() => { p = Math.min(p + 2, 95); bar.style.width = p + '%'; log.innerText = 'ANALYZING... ' + Math.floor(p) + '%'; }, 500);
                try {
                    const r = await fetch(`/deepfake?url=${encodeURIComponent(url)}`, { method: 'POST' });
                    const d = await r.json();
                    clearInterval(inv);
                    bar.style.width = '100%';
                    log.innerText = 'COMPLETE';
                    resCard.style.display = 'block';
                    resCard.className = d.is_fake ? 'danger' : 'safe';
                    document.getElementById('resTitle').innerText = d.message;
                    document.getElementById('resConf').innerText = 'CONFIDENCE: ' + d.confidence + '%';
                } catch(e) { log.innerText = 'ERROR'; } finally { btn.disabled = false; }
            }
        </script>
    </body>
    </html>
    """