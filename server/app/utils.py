import cv2
import yt_dlp
import os
from pathlib import Path
import re
from youtube_transcript_api import YouTubeTranscriptApi
import pandas as pd

# 얼굴 탐지기 초기화
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)


def process_youtube_video(url: str):

    ydl_opts = {
        "format": "best[ext=mp4][height<=480]/worst",
        "outtmpl": "temp_video.mp4",
        "quiet": True,
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])

    cap = cv2.VideoCapture("temp_video.mp4")
    fps = cap.get(cv2.CAP_PROP_FPS)

    target_frame = int(fps * 6) if fps > 0 else 180
    cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)

    ret, frame = cap.read()
    cap.release()
    if os.path.exists("temp_video.mp4"):
        os.remove("temp_video.mp4")

    if not ret:
        return None

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4, minSize=(100, 100))

    if len(faces) == 0:
        return None

    (x, y, w, h) = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)[0]

    pad_w = int(w * 0.5)
    pad_h = int(h * 0.5)

    img_h, img_w = frame.shape[:2]
    y1 = max(0, y - pad_h)
    y2 = min(img_h, y + h + pad_h)
    x1 = max(0, x - pad_w)
    x2 = min(img_w, x + w + pad_w)

    face_img = frame[y1:y2, x1:x2]

    face_img_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
    return face_img_rgb

# 유튜브 일반 영상 및 쇼츠 URL에서 11자리 ID를 추출하는 함수
def extract_video_id(url: str):
    regex = r"(?:v=|\/shorts\/|embed\/|youtu.be\/)([a-zA-Z0-9_-]{11})"
    match = re.search(regex, url)
    return match.group(1) if match else None

# 입력값이 URL인 경우 ID만 추출하고, 이미 ID라면 그대로 사용하는 함수
def get_youtube_text(url_or_id):
    video_id = extract_video_id(url_or_id) if "http" in url_or_id else url_or_id
    
    if not video_id:
        print(f"❌ 유효한 비디오 ID를 찾을 수 없습니다: {url_or_id}")
        return None

    try:
        # 1. API 객체 초기화
        ytt_api = YouTubeTranscriptApi()
        
        # 2. list() 메서드로 해당 영상에서 사용 가능한 자막 목록을 가져옴
        transcript_list = ytt_api.list(video_id)
        
        # 3. 한국어(ko) > 영어(en) 자막 순으로 탐색
        # 수동 자막 > 자동 생성 자막 순으로 탐색
        transcript = transcript_list.find_transcript(['ko', 'en'])
        fetched_transcript = transcript.fetch()
        
        # 4. 반환된 FetchedTranscript 객체를 순회하며 텍스트만 결합
        # 각 snippet은 .text 속성을 통해 자막 내용을 제
        return " ".join([snippet.text for snippet in fetched_transcript])
            
    except Exception as e:
        print(f"❌ 자막 추출 실패 (ID: {video_id}): {e}")
        return None

# 긴 텍스트를 AI 모델이 읽기 좋게 150자 단위로 쪼개는 함수
def split_text(text, chunk_size=150):
    if not text: return []
    # 공백 제거 및 전처리
    text = re.sub(r'\s+', ' ', str(text)).strip()
    # 150자 단위로 분할 (30자 미만 버림)
    return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size) if len(text[i:i+chunk_size]) > 30]

BASE_DIR = Path(__file__).resolve().parent.parent
KEYWORD_FILE = BASE_DIR / "high_risk_keywords.csv"

# 고위험 키워드의 가중치 누적하여 합산
def apply_keyword_bias(text, probability):
    detected_items = []
    try:
        if os.path.exists(KEYWORD_FILE):
            # 키워드 설정 파일 로드
            df = pd.read_csv(KEYWORD_FILE)
            # 가중치 합산용 변수 초기화
            total_boost = 0.0

            for _, row in df.iterrows():
                # 개별 키워드 및 가중치 정보 추출
                word = str(row['keyword']).strip()
                weight = float(row['weight'])
                category = str(row['category'])
                description = str(row['description'])
                
                # 문장 내 키워드 포함 여부 검사
                if word in text:
                    # 가중치 누적 합산
                    total_boost += weight
                    # 탐지된 키워드 상세 정보 추가
                    detected_items.append({
                        "keyword": word,
                        "category": category,
                        "description": description
                    })
            
            if total_boost > 0:
                # 최종 확률 계산 및 최대치(0.99) 제한
                probability = min(0.99, probability + total_boost)
                # 터미널 로그 출력
                print(f"🚨 탐지 키워드 목록: {[item['keyword'] for item in detected_items]}")
    except Exception as e:
        # 예외 발생 시 오류 로그 출력
        print(f"⚠️ 가중치 보정 로직 오류: {e}")
            
    # 보정된 확률 및 탐지 상세 내역 반환
    return probability, detected_items