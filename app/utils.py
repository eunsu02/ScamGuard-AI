import cv2
import yt_dlp
import os
from pathlib import Path
import re
from youtube_transcript_api import YouTubeTranscriptApi

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


# 유튜브 ID를 받아 자막 가져오는 함수
def get_youtube_text(video_id):
    try:
        # 1. API 객체 초기화
        ytt_api = YouTubeTranscriptApi()
        
        # 2. 공식 문서의 fetch 방식 사용
        # 한국어('ko')를 우선 시도하고 없으면 영어('en') 시도
        fetched_transcript = ytt_api.fetch(video_id, languages=['ko', 'en'])
        
        # 3. 텍스트 추출 (snippet.text 또는 snippet['text'] 대응)
        try:
            return " ".join([snippet.text for snippet in fetched_transcript])
        except AttributeError:
            return " ".join([snippet['text'] for snippet in fetched_transcript])
            
    except Exception as e:
        print(f"❌ 자막 가져오기 실패 ({video_id}): {e}")
        return None

# 긴 텍스트를 AI 모델이 읽기 좋게 150자 단위로 쪼개는 함수
def split_text(text, chunk_size=150):
    if not text: return []
    # 공백 제거 및 전처리
    text = re.sub(r'\s+', ' ', str(text)).strip()
    # 150자 단위로 분할 (30자 미만 버림)
    return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size) if len(text[i:i+chunk_size]) > 30]
