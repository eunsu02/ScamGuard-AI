from sqlalchemy import Column, Integer, String, Float, DateTime, JSON
from .database import Base
from datetime import datetime

class AnalysisHistory(Base):
    __tablename__ = "analysis_history"

    id = Column(Integer, primary_key=True, index=True)
    url = Column(String)            # 유튜브 URL
    video_id = Column(String)       # 영상 ID
    status = Column(String)         # 위험, 주의, 안전
    scam_prob = Column(Float)      # 사기 확률
    deepfake_prob = Column(Float)  # 딥페이크 확률
    is_fake = Column(Integer)       # 딥페이크 여부 (0 or 1)
    keywords = Column(JSON)         # 탐지된 키워드 배열
    created_at = Column(DateTime, default=datetime.now)