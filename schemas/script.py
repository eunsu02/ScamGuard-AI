from pydantic import BaseModel
from typing import List

# 키워드 상세 정보 규격
class KeywordDetail(BaseModel):
    keyword: str
    category: str
    description: str

# 사기 의심 문장 분석 상세 규격
class ScamDetail(BaseModel):
    text: str
    scam_probability: str
    reason: List[KeywordDetail] # 탐지된 키워드 리스트

# 유튜브 자막 분석 전용 응답 스키마
class YouTubeScamResponse(BaseModel):
    url: str # 영상 주소
    total_sentences: int # 전체 문장 수
    highest_probability: str # 최고 사기 확률
    detected_scams: List[ScamDetail] # 탐지된 사기 문장 목록
    status: str # 종합 위험 상태

# 텍스트 스크립트 분석 전용 응답 스키마
class ScriptScamResponse(BaseModel):
    input_summary: str # 입력 텍스트 요약
    total_sentences: int # 전체 문장 수
    highest_probability: str # 최고 사기 확률
    detected_scams: List[ScamDetail] # 탐지된 사기 문장 목록
    status: str # 종합 위험 상태