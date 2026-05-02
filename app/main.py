# app/main.py
# uvicorn app.main:app --reload
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from app.model import predict_sentiment

app = FastAPI(title="Movie Review Sentiment Analyzer")

# 정적 파일 마운트 (HTML, CSS, JS 등)
app.mount("/static", StaticFiles(directory="static"), name="static")

# 요청 데이터를 검증할 Pydantic 모델
class ReviewRequest(BaseModel):
    text: str

# 메인 페이지 라우팅
@app.get("/")
async def serve_index():
    return FileResponse("static/index.html")

# 감정 분석 API 엔드포인트
@app.post("/predict")
async def predict(request: ReviewRequest):
    result = predict_sentiment(request.text)
    return result