# app/main.py
# uvicorn app.main:app --reload
import logging
import traceback

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

from app.model import predict_sentiment
from app.issue import create_github_issue

# 1) 로그 포맷: 시간 | 레벨 | 파일:라인(함수) | 메시지
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | "
           "%(filename)s:%(lineno)d (%(funcName)s) | "
           "%(message)s",
)
logger = logging.getLogger("sentiment")

app = FastAPI(title="Movie Review Sentiment Analyzer")
app.mount("/static", StaticFiles(directory="static"), name="static")


class ReviewRequest(BaseModel):
    text: str


@app.get("/")
async def serve_index():
    return FileResponse("static/index.html")


@app.post("/predict")
async def predict(request: ReviewRequest):
    text = request.text

    # (A) 요청 자체를 기록: 언제 / 무엇(endpoint) / 어떤 입력
    logger.info(f"CALL /predict | text='{text}' | len={len(text)}")

    try:
        # 테스트용 의도적 장애 ('crash' 입력 시 에러)
        if text == "crash":
            raise RuntimeError("의도적 장애 추가")

        result = predict_sentiment(text)

        # (B) 정상 결과도 짧게 기록
        logger.info(
            f"OK /predict | label={result['label']} "
            f"confidence={result['confidence']}"
        )
        return result

    except Exception as e:
        # (C) 디버깅 핵심: 에러 종류/메시지 + 스택트레이스 자동 기록
        logger.exception(
            f"FAIL /predict | text='{text}' | error={type(e).__name__}: {e}"
        )

        # (D) GitHub Issue 자동 생성
        tb = traceback.format_exc()
        title = f"[Prod Error] /predict failed: {type(e).__name__}"
        body = (
            f"## Summary\n"
            f"- endpoint: /predict\n"
            f"- input(text, short): `{text}`\n"
            f"- input length: {len(text)}\n\n"
            f"## Exception\n"
            f"- type: {type(e).__name__}\n"
            f"- message: {str(e)}\n\n"
            f"## Traceback (line info)\n"
            f"```text\n{tb}\n```"
        )
        create_github_issue(title, body, logger)

        # (E) 사용자 응답은 심플하게 (프론트가 confidence를 읽음)
        return {"label": "서버 오류", "confidence": -1}