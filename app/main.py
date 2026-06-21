# app/main.py
# uvicorn app.main:app --reload
import logging
import traceback

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

from app.model import predict_sentiment, predict_sentiment_ml, predict_sentiment_ml_canary
from app.issue import create_github_issue
from app.config import MODEL_MODE, LOW_CONFIDENCE_THRESHOLD
from app.model_loader import get_model_info
from app.retrain_issue import update_issue_state
from app.prediction_logger import save_prediction_log
from app.feedback import save_feedback

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


class FeedbackRequest(BaseModel):
    text: str
    prediction: str
    correct_label: str
    confidence: float = 0.0
    serving_model: str = "unknown"


@app.get("/")
async def serve_index():
    return FileResponse("static/index.html")


@app.post("/predict")
async def predict(request: ReviewRequest):
    text = request.text
    logger.info(f"CALL /predict | text='{text}' | len={len(text)}")

    try:
        if text == "crash":
            raise RuntimeError("의도적 장애 추가")

        if MODEL_MODE == "ml":
            result = predict_sentiment_ml_canary(text)
            update_issue_state(
                text, result["label"], result["confidence"], LOW_CONFIDENCE_THRESHOLD
            )
            result["model_info"] = get_model_info(result["serving_model"])
            save_prediction_log(
                text, result["label"], result["confidence"], result["serving_model"]
            )
        else:
            result = predict_sentiment(text)

        logger.info(
            f"OK /predict | label={result['label']} confidence={result['confidence']}"
        )
        return result

    except Exception as e:
        logger.exception(
            f"FAIL /predict | text='{text}' | error={type(e).__name__}: {e}"
        )
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
        return {"label": "서버 오류", "confidence": -1}


@app.post("/feedback")
async def feedback(payload: FeedbackRequest):
    logger.info(
        f"CALL /feedback | prediction={payload.prediction} correct={payload.correct_label}"
    )
    try:
        save_feedback(
            payload.text,
            payload.prediction,
            payload.correct_label,
            payload.confidence,
            payload.serving_model,
        )
        logger.info("OK /feedback | feedback saved")
        return {"status": "feedback saved"}
    except Exception as e:
        logger.exception(f"FAIL /feedback | error={type(e).__name__}: {e}")
        return {"status": "feedback save failed"}