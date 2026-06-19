# app/config.py
import os

# 추론 모드: "rules"(규칙 기반) 또는 "ml"(학습 모델)
MODEL_MODE = os.getenv("MODEL_MODE", "ml")

# --- MLflow 설정 ---
# 로컬 실습 기본값은 sqlite. 외부 서버(ngrok)는 환경변수로 덮어씀.
#   예) export MLFLOW_TRACKING_URI="https://xxxx.ngrok-free.dev"
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")

# 서비스가 사용할 모델(레지스트리 별칭).
# 롤백 시 @champion ↔ @challenger ↔ /<버전번호> 로 변경
MODEL_URI = os.getenv("MODEL_URI", "models:/movie-sentiment-model@champion")

# --- 데이터 / 아티팩트 경로 구성 요소 ---
TRAIN_FILE_NAME = "train.csv"
TEST_FILE_NAME = "test.csv"
MODEL_NAME = "sentiment_model.joblib"
ARTIFACT_DIR_NAME = "artifacts"
DATA_DIR_NAME = "data"

# (폴백용) 로컬 모델 경로 — ml/artifacts/sentiment_model.joblib
LOCAL_MODEL_PATH = os.path.join("ml", ARTIFACT_DIR_NAME, MODEL_NAME)

# --- drift / 재학습 이슈 임계값 ---
LOW_CONFIDENCE_THRESHOLD = float(os.getenv("LOW_CONFIDENCE_THRESHOLD", "0.65"))
LOW_CONFIDENCE_LIMIT = int(os.getenv("LOW_CONFIDENCE_LIMIT", "5"))

# --- 카나리 배포 설정 ---
CANARY_ENABLED = os.getenv("CANARY_ENABLED", "true").lower() == "true"
CANARY_RATIO = float(os.getenv("CANARY_RATIO", "0.5"))  # challenger 비율(수업용 0.5, 실제는 0.1 정도)
CHAMPION_MODEL_URI = os.getenv("CHAMPION_MODEL_URI", "models:/movie-sentiment-model@champion")
CHALLENGER_MODEL_URI = os.getenv("CHALLENGER_MODEL_URI", "models:/movie-sentiment-model@challenger")