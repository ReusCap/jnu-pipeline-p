# ml/train.py
import os
import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from app.config import (
    MLFLOW_TRACKING_URI,
    TRAIN_FILE_NAME,
    TEST_FILE_NAME,
    MODEL_NAME,
    ARTIFACT_DIR_NAME,
    DATA_DIR_NAME,
)

BASE_DIR = os.path.dirname(__file__)
TRAIN_DATA_PATH = os.path.join(BASE_DIR, DATA_DIR_NAME, TRAIN_FILE_NAME)
TEST_DATA_PATH = os.path.join(BASE_DIR, DATA_DIR_NAME, TEST_FILE_NAME)
ARTIFACT_DIR = os.path.join(BASE_DIR, ARTIFACT_DIR_NAME)
MODEL_PATH = os.path.join(ARTIFACT_DIR, MODEL_NAME)

os.makedirs(ARTIFACT_DIR, exist_ok=True)

# 실험 세팅: 로컬 sqlite 또는 외부 서버(ngrok). 환경변수로 제어.
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_registry_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment("movie-sentiment-local")

train_df = pd.read_csv(TRAIN_DATA_PATH)
test_df = pd.read_csv(TEST_DATA_PATH)
X_train, y_train = train_df["text"], train_df["label"]
X_test, y_test = test_df["text"], test_df["label"]

with mlflow.start_run():
    mlflow.log_param("train_data_path", TRAIN_DATA_PATH)
    mlflow.log_param("test_data_path", TEST_DATA_PATH)
    mlflow.log_param("train_row_count", len(train_df))
    mlflow.log_param("test_row_count", len(test_df))

    pipeline = Pipeline([
        ("vectorizer", CountVectorizer()),
        ("classifier", LogisticRegression(max_iter=200)),
    ])
    pipeline.fit(X_train, y_train)

    train_acc = accuracy_score(y_train, pipeline.predict(X_train))
    test_acc = accuracy_score(y_test, pipeline.predict(X_test))
    mlflow.log_metric("train_accuracy", train_acc)
    mlflow.log_metric("test_accuracy", test_acc)

    joblib.dump(pipeline, MODEL_PATH)

    mlflow.log_artifact(TRAIN_DATA_PATH)
    mlflow.log_artifact(TEST_DATA_PATH)
    mlflow.log_artifact(MODEL_PATH)

    # MLflow 모델 형식으로 등록
    mlflow.sklearn.log_model(
        pipeline, name="model", registered_model_name="movie-sentiment-model"
    )

    print(f"Model saved to: {MODEL_PATH}")
    print(f"train_accuracy: {train_acc:.4f}")
    print(f"test_accuracy: {test_acc:.4f}")