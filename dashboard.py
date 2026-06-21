# dashboard.py  (프로젝트 루트)
import os
import json

import pandas as pd
import streamlit as st
import gspread
from google.oauth2.service_account import Credentials
from dotenv import load_dotenv

load_dotenv()

SCOPE = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive",
]
GOOGLE_SHEET_NAME = os.getenv("GOOGLE_SHEET_NAME")


@st.cache_resource
def get_spreadsheet():
    sa_json = os.getenv("GOOGLE_SERVICE_ACCOUNT_JSON")
    sa_file = os.getenv("GOOGLE_SERVICE_ACCOUNT_FILE")
    if sa_json:
        creds = Credentials.from_service_account_info(json.loads(sa_json), scopes=SCOPE)
    else:
        creds = Credentials.from_service_account_file(sa_file, scopes=SCOPE)
    client = gspread.authorize(creds)
    return client.open(GOOGLE_SHEET_NAME)


def load_sheet(tab_name):
    ws = get_spreadsheet().worksheet(tab_name)
    return pd.DataFrame(ws.get_all_records())


st.set_page_config(page_title="MLOps Monitoring Dashboard", layout="wide")
st.title("MLOps Monitoring Dashboard")

# --- 1) 예측 로그 ---
st.subheader("운영 지표 (Prediction)")
pred_df = load_sheet("prediction_logs")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Total Requests", len(pred_df))

if len(pred_df) > 0:
    pred_df["confidence"] = pd.to_numeric(pred_df["confidence"], errors="coerce")
    c2.metric("Average Confidence", round(pred_df["confidence"].mean(), 4))
    c3.metric("Low Confidence (<0.65)", int((pred_df["confidence"] < 0.65).sum()))

    if "serving_model" in pred_df.columns:
        c4.metric("Canary (challenger)", int((pred_df["serving_model"] == "challenger").sum()))

    st.subheader("Confidence Trend")
    st.line_chart(pred_df.reset_index(), x="index", y="confidence")

    if "serving_model" in pred_df.columns:
        st.subheader("Serving Model Count")
        st.bar_chart(pred_df["serving_model"].value_counts())

    st.subheader("Recent Predictions")
    st.dataframe(pred_df.tail(20), use_container_width=True)

# --- 2) 피드백 로그 ---
st.subheader("사용자 피드백 (Feedback)")
feedback_df = load_sheet("feedback_logs")

f1, f2, f3 = st.columns(3)
f1.metric("Feedback Count", len(feedback_df))

if len(feedback_df) > 0:
    wrong_df = feedback_df[feedback_df["prediction"] != feedback_df["correct_label"]]
    f2.metric("Wrong Prediction Feedback", len(wrong_df))
    rate = len(wrong_df) / len(feedback_df) if len(feedback_df) else 0
    f3.metric("Wrong Feedback Rate", f"{rate:.2%}")

    st.subheader("Feedback Label Distribution")
    st.bar_chart(feedback_df["correct_label"].value_counts())

    st.subheader("Recent Feedback")
    st.dataframe(feedback_df.tail(20), use_container_width=True)