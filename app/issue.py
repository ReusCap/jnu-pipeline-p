# app/issue.py
import os
import requests


def create_github_issue(title: str, body: str, logger) -> None:
    repo = os.getenv("GH_REPO")     # 예: "MyName/jnu-pipline-p"
    token = os.getenv("GH_TOKEN")   # GitHub에서 발급한 토큰

    if not repo or not token:
        # 환경 변수가 없으면 조용히 건너뜀 (로컬 실습에서 편함)
        logger.warning("GH_REPO/GH_TOKEN not set; skipping GitHub issue creation.")
        return

    url = f"https://api.github.com/repos/{repo}/issues"
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
    }
    payload = {"title": title, "body": body}

    r = requests.post(url, headers=headers, json=payload, timeout=10)
    if r.status_code >= 300:
        logger.warning(f"Failed to create issue: {r.status_code} {r.text[:200]}")