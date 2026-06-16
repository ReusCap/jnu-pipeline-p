# Dockerfile
FROM python:3.12-slim

WORKDIR /app

# 의존성 먼저 설치(캐시 활용)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 소스 / 정적 / ML 코드 복사
COPY ./app ./app
COPY ./static ./static
COPY ./ml ./ml

# 빌드 시 모델 학습 (아티팩트는 git에 없으므로 여기서 생성)
# 반드시 모듈 형태로 실행해야 app.config import가 동작
RUN python -m ml.train

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]