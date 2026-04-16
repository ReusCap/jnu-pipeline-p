# Dockerfile

# 1. 가볍고 안정적인 파이썬 공식 이미지 사용
FROM python:3.10-slim

# 2. 작업 디렉토리 설정
WORKDIR /app

# 3. 의존성 파일 복사 및 설치 (캐시 활용을 위해 먼저 복사)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 4. 소스 코드 및 정적 파일 복사
COPY ./app ./app
COPY ./static ./static

# 5. FastAPI 실행을 위한 포트 노출
EXPOSE 8000

# 6. 서버 실행 명령어
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]