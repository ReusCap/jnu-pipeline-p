# 🎬 영화 리뷰 감정 분석기 (Movie Review Sentiment Analyzer)

DevOps/MLOps 실습 과목 솔로 프로젝트.
수업 예제(스팸 체크기)를 ML 기반 감정 분석기로 업그레이드하면서, **CI/CD부터 모델 서빙까지 전체 파이프라인을 직접 구축**하는 것이 목표.

> 핵심 전략: **Pipeline-First** — 모델 정확도보다 DevOps/MLOps 인프라 완성도에 집중.

---

## 🚀 실행 방법

### 로컬 실행 (개발용)

```bash
# 1. 가상환경 활성화 (예: conda)
conda activate pipeenv

# 2. 의존성 설치
pip install -r requirements.txt

# 3. 서버 실행
uvicorn app.main:app --reload

# 브라우저에서 http://localhost:8000 접속
```

### 테스트 실행

```bash
python -m pytest tests/
```

### Docker 실행

```bash
docker build -t movie-sentiment .
docker run -p 8000:8000 movie-sentiment
```

---

## 📂 폴더 구조

```
.
├── app/
│   ├── main.py          # FastAPI 진입점 (라우팅)
│   └── model.py         # 예측 로직 (중간: 규칙 기반 / 기말: ML 모델)
├── static/
│   └── index.html       # Tailwind CSS 기반 프론트
├── tests/
│   └── test_model.py    # pytest 단위 테스트
├── .github/workflows/
│   └── ci.yaml          # GitHub Actions CI 파이프라인
├── Dockerfile
├── requirements.txt
└── README.md
```

---

## 🧰 기술 스택

| 영역 | 기술 |
|------|------|
| 백엔드 | FastAPI + Pydantic |
| 프론트 | HTML + Tailwind CSS (CDN) |
| 테스트 | pytest + httpx (TestClient) |
| 컨테이너 | Docker |
| CI/CD | GitHub Actions |
| ML (기말) | scikit-learn, MLFlow, joblib |
| 데이터 (기말) | NSMC (Naver Sentiment Movie Corpus) |
| 배포 (기말) | Render |

---

## 🗺️ 개발 로드맵

### Phase 1 — 중간 보고서 (~8주차)
- [x] FastAPI 서버 + 규칙 기반 `model.py`
- [x] Tailwind CSS UI
- [x] pytest 단위 테스트
- [x] Dockerfile
- [x] GitHub Actions CI
- [ ] Render 배포
