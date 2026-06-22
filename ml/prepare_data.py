# ml/prepare_data.py
"""NSMC(Naver Sentiment Movie Corpus)를 내려받아
ml/data/train.csv, test.csv 를 생성하는 '일회성' 준비 스크립트.

한 번 실행해 CSV를 만든 뒤 결과 파일을 커밋한다.
→ CI/Docker 빌드가 네트워크에 의존하지 않게 하기 위함.

NSMC 라이선스: CC0 (퍼블릭 도메인) — 재배포 가능.
실행:  python -m ml.prepare_data
"""
import os
import urllib.request
import pandas as pd

BASE_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(BASE_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)

URLS = {
    "train": "https://raw.githubusercontent.com/e9t/nsmc/master/ratings_train.txt",
    "test": "https://raw.githubusercontent.com/e9t/nsmc/master/ratings_test.txt",
}
SAMPLE_TOTAL = {"train": 20000, "test": 5000}  # 라벨당 절반씩 균형 샘플링
LABEL_MAP = {1: "긍정", 0: "부정"}
SEED = 42


def build(split: str):
    raw_path = os.path.join(DATA_DIR, f"nsmc_{split}_raw.txt")
    urllib.request.urlretrieve(URLS[split], raw_path)

    df = pd.read_csv(raw_path, sep="\t").dropna(subset=["document"])
    df["label"] = df["label"].map(LABEL_MAP)

    # 라벨 균형 샘플링(긍정/부정 동수) 후 셔플
    per_class = SAMPLE_TOTAL[split] // 2
    df = df.groupby("label", group_keys=False).sample(n=per_class, random_state=SEED)
    df = df.sample(frac=1, random_state=SEED)

    out = df[["document", "label"]].rename(columns={"document": "text"})
    out_path = os.path.join(DATA_DIR, f"{split}.csv")
    out.to_csv(out_path, index=False, encoding="utf-8")
    os.remove(raw_path)
    print(f"[{split}] {len(out)}행 저장 → {out_path} | 분포 {out['label'].value_counts().to_dict()}")


if __name__ == "__main__":
    for s in ["train", "test"]:
        build(s)
