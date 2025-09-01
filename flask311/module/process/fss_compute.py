import os
import pandas as pd
import json
import random  # ✅ 추가

# 🔧 경로 설정
EXCEL_DIR = "C:/Users/bandl/OneDrive/바탕 화면/youtube_data/youtube_data"
TOPIC_PATH = "C:/Users/bandl/OneDrive/바탕 화면/youtube_data/topic_affinity_labels.json"
CLUSTER_PATH = "C:/Users/bandl/OneDrive/바탕 화면/youtube_data/cluster_reliability.json"

# 📥 정량화 JSON 로드
with open(TOPIC_PATH, "r", encoding="utf-8") as f:
    topic_map = json.load(f)

with open(CLUSTER_PATH, "r", encoding="utf-8") as f:
    cluster_map = json.load(f)

# 📐 score 정규화 함수 (-3 ~ 3 → 0 ~ 100)
def normalize_score(score):
    if pd.isna(score):
        score = random.uniform(-3, 3)  # ✅ 비어있으면 랜덤 점수 생성
    return round(((score + 3) / 6) * 100, 2)

# 🧮 FSS 계산 함수
def compute_fss(row):
    topic = row["주제"]
    cluster = row["군집"]
    score = row["score"]

    if pd.isna(topic) or pd.isna(cluster):
        return None

    topic_val = topic_map.get(str(topic), 0)
    cluster_val = cluster_map.get(str(cluster), 0)
    scaled_score = normalize_score(score)

    return round(scaled_score * topic_val * cluster_val, 2)

# 🔁 전체 엑셀 파일 처리
for filename in os.listdir(EXCEL_DIR):
    if filename.endswith(".xlsx"):
        file_path = os.path.join(EXCEL_DIR, filename)
        print(f"\n📂 처리 중: {filename}")

        try:
            df = pd.read_excel(file_path)

            # ✅ 필수 컬럼 확인
            required_cols = ["주제", "군집", "score"]
            if not all(col in df.columns for col in required_cols):
                print("⚠️ 필수 컬럼 누락 → 건너뜀")
                continue

            # ✅ FSS 계산 및 추가
            df["FSS"] = df.apply(compute_fss, axis=1)

            # 💾 덮어쓰기 저장
            df.to_excel(file_path, index=False)
            print(f"✅ FSS 추가 및 저장 완료: {file_path}")

        except Exception as e:
            print(f"❌ 오류 발생: {e}")
