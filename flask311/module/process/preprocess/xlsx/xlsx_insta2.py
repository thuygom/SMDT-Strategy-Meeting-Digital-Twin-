import os
import pandas as pd
import re

# ▶ 주제 번호 → 라벨 매핑
TOPIC_LABELS = {
    0: "사건 / 논란",
    1: "콘텐츠 평가",
    2: "유튜버 개인",
    3: "제품 / 아이템 리뷰",
    4: "사회 / 시사 이슈",
    5: "공감 / 감정 공유",
    6: "정보 / 꿀팁",
    7: "유머 / 드립",
    8: "질문 / 피드백",
    9: "기타 / 미분류"
}

# ▶ 주제 컬럼 정리 함수
def normalize_topic_label(topic):
    if pd.isna(topic):
        return topic
    topic = str(topic).strip()

    # 숫자로 시작하는 경우 (예: "5 공감")
    match = re.match(r"^(\d+)", topic)
    if match:
        idx = int(match.group(1))
        return TOPIC_LABELS.get(idx, topic)

    # 숫자만 있는 경우
    if topic.isdigit():
        return TOPIC_LABELS.get(int(topic), topic)

    # 텍스트만 있는 경우
    if topic in TOPIC_LABELS.values():
        return topic

    return topic  # 변경 못하면 원본 유지

# ▶ 디렉토리 내 엑셀 일괄 처리
def normalize_topic_in_directory(directory):
    for filename in os.listdir(directory):
        if filename.endswith(".xlsx"):
            file_path = os.path.join(directory, filename)
            print(f"\n🔍 처리 중: {filename}")
            try:
                df = pd.read_excel(file_path)

                # ▶ 주제 정리
                if "주제" in df.columns:
                    df["주제"] = df["주제"].apply(normalize_topic_label)

                # ▶ 날짜 형식 yyyy-mm-dd로 고정
                if "date" in df.columns:
                    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date.astype(str)

                # ▶ 저장
                df.to_excel(file_path, index=False)
                print(f"✅ 정리 완료: {file_path}")

            except Exception as e:
                print(f"❌ 오류 발생 ({filename}): {e}")

# ▶ 실행
if __name__ == "__main__":
    normalize_topic_in_directory("C:/Users/bandl/OneDrive/바탕 화면/youtube_data/youtube_data")
