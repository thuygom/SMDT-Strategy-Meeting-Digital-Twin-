import os
import pandas as pd
import json
from openai import OpenAI
from tqdm import tqdm

# 🔑 API 키 설정

# 📁 경로 설정
EXCEL_DIR = "C:/Users/bandl/OneDrive/바탕 화면/youtube_data/tiktok_data"
JSON_DIR = "D:/tiktok_downloads"
TXT_DIR = "D:/tiktok_downloads"

# 📄 감성 시퀀스 로드 함수
def load_emotion_sequence(video_id):
    path = os.path.join(JSON_DIR, f"{video_id}_emotion.json")
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None

# 📄 STT 텍스트 로드 함수
def load_transcript_text(video_id):
    path = os.path.join(TXT_DIR, f"{video_id}.txt")
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return f.read().strip()
    return None

# 🤖 OpenAI 호출 함수
def get_preference_score(comment, emotion_seq=None, transcript=None):
    prompt = f"다음은 SNS 영상의 댓글입니다:\n\"{comment}\"\n\n"

    if transcript:
        prompt += f"[해당 영상 발화 내용 요약]\n{transcript}\n\n"

    if emotion_seq:
        prompt += f"[해당 영상 감정 흐름 데이터]\n{emotion_seq}\n\n"

    prompt += "위의 정보를 종합하여, 사용자가 해당 영상을 얼마나 선호할지 -3에서 3 사이의 숫자로 추정해주세요. 숫자 하나만 출력하세요."

    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=10
        )
        score_text = response.choices[0].message.content.strip()
        return float(score_text)
    except Exception as e:
        print(f"❌ OpenAI 오류: {e}")
        return None

# 📊 메인 처리 함수
def process_excel_file(file_path):
    print(f"\n📂 처리 중: {os.path.basename(file_path)}")
    try:
        df = pd.read_excel(file_path)

        # ✅ 이미 score 열이 있고 값이 모두 존재하면 건너뛰기
        if "score" in df.columns and df["score"].notna().all():
            print("⚠️ 이미 score 열이 완전히 채워져 있어 건너뜁니다.")
            return

        scores = []

        for _, row in tqdm(df.iterrows(), total=len(df), desc="🔎 댓글 평가"):
            video_id = row["video_url"]
            comment = row["comment"]

            emotion_seq = load_emotion_sequence(video_id)
            transcript = load_transcript_text(video_id)

            score = get_preference_score(comment, emotion_seq, transcript)
            scores.append(score)

        # ✅ score 열이 이미 있다면 덮어쓰기, 없으면 추가
        if "score" in df.columns:
            df["score"] = scores
        else:
            df.insert(6, "score", scores)

        # 💾 엑셀 덮어쓰기 저장
        df.to_excel(file_path, index=False)
        print(f"✅ 저장 완료: {file_path}")

    except Exception as e:
        print(f"❌ 파일 처리 중 오류 발생: {e}")


# ▶ 전체 디렉토리 순회
if __name__ == "__main__":
    for filename in os.listdir(EXCEL_DIR):
        if filename.endswith(".xlsx"):
            excel_path = os.path.join(EXCEL_DIR, filename)
            process_excel_file(excel_path)
