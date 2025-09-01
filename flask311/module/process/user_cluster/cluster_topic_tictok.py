import pandas as pd
import os
from tqdm import tqdm
from openai import OpenAI
import time
import math

# OpenAI 초기화

# 디렉토리 경로
dir_path = 'C:/Users/bandl/OneDrive/바탕 화면/youtube_data/tiktok_data/influencer'

# 🔍 GPT 배치 분류 함수 (10개씩 묶음)
def classify_batch(comments, sentiments):
    comments_block = "\n".join(
        f"{i+1}. \"{comment}\" [Sentiment: {sentiment}]"
        for i, (comment, sentiment) in enumerate(zip(comments, sentiments))
    )

    prompt = f"""
You are a comment classifier.

For each of the following comments, return:
- Topic: one of the following categories:
  0 사건 / 논란
  1 콘텐츠 평가
  2 유튜버 개인
  3 제품 / 아이템 리뷰
  4 사회 / 시사 이슈
  5 공감 / 감정 공유
  6 정보 / 꿀팁
  7 유머 / 드립
  8 질문 / 피드백
  9 기타 / 미분류

- Cluster: one of these:
  Aggressive, Supportive, Neutral Informative, Sarcastic/Playful, Analytical, Spam/Promotional, Empathetic

Respond in this format only:
[Number]. Topic: [label], Cluster: [label]

Here are the comments:
{comments_block}
"""

    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=300
        )
        lines = response.choices[0].message.content.strip().splitlines()
        topics = []
        clusters = []
        for line in lines:
            if "." in line and "Topic:" in line and "Cluster:" in line:
                parts = line.split("Topic:")[1].split(", Cluster:")
                topic = parts[0].strip()
                cluster = parts[1].strip()
                topics.append(topic)
                clusters.append(cluster)
        return topics, clusters

    except Exception as e:
        print(f"❌ GPT 오류: {e}")
        return ["오류"] * len(comments), ["오류"] * len(comments)

# 🔁 모든 엑셀 파일 처리
for filename in os.listdir(dir_path):
    if filename.endswith('.xlsx'):
        file_path = os.path.join(dir_path, filename)
        print(f"\n📂 처리 중: {filename}")

        try:
            df = pd.read_excel(file_path)

            if 'comment' not in df.columns or '감정' not in df.columns:
                print(f"⚠️ '댓글' 또는 '감정' 열이 없음: {filename}")
                continue

            topic_result = []
            cluster_result = []

            batch_size = 10
            total = math.ceil(len(df) / batch_size)

            for i in tqdm(range(0, len(df), batch_size), desc="🔍 GPT 분류 중"):
                comment_batch = df['comment'].iloc[i:i+batch_size].tolist()
                sentiment_batch = df['감정'].iloc[i:i+batch_size].tolist()
                topics, clusters = classify_batch(comment_batch, sentiment_batch)

                # ✅ index 매칭 오류 방지: 결과 개수 검증
                if len(topics) != len(comment_batch) or len(clusters) != len(comment_batch):
                    print(f"⚠️ 결과 누락 → 채워넣음 (i={i}): topics={len(topics)}, clusters={len(clusters)}, expected={len(comment_batch)}")
                    topics = ["오류"] * len(comment_batch)
                    clusters = ["오류"] * len(comment_batch)

                topic_result.extend(topics)
                cluster_result.extend(clusters)

                time.sleep(0.1)

            # 중복 제거
            if '주제' in df.columns:
                df.drop(columns=['주제'], inplace=True)
            if '군집' in df.columns:
                df.drop(columns=['군집'], inplace=True)

            # ✅ index 개수 일치 여부 최종 체크
            if len(topic_result) != len(df) or len(cluster_result) != len(df):
                raise ValueError(f"❌ 결과 길이 불일치: topic={len(topic_result)}, cluster={len(cluster_result)}, df={len(df)}")

            df.insert(5, '주제', topic_result)
            df.insert(6, '군집', cluster_result)

            df.to_excel(file_path, index=False)
            print(f"✅ 저장 완료: {filename}")

        except Exception as e:
            print(f"❌ 오류 발생 ({filename}): {e}")
