import pandas as pd
import os

# 디렉토리 경로
dir_path = 'C:/Users/bandl/OneDrive/바탕 화면/youtube_data/instagram_data/influencer/2025-04-30'

# 4번째 열 기준 (0부터 시작 → index 3)
LENGTH_COL_INDEX = 3

for filename in os.listdir(dir_path):
    if filename.endswith('.xlsx'):
        file_path = os.path.join(dir_path, filename)
        print(f"\n📂 처리 중: {filename}")

        try:
            df = pd.read_excel(file_path).reset_index(drop=True)

            # ✅ 4번째 열 이름 가져오기
            group_key = df.columns[LENGTH_COL_INDEX]

            # ✅ 그룹핑 (예: 영상 ID 또는 댓글 수)
            grouped = df.groupby(df[group_key])

            trimmed_blocks = []

            for key, group in grouped:
                if len(group) > 50:
                    print(f"✂️ 그룹 '{key}' 댓글 {len(group)}개 → 50개로 자름")
                    group = group.iloc[:50]
                trimmed_blocks.append(group)

            final_df = pd.concat(trimmed_blocks, ignore_index=True)
            final_df.to_excel(file_path, index=False)
            print(f"✅ 저장 완료: {filename} (총 {len(final_df)}행)")

        except Exception as e:
            print(f"❌ 오류 발생 ({filename}): {e}")
