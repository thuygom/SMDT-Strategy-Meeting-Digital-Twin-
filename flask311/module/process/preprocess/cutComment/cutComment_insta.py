import pandas as pd
import os

# 설정
dir_path = 'C:/Users/bandl/OneDrive/바탕 화면/youtube_data/instagram_data/influencer/2025-03-21'
LENGTH_COL_INDEX = 3  # 4번째 열

for filename in os.listdir(dir_path):
    if filename.endswith('.xlsx'):
        file_path = os.path.join(dir_path, filename)
        print(f"\n📂 처리 중: {filename}")

        try:
            df = pd.read_excel(file_path).reset_index(drop=True)

            # 1. 영상 시작점 찾기 (4번째 열에 숫자가 있는 행들)
            start_indices = []
            for idx in range(len(df)):
                val = df.iloc[idx, LENGTH_COL_INDEX]
                if pd.notnull(val) and isinstance(val, (int, float)):
                    start_indices.append(idx)

            if not start_indices:
                print("⚠️ 영상 시작 인덱스를 찾을 수 없습니다.")
                continue

            # 2. 블록 단위 자르기
            blocks = []
            for i in range(len(start_indices)):
                start = start_indices[i]
                end = start_indices[i + 1] if i + 1 < len(start_indices) else len(df)

                block = df.iloc[start:end]
                if len(block) > 50:
                    print(f"✂️ {start+1}~{end}행 중 50개로 잘림")
                    block = block.iloc[:50]

                blocks.append(block)

            # 3. 병합 및 저장
            final_df = pd.concat(blocks, ignore_index=True)
            final_df.to_excel(file_path, index=False)
            print(f"✅ 저장 완료: {filename} (총 {len(final_df)}행)")

        except Exception as e:
            print(f"❌ 오류 발생 ({filename}): {e}")
