import pandas as pd
import os

# 설정
dir_path = 'C:/Users/bandl/OneDrive/바탕 화면/youtube_data/tiktok_data/influencer/'
DATE_CUTOFF = pd.to_datetime("2025-02-23")

for filename in os.listdir(dir_path):
    if filename.endswith('.xlsx'):
        file_path = os.path.join(dir_path, filename)
        print(f"\n📂 처리 중: {filename}")

        try:
            df = pd.read_excel(file_path)

            if 'date' in df.columns:
                # 1. 날짜 파싱
                df['date'] = pd.to_datetime(df['date'], errors='coerce')

                # 2. 날짜 필터링
                before_len = len(df)
                df = df[df['date'] >= DATE_CUTOFF].reset_index(drop=True)
                after_len = len(df)
                print(f"🧹 날짜 필터링 완료: {before_len}개 → {after_len}개 (2025-02-23 이후만 유지)")

                # 3. 날짜별 그룹핑 후 50개 제한
                grouped = df.groupby(df['date'].dt.date)
                trimmed_blocks = []

                for date_val, group in grouped:
                    if len(group) > 50:
                        print(f"✂️ {date_val} 댓글 {len(group)}개 → 50개로 자름")
                        group = group.iloc[:50]
                    trimmed_blocks.append(group)

                # 4. 병합 및 저장
                final_df = pd.concat(trimmed_blocks, ignore_index=True)
                final_df.to_excel(file_path, index=False)
                print(f"✅ 저장 완료: {filename} (총 {len(final_df)}행)")

            else:
                print(f"⚠️ 'date' 열이 없음 → 필터링 생략")

        except Exception as e:
            print(f"❌ 오류 발생 ({filename}): {e}")
