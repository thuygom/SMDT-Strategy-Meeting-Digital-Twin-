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
                # 문자열을 datetime으로 변환
                df['date'] = pd.to_datetime(df['date'], errors='coerce')

                before_len = len(df)
                df = df[df['date'] >= DATE_CUTOFF].reset_index(drop=True)
                after_len = len(df)
                print(f"🧹 날짜 필터링 완료: {before_len}개 → {after_len}개 (2025-02-23 이후만 유지)")

                # 저장
                df.to_excel(file_path, index=False)
                print(f"✅ 저장 완료: {filename}")
            else:
                print(f"⚠️ 'date' 열이 없음 → 필터링 생략")

        except Exception as e:
            print(f"❌ 오류 발생 ({filename}): {e}")
