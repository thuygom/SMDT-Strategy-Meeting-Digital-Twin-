import os
import pandas as pd

# ▶ 최종 원하는 컬럼 순서
COLUMNS_FINAL = ["video_url", "comment", "date", "감정", "주제", "군집"]

def reorder_tiktok_excel_columns(directory):
    for filename in os.listdir(directory):
        if filename.endswith(".xlsx"):
            file_path = os.path.join(directory, filename)
            print(f"\n🔍 처리 중: {filename}")
            try:
                df = pd.read_excel(file_path)

                # ✅ 컬럼명 리매핑: video_id → video_url
                if "video_id" in df.columns and "video_url" not in df.columns:
                    df.rename(columns={"video_id": "video_url"}, inplace=True)

                # ✅ 날짜 형식 yyyy-mm-dd로 정제
                if "date" in df.columns:
                    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date.astype(str)

                # ▶ 필요한 컬럼만 추출 & 순서 정렬
                ordered_cols = [col for col in COLUMNS_FINAL if col in df.columns]
                df = df[ordered_cols]

                # ▶ 덮어쓰기 저장
                df.to_excel(file_path, index=False)
                print(f"✅ 정리 및 저장 완료: {file_path}")

            except Exception as e:
                print(f"❌ 오류 발생 ({filename}): {e}")

# ▶ 실행
if __name__ == "__main__":
    target_dir = "C:/Users/bandl/OneDrive/바탕 화면/youtube_data/youtube_data"
    reorder_tiktok_excel_columns(target_dir)
