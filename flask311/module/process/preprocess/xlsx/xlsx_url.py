import os
import pandas as pd
import re

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

                # ✅ video_url에서 video ID만 추출
                if "video_url" in df.columns:
                    df["video_url"] = df["video_url"].apply(lambda x: extract_video_id(x))

                # ▶ 필요한 컬럼만 추출 & 순서 정렬
                ordered_cols = [col for col in COLUMNS_FINAL if col in df.columns]
                df = df[ordered_cols]

                # ▶ 덮어쓰기 저장
                df.to_excel(file_path, index=False)
                print(f"✅ 정리 및 저장 완료: {file_path}")

            except Exception as e:
                print(f"❌ 오류 발생 ({filename}): {e}")

def extract_video_id(url):
    """
    Instagram (post/reel) 및 TikTok URL에서 video ID만 추출.
    """
    if isinstance(url, str):
        # Instagram: /p/ID/ 또는 /reel/ID/
        match_instagram = re.search(r"/(p|reel)/([^/]+)/?", url)
        if match_instagram:
            return match_instagram.group(2)
        
        # TikTok: /video/ID
        match_tiktok = re.search(r"/video/(\d+)", url)
        if match_tiktok:
            return match_tiktok.group(1)

    return url  # URL 형식이 아니면 원본 유지

# ▶ 실행
if __name__ == "__main__":
    target_dir = "C:/Users/bandl/OneDrive/바탕 화면/youtube_data/instagram_data"
    reorder_tiktok_excel_columns(target_dir)
