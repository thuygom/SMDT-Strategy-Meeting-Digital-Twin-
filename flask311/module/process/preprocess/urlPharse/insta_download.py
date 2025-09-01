import yt_dlp
from moviepy.editor import AudioFileClip
import os
import pandas as pd
from urllib.parse import urlparse, parse_qs

# ▶ 설정
EXCEL_PATH = "unique_urls_instagram_692.xlsx"
SAVE_DIR = "D:/insta_downloads"
os.makedirs(SAVE_DIR, exist_ok=True)

# ▶ video ID 추출 함수 (인스타는 ID 없을 수도 있음)
def extract_video_id(url):
    parsed_url = urlparse(url)
    if "youtube.com" in parsed_url.netloc:
        query = parse_qs(parsed_url.query)
        return query.get("v", [None])[0]
    elif "tiktok.com" in parsed_url.netloc or "instagram.com" in parsed_url.netloc:
        return parsed_url.path.strip("/").split("/")[-1]
    else:
        return None

# ▶ URL 목록 읽기
df = pd.read_excel(EXCEL_PATH)
url_list = df.iloc[:, 0].dropna().astype(str).tolist()

# ▶ URL별 다운로드 및 변환
for url in url_list:
    try:
        video_id = extract_video_id(url)
        if not video_id:
            print(f"⚠️ 유효하지 않은 URL: {url}")
            continue

        print(f"\n🎬 다운로드 시도: {url} (ID: {video_id})")

        mp4_path = os.path.join(SAVE_DIR, f"{video_id}.mp4")
        wav_path = os.path.join(SAVE_DIR, f"{video_id}.wav")

        # 이미 wav 변환된 경우 건너뛰기
        if os.path.exists(wav_path):
            print(f"⏭️ 이미 존재하는 파일입니다. 건너뜁니다: {wav_path}")
            continue

        # ▶ yt-dlp 옵션 설정
        ydl_opts = {
            'format': 'bestaudio/best',
            'outtmpl': mp4_path,
            'cookiefile': 'instagram_cookies.txt',
            'quiet': True  # 에러만 표시,
        }

        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([url])
        except Exception as download_error:
            print(f"🚫 다운로드 실패 (릴스 아님 또는 접근 불가): {download_error}")
            continue

        # ▶ moviepy로 wav 변환
        try:
            audio_clip = AudioFileClip(mp4_path)
            audio_clip.write_audiofile(wav_path)
            audio_clip.close()

            print(f"✅ 변환 완료: {wav_path}")

            # 원본 삭제
            os.remove(mp4_path)
            print(f"🗑️ 원본 삭제 완료: {mp4_path}")

        except Exception as convert_error:
            print(f"❌ 변환 실패 ({mp4_path}): {convert_error}")
            if os.path.exists(mp4_path):
                os.remove(mp4_path)  # 실패 시 mp4 삭제
            continue

    except Exception as e:
        print(f"❌ 예기치 못한 오류 발생 ({url}): {e}")
