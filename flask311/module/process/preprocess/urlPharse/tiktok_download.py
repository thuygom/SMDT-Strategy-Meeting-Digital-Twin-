import yt_dlp
from moviepy.editor import AudioFileClip
import os
import pandas as pd
from urllib.parse import urlparse, parse_qs

# ▶ 설정
EXCEL_PATH = "unique_urls_tiktok_961.xlsx"  # 엑셀 파일 경로
SAVE_DIR = "D:/yt_downloads"                  # 저장 폴더
os.makedirs(SAVE_DIR, exist_ok=True)

# ▶ 유튜브/TikTok URL에서 video ID 추출
def extract_video_id(url):
    parsed_url = urlparse(url)
    if "youtube.com" in parsed_url.netloc:
        query = parse_qs(parsed_url.query)
        return query.get("v", [None])[0]
    elif "tiktok.com" in parsed_url.netloc:
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

        print(f"\n🎬 다운로드 중: {url} (ID: {video_id})")

        # ▶ yt-dlp 옵션 (mp4 저장)
        mp4_path = os.path.join(SAVE_DIR, f"{video_id}.mp4")
        ydl_opts = {
            'format': 'bestvideo+bestaudio/best',
            'outtmpl': mp4_path,
            'merge_output_format': 'mp4'
        }

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])

        # ▶ .wav 경로
        wav_path = os.path.join(SAVE_DIR, f"{video_id}.wav")

        # ▶ moviepy로 wav 변환
        audio_clip = AudioFileClip(mp4_path)
        audio_clip.write_audiofile(wav_path)
        audio_clip.close()

        print(f"✅ 변환 완료: {wav_path}")

        # ▶ 원본 삭제
        os.remove(mp4_path)
        print(f"🗑️ 원본 삭제 완료: {mp4_path}")

    except Exception as e:
        print(f"❌ 오류 발생 ({url}): {e}")
