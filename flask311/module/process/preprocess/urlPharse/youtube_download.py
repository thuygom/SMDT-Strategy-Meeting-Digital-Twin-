from pytubefix import YouTube
from pytubefix.cli import on_progress
from moviepy.editor import AudioFileClip
import os
import pandas as pd
from urllib.parse import urlparse, parse_qs

# ▶ 설정
EXCEL_PATH = "unique_urls_youtube_1392.xlsx"  # 엑셀 파일 경로
SAVE_DIR = "D:/youtube_audio"                 # 저장 폴더
MAX_DURATION_SECONDS = 20 * 60                # 20분 제한
os.makedirs(SAVE_DIR, exist_ok=True)

# ▶ 유튜브 URL에서 video ID 추출 함수
def extract_video_id(url):
    parsed_url = urlparse(url)
    query = parse_qs(parsed_url.query)
    return query.get("v", [None])[0]

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

        wav_path = os.path.join(SAVE_DIR, f"{video_id}.wav")

        # ▶ 이미 .wav 파일이 존재하면 건너뜀
        if os.path.exists(wav_path):
            print(f"⏭️ 이미 변환된 파일이 존재합니다. 건너뜁니다: {wav_path}")
            continue

        yt = YouTube(url, on_progress_callback=on_progress)

        # ▶ 영상 길이 확인
        if yt.length >= MAX_DURATION_SECONDS:
            print(f"⏩ 영상 길이 초과: {yt.length // 60}분 ({video_id}) → 건너뜁니다.")
            continue

        print(f"\n🎬 다운로드 중: {yt.title} (ID: {video_id}) | 길이: {yt.length // 60}분")

        # 오디오 전용 stream 선택
        audio_stream = yt.streams.filter(only_audio=True).first()
        temp_path = audio_stream.download(output_path=SAVE_DIR, filename=video_id + ".temp")

        # wav 변환
        audio_clip = AudioFileClip(temp_path)
        audio_clip.write_audiofile(wav_path)
        audio_clip.close()

        print(f"✅ 변환 완료: {wav_path}")

        # 원본 삭제
        os.remove(temp_path)
        print(f"🗑️ 원본 삭제 완료: {temp_path}")

    except Exception as e:
        print(f"❌ 오류 발생 ({url}): {e}")
