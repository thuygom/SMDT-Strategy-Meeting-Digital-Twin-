import os
import wave
import io
from pydub import AudioSegment
from google.cloud import speech_v1p1beta1 as speech

# Google Cloud 인증 키 경로
API_KEY_PATH = '../apiKey/myKey.json'

# 설정
AUDIO_DIR = "D:/tiktok_downloads"
MAX_DURATION_SECONDS = 120
CHUNK_MS = 30000  # 30초

def get_sample_rate(path):
    with wave.open(path, 'rb') as wf:
        return wf.getframerate()

def get_duration(path):
    with wave.open(path, 'rb') as wf:
        frames = wf.getnframes()
        rate = wf.getframerate()
        return frames / float(rate)

def prepare_audio(file_path):
    audio = AudioSegment.from_file(file_path)
    return audio.set_channels(1).set_sample_width(2)

def transcribe_chunk(chunk, sample_rate, client):
    with io.BytesIO() as buf:
        chunk.export(buf, format="wav")
        buf.seek(0)
        content = buf.read()

    audio = speech.RecognitionAudio(content=content)
    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=sample_rate,
        language_code="ko-KR"
    )

    operation = client.long_running_recognize(config=config, audio=audio)
    response = operation.result(timeout=90)

    return [r.alternatives[0].transcript for r in response.results]

def transcribe_file(wav_path, client):
    txt_path = os.path.splitext(wav_path)[0] + ".txt"

    # 이미 텍스트 파일이 존재하면 생략
    if os.path.exists(txt_path):
        print(f"⏭️ {os.path.basename(txt_path)} 이미 존재 - 건너뜁니다.")
        return

    # 2분 초과 파일은 생략
    duration = get_duration(wav_path)
    if duration > MAX_DURATION_SECONDS:
        print(f"⏩ {os.path.basename(wav_path)}: {duration:.1f}초 (2분 초과) - 건너뜁니다.")
        return

    print(f"🎧 STT 시작: {os.path.basename(wav_path)} ({duration:.1f}초)")

    audio = prepare_audio(wav_path)
    sample_rate = get_sample_rate(wav_path)

    all_transcripts = []
    for i in range(0, len(audio), CHUNK_MS):
        chunk = audio[i:i + CHUNK_MS]
        print(f"  🎙️ 청크 {i // CHUNK_MS + 1} STT 중...")
        try:
            texts = transcribe_chunk(chunk, sample_rate, client)
            all_transcripts.extend(texts)
        except Exception as e:
            print(f"  ⚠️ 오류 발생 (chunk {i // CHUNK_MS + 1}): {e}")

    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(all_transcripts))
    print(f"✅ 저장 완료: {txt_path}")

def main():
    client = speech.SpeechClient.from_service_account_file(API_KEY_PATH)

    for fname in os.listdir(AUDIO_DIR):
        if fname.lower().endswith(".wav"):
            fpath = os.path.join(AUDIO_DIR, fname)
            try:
                transcribe_file(fpath, client)
            except Exception as e:
                print(f"❌ {fname} 처리 실패: {e}")

if __name__ == "__main__":
    main()
