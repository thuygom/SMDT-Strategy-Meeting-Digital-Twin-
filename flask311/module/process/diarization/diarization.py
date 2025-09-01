import os
import torchaudio
import torch
import pandas as pd
from pyannote.audio import Pipeline
from pyannote.audio.pipelines.utils.hook import ProgressHook

# ▶ Hugging Face 액세스 토큰

# ▶ 디렉토리 고정
AUDIO_DIR = "D:/youtube_downloads"

# ▶ pyannote pipeline 로드
pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", use_auth_token=ACCESS_TOKEN)
pipeline.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

# ▶ 전체 디렉토리 처리
def process_directory(directory_path):
    for filename in os.listdir(directory_path):
        if filename.lower().endswith(".wav"):
            wav_path = os.path.join(directory_path, filename)
            print(f"\n🎧 처리 중: {filename}")

            try:
                # ▶ 고정 설정: 화자 수 1명, 이름 "speaker"
                num_speakers = 1
                speaker_names = ["speaker"]

                # ▶ 오디오 로드 및 (channel, time) 보정
                waveform, sample_rate = torchaudio.load(wav_path)

                if waveform.ndim == 1:
                    waveform = waveform.unsqueeze(0)
                elif waveform.ndim == 2 and waveform.shape[0] > waveform.shape[1]:
                    waveform = waveform.T

                # 화자 분리 실행
                with ProgressHook() as hook:
                    diarization = pipeline(
                        {"waveform": waveform, "sample_rate": sample_rate},
                        hook=hook,
                        num_speakers=num_speakers
                    )

                # 결과 파싱
                results = []
                for turn, _, speaker_idx in diarization.itertracks(yield_label=True):
                    speaker_label = speaker_names[0]
                    results.append({
                        "start": round(turn.start, 2),
                        "stop": round(turn.end, 2),
                        "speaker": speaker_label
                    })

                # 엑셀 저장
                out_path = os.path.splitext(wav_path)[0] + "_diarization.xlsx"
                pd.DataFrame(results).to_excel(out_path, index=False)
                print(f"✅ 저장 완료: {out_path}")

            except Exception as e:
                print(f"❌ 오류 발생 - 건너뜀: {filename}")
                print(f"   └─ 이유: {e}")

# ▶ 메인 실행
if __name__ == "__main__":
    process_directory(AUDIO_DIR)
