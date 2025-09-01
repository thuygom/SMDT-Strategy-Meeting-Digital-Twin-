import os
import torch
import torchaudio
import json
import numpy as np
import pandas as pd
from transformers import Wav2Vec2FeatureExtractor, HubertForSequenceClassification

class HuBERTEmotionByDiarization:
    def __init__(self, model_name="superb/hubert-large-superb-er", device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
        self.model = HubertForSequenceClassification.from_pretrained(model_name).to(self.device)
        self.model.eval()

        self.label_map = {
            0: "neutral",
            1: "calm",
            2: "happy",
            3: "sad",
            4: "angry",
            5: "fearful",
            6: "disgust",
            7: "surprised"
        }

    def analyze_segment(self, waveform, sample_rate):
        if sample_rate != 16000:
            resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
            waveform = resampler(waveform)
            sample_rate = 16000

        inputs = self.feature_extractor(waveform.squeeze().numpy(), sampling_rate=sample_rate, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=1)[0].cpu().numpy()  # 🔥 수정된 부분

        top_label = self.label_map[int(np.argmax(probs))]
        return top_label, probs

    def analyze_file(self, wav_path, diarization_path):
        waveform, sample_rate = torchaudio.load(wav_path)
        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)

        diar_df = pd.read_excel(diarization_path)
        results = []

        for _, row in diar_df.iterrows():
            start_sec = float(row["start"])
            stop_sec = float(row["stop"])
            start_sample = int(start_sec * sample_rate)
            stop_sample = int(stop_sec * sample_rate)

            segment = waveform[:, start_sample:stop_sample]
            if segment.shape[1] < sample_rate:  # 1초 이하 skip
                continue

            emotion, probs = self.analyze_segment(segment, sample_rate)
            results.append({
                "start_sec": round(start_sec, 2),
                "stop_sec": round(stop_sec, 2),
                "emotion": emotion,
                "softmax": {self.label_map[i]: round(float(p), 4) for i, p in enumerate(probs)}
            })

        return results

    def analyze_directory(self, directory_path):
        for filename in os.listdir(directory_path):
            if filename.endswith("_diarization.xlsx"):
                base = filename.replace("_diarization.xlsx", "")
                wav_path = os.path.join(directory_path, base + ".wav")
                diar_path = os.path.join(directory_path, filename)

                if not os.path.exists(wav_path):
                    print(f"❌ WAV 파일 없음: {wav_path}")
                    continue

                print(f"🔍 분석 중: {base}")
                try:
                    results = self.analyze_file(wav_path, diar_path)

                    json_path = os.path.join(directory_path, base + "_emotion.json")
                    with open(json_path, "w", encoding="utf-8") as f:
                        json.dump(results, f, ensure_ascii=False, indent=2)
                    print(f"✅ 저장 완료: {json_path}")
                except Exception as e:
                    print(f"❌ 오류 발생 - 건너뜀: {filename}")
                    print(f"   └─ 이유: {e}")

# ✅ 실행
if __name__ == "__main__":
    analyzer = HuBERTEmotionByDiarization()
    analyzer.analyze_directory("D:/tiktok_downloads")
