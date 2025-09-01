
from transformers import Wav2Vec2FeatureExtractor, HubertForSequenceClassification
import torchaudio
import torch

class HuBERTEmotionClassifier:
    def __init__(self, model_name="superb/hubert-large-superb-er"):
        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
        self.model = HubertForSequenceClassification.from_pretrained(model_name)
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

    def predict_emotion(self, audio_path):
        waveform, sample_rate = torchaudio.load(audio_path)
        waveform = waveform.mean(dim=0, keepdim=True)  # Convert stereo to mono if needed
        if sample_rate != 16000:
            waveform = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)(waveform)

        inputs = self.feature_extractor(waveform.squeeze().numpy(), sampling_rate=16000, return_tensors="pt")
        with torch.no_grad():
            outputs = self.model(**inputs)
        pred = torch.argmax(outputs.logits, dim=1).item()
        return self.label_map[pred]

# Example usage
if __name__ == "__main__":
    classifier = HuBERTEmotionClassifier()
    emotion = classifier.predict_emotion("example.wav")
    print("예측된 감정:", emotion)
