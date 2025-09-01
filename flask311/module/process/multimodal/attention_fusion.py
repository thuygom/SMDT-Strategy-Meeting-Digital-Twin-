import torch
import torch.nn.functional as F
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    AutoFeatureExtractor, AutoModelForImageClassification,
    Wav2Vec2FeatureExtractor, HubertForSequenceClassification
)
from PIL import Image
import torchaudio

# --- 텍스트 임베딩 (KcBERT fine-tuned) ---
class TextEmbeddingEncoder:
    def __init__(self, model_path="D:/kcbert-emotion-finetuned"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.model.eval()

    def encode(self, text):
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)
        embedding = outputs.hidden_states[-1].mean(dim=1).squeeze(0)  # [768]
        return embedding

# --- 오디오 임베딩 (HuBERT base → 768) ---
class AudioEmbeddingEncoder:
    def __init__(self, model_name="superb/hubert-base-superb-er"):  # ✅ base로 변경
        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
        self.model = HubertForSequenceClassification.from_pretrained(model_name)
        self.model.eval()

    def encode(self, audio_path):
        waveform, sample_rate = torchaudio.load(audio_path)
        waveform = waveform.mean(dim=0, keepdim=True)
        if sample_rate != 16000:
            waveform = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)(waveform)
        inputs = self.feature_extractor(waveform.squeeze().numpy(), sampling_rate=16000, return_tensors="pt")
        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)
        embedding = outputs.hidden_states[-1].mean(dim=1).squeeze(0)  # ✅ [768]
        return embedding

# --- 이미지 임베딩 (ViT) ---
class ImageEmbeddingEncoder:
    def __init__(self, model_name="trpakov/vit-face-expression"):
        self.extractor = AutoFeatureExtractor.from_pretrained(model_name)
        self.model = AutoModelForImageClassification.from_pretrained(model_name)
        self.model.eval()

    def encode(self, image_path):
        image = Image.open(image_path).convert("RGB")
        inputs = self.extractor(images=image, return_tensors="pt")
        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)
        embedding = outputs.hidden_states[-1].mean(dim=1).squeeze(0)  # [768]
        return embedding

# --- Attention Fusion ---
def attention_fusion(vectors, query_idx=0):
    print(f"[Fusion] ▶️ 입력 벡터 수: {len(vectors)}")
    for i, v in enumerate(vectors):
        print(f"[Fusion] 🔹 벡터 {i} 크기: {v.shape}")

    stacked = torch.stack(vectors)             # [n, 768]
    print(f"[Fusion] ✅ stacked shape: {stacked.shape}")

    Q = vectors[query_idx].unsqueeze(0)        # [1, 768]
    print(f"[Fusion] ✅ Q shape: {Q.shape}")

    attn_score = torch.matmul(Q, stacked.T) / Q.shape[-1] ** 0.5  # [1, n]
    print(f"[Fusion] ✅ Attention score shape: {attn_score.shape}")

    weights = F.softmax(attn_score, dim=-1)    # [1, n]
    print(f"[Fusion] ✅ Softmax weights: {weights}")

    fused = torch.matmul(weights, stacked).squeeze(0)  # shape: [768]
    print(f"[Fusion] ✅ Fused vector shape: {fused.shape}")

    return fused


def summarize_vector(name, vec):
    normed = F.normalize(vec, dim=0)  # L2 정규화
    print(f"📌 [{name}] Summary")
    print(f" - Shape       : {normed.shape}")
    print(f" - Min         : {normed.min().item():.4f}")
    print(f" - Max         : {normed.max().item():.4f}")
    print(f" - Mean        : {normed.mean().item():.4f}")
    print(f" - Std Dev     : {normed.std().item():.4f}")
    return normed


# --- 예시 실행 (직접 실행 시)
if __name__ == "__main__":
    # 테스트 입력
    text_input = "아 왜이렇게 좆같냐 시발 진자 텍스트는 제대로 분석도 못하나 병신이 값이 왜 제대로 안나오는데?"
    audio_path = "example.wav"
    image_path = "example.jpg"

    # 인코더 로딩
    text_encoder = TextEmbeddingEncoder()
    audio_encoder = AudioEmbeddingEncoder()
    image_encoder = ImageEmbeddingEncoder()

    # 각 임베딩 벡터 추출 및 L2 정규화
    text_vec = summarize_vector("Text", text_encoder.encode(text_input))
    audio_vec = summarize_vector("Audio", audio_encoder.encode(audio_path))
    image_vec = summarize_vector("Image", image_encoder.encode(image_path))

    # Attention Fusion (기준: 이미지)
    fused_vec = attention_fusion([text_vec, audio_vec, image_vec], query_idx=2) 

    # 출력
    print("🎯 Fused Vector Shape:", fused_vec.shape)
    print("💡 First 10 Values:", fused_vec[:10].tolist())
