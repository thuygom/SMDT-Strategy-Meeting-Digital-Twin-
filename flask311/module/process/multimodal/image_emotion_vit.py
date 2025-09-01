from transformers import AutoFeatureExtractor, AutoModelForImageClassification
from PIL import Image
import torch

class ViTEmotionClassifier:
    def __init__(self, model_name="trpakov/vit-face-expression"):
        self.extractor = AutoFeatureExtractor.from_pretrained(model_name)
        self.model = AutoModelForImageClassification.from_pretrained(model_name)
        self.model.eval()

    def predict_emotion(self, image_path):
        image = Image.open(image_path).convert("RGB")
        inputs = self.extractor(images=image, return_tensors="pt")
        with torch.no_grad():
            outputs = self.model(**inputs)
        pred = torch.argmax(outputs.logits, dim=1).item()
        label = self.model.config.id2label[pred]
        return label

# 예제 실행
if __name__ == "__main__":
    classifier = ViTEmotionClassifier()
    emotion = classifier.predict_emotion("example.jpg")
    print("예측된 감정:", emotion)
