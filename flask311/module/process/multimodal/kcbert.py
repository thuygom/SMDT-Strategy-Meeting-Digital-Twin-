import pandas as pd
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from sklearn.metrics import accuracy_score

from sklearn.metrics import confusion_matrix
import numpy as np

# 🔄 tqdm 설정
tqdm.pandas()  # Pandas의 .apply에 progress bar 추가

# 1. 모델 로드
model_path = "D:/kcbert-emotion-finetuned"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)
model.eval()

# 2. 감정 라벨 매핑
id2label = {
    0: '공포',
    1: '놀람',
    2: '분노',
    3: '슬픔',
    4: '중립',
    5: '행복',
    6: '혐오'
}
label2id = {v: k for k, v in id2label.items()}

# 3. 감정 예측 함수
def predict_label(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = F.softmax(logits, dim=1)
        pred = torch.argmax(probs, dim=1).item()
    return pred

# 4. 데이터 로드 및 전처리
df = pd.read_excel("./test.xlsx")
df = df[['Sentence', 'Emotion']].copy()
df.columns = ['sentence', 'label']
df = df[df['label'].isin(label2id.keys())].copy()
df['label_id'] = df['label'].map(label2id)

# 5. 데이터 분할 (50% 테스트용)
_, test_df = train_test_split(df, test_size=0.5, stratify=df['label_id'], random_state=42)

# 6. 예측 수행 (progress bar)
test_df['predicted_id'] = test_df['sentence'].progress_apply(predict_label)

# 7. 평가 지표 출력

print("📊 Classification Report:\n")
print(classification_report(
    test_df['label_id'],
    test_df['predicted_id'],
    target_names=[id2label[i] for i in range(7)],
    digits=4
))

# confusion matrix
cm = confusion_matrix(test_df['label_id'], test_df['predicted_id'], labels=list(range(7)))

# 감정별 정확도 계산
print("📌 진짜 감정별 정확도 (Per-class Accuracy):")
total = cm.sum()
for i, label in id2label.items():
    TP = cm[i, i]
    TN = total - cm[i, :].sum() - cm[:, i].sum() + TP
    acc = (TP + TN) / total
    print(f"  {label}: {acc:.4f}")
