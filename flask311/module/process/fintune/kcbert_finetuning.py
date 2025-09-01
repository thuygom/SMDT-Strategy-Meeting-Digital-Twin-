import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import numpy as np

print("✅ CUDA 사용 가능 여부:", torch.cuda.is_available())

# 1. 엑셀 데이터 불러오기
df = pd.read_excel("test.xlsx")
df = df[['Sentence', 'Emotion']].copy()
df.columns = ['sentence', 'label']

# 2. 감정 라벨 숫자 인덱스로 매핑
label2id = {'공포': 0, '놀람': 1, '분노': 2, '슬픔': 3, '중립': 4, '행복': 5, '혐오': 6}
df = df[df['label'].isin(label2id.keys())].copy()
df['label'] = df['label'].map(label2id)

# 3. HuggingFace Dataset으로 변환
dataset = Dataset.from_pandas(df)

# 4. Tokenizer 로딩 및 토크나이징
model_name = "beomi/KcBERT-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)

def tokenize(example):
    return tokenizer(example['sentence'], truncation=True, padding='max_length', max_length=128)

dataset = dataset.map(tokenize)

# 5. 학습/검증 분할
dataset = dataset.train_test_split(test_size=0.1)
train_dataset = dataset['train']
eval_dataset = dataset['test']

# 6. 사전학습 모델 불러오기 (num_labels=7)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=7)

# 7. 학습 인자 설정 (D드라이브에 저장)
training_args = TrainingArguments(
    output_dir="D:/kcbert-emotion-checkpoints",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=2,
    load_best_model_at_end=True,
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir="D:/kcbert-emotion-logs",
    logging_steps=100
)

# 8. Trainer 구성
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
)

# 9. 학습 시작
trainer.train()

# 10. 최종 모델 저장 (D드라이브)
trainer.save_model("D:/kcbert-emotion-finetuned")
tokenizer.save_pretrained("D:/kcbert-emotion-finetuned")

# 예측
predictions = trainer.predict(eval_dataset)
preds = np.argmax(predictions.predictions, axis=1)
labels = predictions.label_ids

# 라벨 복원
id2label = {v: k for k, v in label2id.items()}
target_names = [id2label[i] for i in range(len(id2label))]

# 성능 분석표 출력
report = classification_report(labels, preds, target_names=target_names, digits=4)
print("📊 감정별 성능 분석:\n")
print(report)