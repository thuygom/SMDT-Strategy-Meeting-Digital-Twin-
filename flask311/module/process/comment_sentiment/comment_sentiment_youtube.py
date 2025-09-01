import pandas as pd
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm
import os

tqdm.pandas()

# 원본 파일 경로
dir_path = 'C:/Users/bandl/OneDrive/바탕 화면/youtube_data/youtube_data/지무비/'

# 모델 로드
model_path = "D:/kcbert-emotion-finetuned"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)
model.eval()

# 감정 라벨 매핑
id2label = {
    0: '공포',
    1: '놀람',
    2: '분노',
    3: '슬픔',
    4: '중립',
    5: '행복',
    6: '혐오'
}

def predict_label(text):
    try:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            probs = F.softmax(logits, dim=1)
            pred = torch.argmax(probs, dim=1).item()
        return id2label[pred]
    except:
        return "오류"

# ✅ 해당 폴더의 모든 .xlsx 파일 처리
for filename in os.listdir(dir_path):
    if filename.endswith('.xlsx'):
        file_path = os.path.join(dir_path, filename)
        print(f"\n📂 처리 중: {filename}")

        try:
            df = pd.read_excel(file_path)

            if 'comment' in df.columns:
                감정_리스트 = df['comment'].progress_apply(predict_label)

                # 이미 감정 컬럼이 있으면 제거 후 재삽입 (중복 방지)
                if '감정' in df.columns:
                    df.drop(columns=['감정'], inplace=True)

                df.insert(5, '감정', 감정_리스트)

                # 🔄 덮어쓰기 저장
                df.to_excel(file_path, index=False)
                print(f"✅ 저장 완료: {filename}")
            else:
                print(f"⚠️ '댓글' 열이 존재하지 않음: {filename}")

        except Exception as e:
            print(f"❌ 오류 발생 ({filename}): {e}")