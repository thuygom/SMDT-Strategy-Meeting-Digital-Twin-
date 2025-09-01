from mmsdk import mmdatasdk as md
import numpy as np
import os

# 데이터 경로 설정
data_path = "D:/CMU-MultimodalSDK/data/mosei"
label_config = {
    "labels": os.path.join(data_path, "CMU_MOSEI_Labels.csd")
}

# 데이터셋 로드
label_dataset = md.mmdataset(label_config)

# 잘못된 접근 방식: label_dataset.get("labels") ❌
# 올바른 접근 방식:
all_labels = label_dataset['labels'].data

# 전체 soft label 수집
label_vectors = []
for video_id in all_labels:
    for segment in all_labels[video_id]:
        label_vectors.append(segment[0])  # 7차원 감정 soft label

# numpy array로 변환 후 평균 분포 출력
label_array = np.array(label_vectors)
mean_distribution = label_array.mean(axis=0)

# 감정 라벨 이름
emotion_names = ['Happy', 'Sad', 'Angry', 'Disgust', 'Fear', 'Surprise', 'Neutral']

print("📊 MOSEI 감정 soft label 평균 분포:")
for name, val in zip(emotion_names, mean_distribution):
    print(f"{name:10s} : {val:.4f}")
