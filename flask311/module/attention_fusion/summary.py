import numpy as np

# 파일 불러오기
#X = np.load("X_augmented.npy")   # shape: (N, 768)
#y = np.load("y_augmented.npy")  # shape: (N,)

# ✅ 데이터 로딩
X = np.load("x_attention.npy")  # shape: (N, 768)
y = np.load("y_attention.npy")  # shape: (N, 7) soft label

print("📦 X shape:", X.shape)
print("🔢 y shape:", y.shape)

# NaN 검사
nan_count = np.isnan(X).sum()
print(f"\n⚠️ NaN 개수: {nan_count} (비율: {nan_count / X.size:.6f})")

# 전체 통계
print("\n📊 X 통계 (전체):")
print(" - 평균:", np.nanmean(X))
print(" - 표준편차:", np.nanstd(X))
print(" - 최소값:", np.nanmin(X))
print(" - 최대값:", np.nanmax(X))

# 축별 통계
print("\n📈 X 통계 (특성별):")
feature_means = np.nanmean(X, axis=0)
feature_stds = np.nanstd(X, axis=0)
print(" - 첫 10개 특성 평균:", feature_means[:10])
print(" - 첫 10개 특성 표준편차:", feature_stds[:10])

# ✅ y 소프트 라벨 분포 (평균 확률 기반)
emotion_names = ['Happy', 'Sad', 'Angry', 'Disgust', 'Fear', 'Surprise', 'Neutral']
mean_emotion_dist = y.mean(axis=0)

print("\n📊 y 소프트 라벨 평균 분포:")
for i, (name, val) in enumerate(zip(emotion_names, mean_emotion_dist)):
    print(f" - {i}: {name:10s} | 평균 확률: {val:.4f}")

