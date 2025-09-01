import pickle
import numpy as np
import torch

# 파일 경로
pkl_path = "../meld_dataset/test_glove.pkl"

# 파일 로드
with open(pkl_path, "rb") as f:
    data = pickle.load(f)

print(f"📦 총 샘플 수: {len(data)}")

# 20개 샘플만 출력
for i, (key, sample) in enumerate(data.items()):
    print(f"\n📌 [샘플 ID] {key} ({i+1}/{len(data)})")

    # 각 항목 정보 출력
    for k, v in sample.items():
        if isinstance(v, np.ndarray):
            print(f"  ├─ {k}: numpy.ndarray / shape: {v.shape}")
        elif isinstance(v, list):
            print(f"  ├─ {k}: list / len: {len(v)}")
        elif isinstance(v, int):
            print(f"  ├─ {k}: int / value: {v}")
        else:
            print(f"  ├─ {k}: {type(v)} / {getattr(v, 'shape', v)}")

    # text 임베딩 shape 체크
    text = sample.get("text", None)
    if isinstance(text, np.ndarray):
        if text.ndim == 2 and text.shape[1] == 300:
            print(f"  ✅ text 임베딩 정상 (shape: {text.shape})")
        else:
            print(f"  ⚠️ text shape 이상: {text.shape}")
    else:
        print(f"  ⚠️ text 없음 또는 형식 오류: {type(text)}")

    if i >= 19:
        print("... (이후 생략)")
        break
