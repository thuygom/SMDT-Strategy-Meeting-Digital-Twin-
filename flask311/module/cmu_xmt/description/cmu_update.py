import pickle
import numpy as np
import os

def fix_array(arr):
    """NaN, +inf, -inf를 모두 0.0으로 대체"""
    return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)

def fix_sample(sample):
    """sample 딕셔너리 내부의 모달 배열을 fix_array로 처리"""
    for mod in ['text', 'audio', 'visual']:
        if mod in sample and isinstance(sample[mod], np.ndarray):
            sample[mod] = fix_array(sample[mod])
    return sample

def fix_pkl_file(input_path, output_path):
    """입력 경로에서 불러와 결측값을 보정 후 출력 경로에 저장"""
    with open(input_path, "rb") as f:
        data = pickle.load(f)

    print(f"📦 원본 샘플 수: {len(data)}")

    # 보정
    fixed_data = {k: fix_sample(v) for k, v in data.items()}

    # 저장
    with open(output_path, "wb") as f:
        pickle.dump(fixed_data, f)

    print(f"✅ 결측치 보정 완료: {output_path}")

# 예시 실행
fix_pkl_file(
    input_path="../cmu_dataset/cmu_mosei_train.pkl",
    output_path="../cmu_dataset/cmu_mosei_train_fixed.pkl"
)
