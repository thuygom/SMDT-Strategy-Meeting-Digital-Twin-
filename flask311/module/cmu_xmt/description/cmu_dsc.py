import pickle
import numpy as np

# 파일 경로
pkl_path = "../cmu_dataset/cmu_mosei_test_fixed.pkl"  # 필요 시 수정

# 파일 로드
with open(pkl_path, "rb") as f:
    data = pickle.load(f)

print(f"📦 총 샘플 수: {len(data)}")

# 결측치 이상값 포함 샘플 개수 카운트
problem_count = 0

# 전체 순회
for i, (key, sample) in enumerate(data.items()):
    print(f"\n📌 [샘플 ID] {key} ({i+1}/{len(data)})")

    has_issue = False

    for mod in ['text', 'audio', 'visual']:
        arr = sample.get(mod, None)
        if arr is None:
            print(f"  ⚠️ {mod} 없음")
            continue

        if not isinstance(arr, np.ndarray):
            print(f"  ⚠️ {mod}의 데이터 타입이 numpy.ndarray가 아님: {type(arr)}")
            continue

        if np.isnan(arr).any():
            print(f"  ❌ [NaN] in {mod} → shape: {arr.shape}")
            print(f"     ▶ 값 예시:\n{arr}")
            has_issue = True

        if np.isinf(arr).any():
            print(f"  ❌ [Inf] in {mod} → shape: {arr.shape}")
            print(f"     ▶ 값 예시:\n{arr}")
            has_issue = True

        if not has_issue:
            print(f"  ✅ {mod}: 정상 (shape: {arr.shape})")

    if has_issue:
        problem_count += 1

print(f"\n🔎 결측치 또는 이상값 포함 샘플 수: {problem_count} / {len(data)}")
