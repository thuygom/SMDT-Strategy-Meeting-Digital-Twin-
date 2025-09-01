import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
from tqdm import tqdm
from mmsdk import mmdatasdk

# ==========================================
# 1. GPU 설정
# ==========================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[Info] Using device: {device}")

# ==========================================
# 2. Helper Functions
# ==========================================
def get_avg_feature(seq):
    try:
        if isinstance(seq, dict):
            features = seq.get("features", None)
        else:
            features = seq
        if features is not None:
            features = features[:]  # HDF5 → ndarray
        avg = np.nanmean(features, axis=0)
        return avg
    except:
        return None

def safe_tensor(tensor_np):
    # NaN만 0으로 대체, inf는 유지
    tensor_np = np.nan_to_num(tensor_np, nan=0.0, posinf=np.inf, neginf=-np.inf)
    return torch.tensor(tensor_np, dtype=torch.float32).to(device)

def attention_fusion(vectors, query_idx=1):
    """
    vectors: List[Tensor] → 각 모달의 임베딩 (torch.tensor shape: [d])
    query_idx: 기준 모달 (default: 1 = audio)
    """
    # 유효한 모달리티만 필터링
    valid_vectors = []
    valid_indices = []

    for i, vec in enumerate(vectors):
        if not torch.isinf(vec).any():
            valid_vectors.append(vec)
            valid_indices.append(i)

    # 예외 처리: 모두 invalid일 경우 → zero vector 반환
    if len(valid_vectors) == 0:
        return torch.zeros_like(vectors[0]).cpu().numpy()

    # query index가 제외되었으면 대체 (기본적으로 첫 번째 유효 모달 사용)
    if query_idx not in valid_indices:
        query_idx = valid_indices[0]

    # query 벡터와 attention 계산
    stacked = torch.stack(valid_vectors)  # [n_valid, d]
    query_vec = vectors[query_idx].unsqueeze(0)  # [1, d]
    attn_score = torch.matmul(query_vec, stacked.T) / (query_vec.shape[-1] ** 0.5)
    weights = torch.softmax(attn_score, dim=-1)
    fused = torch.matmul(weights, stacked).squeeze(0)
    return fused.detach().cpu().numpy()

# ==========================================
# 3. Load Dataset
# ==========================================
data_path = "D:/CMU-MultimodalSDK/data/mosei"
features = {
    "glove_vectors": os.path.join(data_path, "CMU_MOSEI_TimestampedWordVectors.csd"),
    "COVAREP": os.path.join(data_path, "CMU_MOSEI_COVAREP.csd"),
    "FACET": os.path.join(data_path, "CMU_MOSEI_VisualFacet42.csd"),
    "labels": os.path.join(data_path, "CMU_MOSEI_Labels.csd"),
}
dataset = mmdatasdk.mmdataset(features, data_path)

# ==========================================
# 4. Get Common Segments
# ==========================================
def get_segment_ids(data):
    return set((vid, seg_id) for vid in data for seg_id in data[vid])

glove_ids = get_segment_ids(dataset["glove_vectors"].data)
covarep_ids = get_segment_ids(dataset["COVAREP"].data)
facet_ids = get_segment_ids(dataset["FACET"].data)
label_ids = get_segment_ids(dataset["labels"].data)
common_ids = glove_ids & covarep_ids & facet_ids & label_ids

# ==========================================
# 5. Define projection layers (GPU)
# ==========================================
linear_text = nn.Linear(300, 768).to(device)
linear_audio = nn.Linear(74, 768).to(device)
linear_visual = nn.Linear(35, 768).to(device)

# ==========================================
# 6. Build Dataset
# ==========================================
def build_dataset(dataset, common_ids):
    X, y = [], []
    total, dropped = 0, 0
    drop_reason = {"label_nan": 0, "interval": 0, "empty_modality": 0, "exception": 0}

    print("\n[Info] Building dataset with Attention Fusion...")
    for vid, seg_id in tqdm(common_ids, desc="🔄 Processing", unit="segment"):
        total += 1
        try:
            label = dataset["labels"].data[vid][seg_id][:]
            if label.shape[1] != 7:
                dropped += 1
                drop_reason["interval"] += 1
                continue

            label = label.mean(axis=0)  # shape: (7,)

            # NaN 또는 음수 클리어
            label = np.nan_to_num(label, nan=0.0)
            label = np.clip(label, a_min=0.0, a_max=None)

            label_sum = label.sum()
            if label_sum == 0:
                dropped += 1
                drop_reason["label_nan"] += 1
                continue
            label = label / label_sum  # 정규화된 soft label

            text_feat = get_avg_feature(dataset["glove_vectors"].data[vid][seg_id])
            audio_feat = get_avg_feature(dataset["COVAREP"].data[vid][seg_id])
            visual_feat = get_avg_feature(dataset["FACET"].data[vid][seg_id])

            if text_feat is None or audio_feat is None or visual_feat is None:
                dropped += 1
                drop_reason["empty_modality"] += 1
                continue

            proj_text = linear_text(safe_tensor(text_feat))
            proj_audio = linear_audio(safe_tensor(audio_feat))
            proj_visual = linear_visual(safe_tensor(visual_feat))

            fused = attention_fusion([proj_text, proj_audio, proj_visual], 2)

            if np.isnan(fused).any():
                dropped += 1
                drop_reason["label_nan"] += 1
                continue

            X.append(fused)
            y.append(label)

        except Exception as e:
            dropped += 1
            drop_reason["exception"] += 1
            print(f"[Error] {vid} / {seg_id} → {type(e).__name__}: {e}")
            continue

    print("\n[Debug] Dataset Summary")
    print(f" - 총 시도한 세그먼트 수: {total}")
    print(f" - 누락된 세그먼트 수: {dropped}")
    print(f"    • NaN 라벨로 제거됨: {drop_reason['label_nan']}")
    print(f"    • 비어있는 modality 제거됨: {drop_reason['empty_modality']}")
    print(f"    • 예외 처리로 제거됨: {drop_reason['interval']}")
    print(f" - 최종 학습 데이터 수: {len(X)}")

    if X:
        print(f" - 입력 벡터 shape: {X[0].shape}")
        for i in range(min(5, len(X))):
            print(f"\n🧪 샘플 {i+1}")
            print(f"  • 소프트 라벨: {y[i]}")
            print(f"  • 벡터 일반 (20): {X[i][:20]}")
    else:
        print("❌ 유효한 데이터가 없어 학습 불가")

    return np.array(X), np.array(y)

# ==========================================
# 7. Run
# ==========================================
if __name__ == "__main__":
    X, y = build_dataset(dataset, common_ids)
    np.save("X_mosei_fused.npy", X)
    np.save("y_mosei_labels.npy", y)
