import torch
from torch.utils.data import Dataset
import pickle
import os
import numpy as np

class MELDDataset(Dataset):
    def __init__(self, split='train', data_dir='../dataset_meld'):
        assert split in ['train', 'dev', 'test'], "split must be 'train', 'dev', or 'test'"
        self.file_path = os.path.join(data_dir, f"{split}_glove.pkl")  # ✅ GloVe 적용된 파일 사용

        with open(self.file_path, 'rb') as f:
            self.data = pickle.load(f)

        self.samples = list(self.data.values())

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]

        def to_tensor(data, expected_dim=None):
            if data is None:
                if expected_dim is None:
                    return torch.tensor([])  # empty tensor
                return torch.zeros((1, expected_dim), dtype=torch.float32)
            if isinstance(data, np.ndarray):
                return torch.tensor(data, dtype=torch.float32)
            if isinstance(data, list):
                return torch.tensor(data, dtype=torch.float32)
            raise ValueError(f"Unsupported data type: {type(data)}")

        # ✅ GloVe 임베딩 사용
        text = to_tensor(item.get('text'), expected_dim=300)             # shape: [T, 300]
        audio = to_tensor(item.get('audio_features'), expected_dim=32)   # shape: [T, 32]
        visual = to_tensor(item.get('video_features'), expected_dim=2048)  # shape: [T, 2048]
        label = torch.tensor(item['label'], dtype=torch.long)            # shape: []

        return text, audio, visual, label
