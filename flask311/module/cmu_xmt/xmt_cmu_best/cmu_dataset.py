import torch
from torch.utils.data import Dataset
import pickle
import os
import numpy as np

class CMUMOSEIDataset(Dataset):
    def __init__(self, split='train', data_dir='../dataset_cmu'):
        assert split in ['train', 'dev', 'test'], "split must be 'train', 'dev', or 'test'"
        file_path = os.path.join(data_dir, f"cmu_mosei_{split}_fixed.pkl")

        with open(file_path, 'rb') as f:
            self.data = pickle.load(f)

        self.samples = list(self.data.values())

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]

        def to_tensor(data, expected_dim=None):
            if data is None:
                return torch.zeros((1, expected_dim), dtype=torch.float32)

            if isinstance(data, list):
                data = np.array(data)

            if isinstance(data, np.ndarray):
                data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)

                if data.ndim == 1:
                    data = data.reshape(1, -1)

                if data.ndim == 2 and expected_dim is not None:
                    current_dim = data.shape[1]
                    if current_dim != expected_dim:
                        fixed = np.zeros((data.shape[0], expected_dim), dtype=np.float32)
                        dim_to_copy = min(current_dim, expected_dim)
                        fixed[:, :dim_to_copy] = data[:, :dim_to_copy]
                        data = fixed

                return torch.tensor(data, dtype=torch.float32)

            raise ValueError(f"Unsupported data type: {type(data)}")

        # Feature tensors
        text = to_tensor(item.get('text'), expected_dim=300)
        audio = to_tensor(item.get('audio'), expected_dim=74)
        visual = to_tensor(item.get('visual'), expected_dim=35)

        # Label 처리 (7-class 및 acc2 binary)
        label = item.get('label')
        if isinstance(label, np.ndarray) and label.ndim > 0:
            label7 = int(np.argmax(label))
        else:
            label7 = int(label)

        label2 = 1 if label7 >= 3 else 0  # 중립 이상은 positive

        return text, audio, visual, torch.tensor(label7, dtype=torch.long), torch.tensor(label2, dtype=torch.long)
