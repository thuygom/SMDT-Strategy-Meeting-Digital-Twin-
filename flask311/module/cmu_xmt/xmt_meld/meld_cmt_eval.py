import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
import argparse
import os
import json

from meld_dataset import MELDDataset
from meld_cmt import CrossModalTransformer  # mask 인자 없이 실행되는 버전

from fvcore.nn import FlopCountAnalysis

class WrappedModel(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
    def forward(self, text, audio, visual):
        output, _ = self.model(text, audio, visual)  # only use logits
        return output

def estimate_flops(model, device, seq_len=50):
    model.eval()

    # 추출
    dim_text = model.text_proj[0].in_features
    dim_audio = model.audio_proj[0].in_features
    dim_visual = model.visual_proj[0].in_features

    dummy_text = torch.randn(1, seq_len, dim_text).to(device)
    dummy_audio = torch.randn(1, seq_len, dim_audio).to(device)
    dummy_visual = torch.randn(1, seq_len, dim_visual).to(device)

    # Wrapped model
    wrapped_model = WrappedModel(model)

    flops = FlopCountAnalysis(wrapped_model, (dummy_text, dummy_audio, dummy_visual))
    print(f"🔧 Estimated FLOPs (1 sample @ seq_len={seq_len}): {flops.total() / 1e6:.2f} MFLOPs")


# ✅ Collate function
def meld_collate_fn(batch):
    texts, audios, visuals, labels = zip(*batch)
    padded_texts = pad_sequence(texts, batch_first=True)
    padded_audios = pad_sequence(audios, batch_first=True)
    padded_visuals = pad_sequence(visuals, batch_first=True)
    labels = torch.stack(labels)
    return padded_texts, padded_audios, padded_visuals, labels

# ✅ Evaluation
def evaluate(model, loader, device):
    model.eval()
    correct, total = 0, 0

    with torch.no_grad():
        for text, audio, visual, labels in loader:
            text, audio, visual, labels = text.to(device), audio.to(device), visual.to(device), labels.to(device)
            logits, _ = model(text, audio, visual)
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    return correct / total

# ✅ Main
def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"✅ Using device: {device}")

    # 🔹 Load mask
    with open(args.mask_path, 'r') as f:
        best_mask = json.load(f)
    print(f"🔍 해당 모델은 다음 마스크 조합으로 학습되었습니다: {best_mask}")

    # 🔹 Load dataset
    test_dataset = MELDDataset(split='test', data_dir=args.data_dir)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=meld_collate_fn)

    # 🔹 Load model
    model = CrossModalTransformer(
        dim_text=300,
        dim_audio=32,
        dim_visual=2048,
        dim_model=args.dim_model,
        n_heads=args.n_heads,
        dropout=args.dropout
    ).to(device)

    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()

    # 🔹 Evaluate Accuracy
    print("\n📊 Test Accuracy:")
    test_acc = evaluate(model, test_loader, device)
    print(f"✅ Test Accuracy: {test_acc:.4f}")

    # 🔹 Evaluate FLOPs
    estimate_flops(model, device)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="../meld_dataset")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--dim_model", type=int, default=256)
    parser.add_argument("--n_heads", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--model_path", type=str, default="./best_model.pth")
    parser.add_argument("--mask_path", type=str, default="../xmt_meld_supernet/checkpoints/best_mask.json")

    args = parser.parse_args()
    main(args)
