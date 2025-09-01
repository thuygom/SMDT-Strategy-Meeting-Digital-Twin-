import torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
import argparse
import os
from fvcore.nn import FlopCountAnalysis

from cmu_dataset import CMUMOSEIDataset
from cmu_cmt import CrossModalTransformer  # ✅ 마스크 없는 full pairwise CMT


# ✅ Forward Wrapper (fvcore용)
class ForwardOnlyWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, text, audio, visual):
        logits, _ = self.model(text, audio, visual)
        return logits


# ✅ FLOPs 측정
def estimate_flops(model, device):
    model.eval()
    dummy_text = torch.randn(1, 50, 300).to(device)
    dummy_audio = torch.randn(1, 50, 74).to(device)
    dummy_visual = torch.randn(1, 50, 35).to(device)

    wrapped_model = ForwardOnlyWrapper(model)
    flops = FlopCountAnalysis(wrapped_model, (dummy_text, dummy_audio, dummy_visual))

    print(f"🔧 Estimated FLOPs (1 sample @ seq_len=50): {flops.total() / 1e6:.2f} MFLOPs")


# ✅ Collate function
def cmu_collate_fn(batch):
    texts, audios, visuals, labels7, labels2 = zip(*batch)
    padded_texts = pad_sequence(texts, batch_first=True)
    padded_audios = pad_sequence(audios, batch_first=True)
    padded_visuals = pad_sequence(visuals, batch_first=True)
    labels7 = torch.tensor(labels7)
    labels2 = torch.tensor(labels2)
    return padded_texts, padded_audios, padded_visuals, labels7, labels2


# ✅ Evaluation
def evaluate(model, loader, device):
    model.eval()
    correct7, correct2, total = 0, 0, 0

    with torch.no_grad():
        for text, audio, visual, labels7, labels2 in loader:
            text, audio, visual = text.to(device), audio.to(device), visual.to(device)
            labels7, labels2 = labels7.to(device), labels2.to(device)

            logits, _ = model(text, audio, visual)
            preds7 = logits.argmax(dim=1)
            preds2 = (preds7 >= 3).long()

            correct7 += (preds7 == labels7).sum().item()
            correct2 += (preds2 == labels2).sum().item()
            total += labels7.size(0)

    return correct7 / total, correct2 / total


# ✅ Main
def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"✅ Using device: {device}")

    # Load dataset
    test_dataset = CMUMOSEIDataset(split='test', data_dir=args.data_dir)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=cmu_collate_fn)

    # Load full CMT model (모든 경로 활성화)
    model = CrossModalTransformer(
        dim_text=300,
        dim_audio=74,
        dim_visual=35,
        dim_model=args.dim_model,
        n_heads=args.n_heads,
        dropout=args.dropout
    ).to(device)

    model.load_state_dict(torch.load(args.model_path, map_location=device))

    print("\n📊 Test Accuracy:")
    acc7, acc2 = evaluate(model, test_loader, device)
    print(f"✅ 7-class Accuracy: {acc7:.4f}")
    print(f"✅ 2-class Accuracy: {acc2:.4f}")

    # FLOPs 계산
    estimate_flops(model, device)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="../cmu_dataset")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--dim_model", type=int, default=256)
    parser.add_argument("--n_heads", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--model_path", type=str, default="./best_model_cmu.pth")

    args = parser.parse_args()
    main(args)
