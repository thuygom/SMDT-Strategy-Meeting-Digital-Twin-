import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
import argparse
import os
import json
from itertools import product

from meld_dataset import MELDDataset
from cmu_xmt.StackCMT.meld_cmt_stack import StackedCrossModalSuperNet


def meld_collate_fn(batch):
    texts, audios, visuals, labels = zip(*batch)
    padded_texts = pad_sequence(texts, batch_first=True)
    padded_audios = pad_sequence(audios, batch_first=True)
    padded_visuals = pad_sequence(visuals, batch_first=True)
    labels = torch.stack(labels)
    return padded_texts, padded_audios, padded_visuals, labels


def evaluate_mask(model, loader, device, mask):
    model.eval()
    correct, total = 0, 0

    with torch.no_grad():
        for text, audio, visual, labels in loader:
            text, audio, visual, labels = text.to(device), audio.to(device), visual.to(device), labels.to(device)
            logits = model.forward_one_mask(text, audio, visual, mask)
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    return correct / total


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"✅ Using device: {device}")

    # Load path mask list used for training (for reference)
    with open(os.path.join(args.data_dir, "GAT/supernet_mask_prior.json"), "r") as f:
        trained_masks = [entry["mask"] for entry in json.load(f)]

    # Load test dataset
    test_dataset = MELDDataset(split='test', data_dir=args.data_dir)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=meld_collate_fn)

    # Load SuperNet model
    model = StackedCrossModalSuperNet(
        path_mask_list=trained_masks,
        dim_text=300,
        dim_audio=32,
        dim_visual=2048,
        dim_model=args.dim_model,
        n_heads=args.n_heads,
        dropout=args.dropout
    ).to(device)
    model.load_state_dict(torch.load("best_model_supernet.pth", map_location=device))

    # Evaluate all 64 subnetworks
    all_masks = list(product([0, 1], repeat=6))
    all_masks = [mask for mask in all_masks if any(mask)]  # 제외: all-zero

    print("\n🔍 Evaluating All 63 Subnetworks:")
    mask_acc_list = []
    for mask in tqdm(all_masks):
        acc = evaluate_mask(model, test_loader, device, list(mask))
        mask_acc_list.append((mask, acc))

    # 정렬 및 출력
    mask_acc_list.sort(key=lambda x: x[1], reverse=True)
    print("\n🏁 Top 10 Subnetworks by Accuracy:")
    for i, (mask, acc) in enumerate(mask_acc_list[:10]):
        print(f"{i+1:2d}. Mask {mask} → Accuracy: {acc:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="./meld_dataset")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--dim_model", type=int, default=256)
    parser.add_argument("--n_heads", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.3)

    args = parser.parse_args()
    main(args)
