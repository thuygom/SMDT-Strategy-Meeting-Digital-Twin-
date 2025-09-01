import argparse
import torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
import json
import os
from tqdm import tqdm

from cmu_dataset import CMUMOSEIDataset
from cmu_cmt_supernet import CrossModalSuperNet
from cmu_nas import NASController


def collate_fn(batch):
    texts, audios, visuals, labels7, _ = zip(*batch)  # label2는 사용 안함
    return pad_sequence(texts, batch_first=True), pad_sequence(audios, batch_first=True), pad_sequence(visuals, batch_first=True), torch.stack(labels7)


def train_one_epoch(model, loader, criterion, optimizer, device, subnet_masks):
    model.train()
    total, correct = 0, 0
    for text, audio, visual, labels in tqdm(loader, desc="Training", leave=False):
        text, audio, visual, labels = text.to(device), audio.to(device), visual.to(device), labels.to(device)
        optimizer.zero_grad()
        logits_list = []
        for mask in subnet_masks:
            if sum(mask) == 0:
                continue
            logits, _ = model(text, audio, visual, mask=mask)
            logits_list.append(logits)

        if len(logits_list) == 0:
            raise ValueError("🚫 All subnet masks are invalid (no active paths).")

        logits_avg = torch.mean(torch.stack(logits_list, dim=0), dim=0)
        loss = criterion(logits_avg, labels)
        loss.backward()
        optimizer.step()
        correct += (logits_avg.argmax(dim=1) == labels).sum().item()
        total += labels.size(0)
    return correct / total


def evaluate(model, data_loader, device, mask_list):
    model.eval()
    results = []

    with torch.no_grad():
        for i, mask in enumerate(mask_list):
            all_preds, all_labels = [], []
            for batch in tqdm(data_loader, desc=f"Evaluating Mask {i+1}/{len(mask_list)}: {mask}", leave=False):
                text, audio, visual, labels = [x.to(device) for x in batch]
                logits, _ = model(text, audio, visual, mask=mask)
                preds = torch.argmax(logits, dim=1)
                all_preds.append(preds)
                all_labels.append(labels)
            preds = torch.cat(all_preds)
            labels = torch.cat(all_labels)
            acc = (preds == labels).float().mean().item()
            flops = model.compute_flops(mask)
            results.append({"mask": mask, "acc": acc, "flops": flops})
    return results


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("✅ Device:", device)

    os.makedirs("checkpoints_cmu", exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    train_data = CMUMOSEIDataset(split='train', data_dir=args.data_dir)
    dev_data = CMUMOSEIDataset(split='dev', data_dir=args.data_dir)
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    dev_loader = DataLoader(dev_data, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    nas = NASController(
        prior_path=args.prior_path,
        mutation_rate=args.mutation,
        alpha=args.alpha,
        weight=args.weight
    )

    best_overall_acc = 0.0
    best_overall_mask = None
    best_model_state_dict = None

    for round in range(args.nas_rounds):
        print(f"\n🔁 NAS Round {round + 1}/{args.nas_rounds}")
        current_subnets = nas.sample_subnets(top_k=args.top_k)
        print("   🔹 Active Subnets:")
        for idx, mask in enumerate(current_subnets):
            print(f"     - {idx + 1}: {mask}")

        model = CrossModalSuperNet(
            dim_text=300,
            dim_audio=74,
            dim_visual=35,
            dim_model=args.dim_model,
            n_heads=args.n_heads,
            dropout=args.dropout
        ).to(device)

        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
        criterion = torch.nn.CrossEntropyLoss()

        best_acc = 0.0
        best_mask = None

        for epoch in range(args.epochs_per_round):
            acc = train_one_epoch(model, train_loader, criterion, optimizer, device, current_subnets)
            print(f"  Epoch {epoch + 1}: Train Acc = {acc:.4f}")

            val_results = evaluate(model, dev_loader, device, current_subnets)
            avg_val_acc = sum([v["acc"] for v in val_results]) / len(val_results)
            print(f"📌 Round {round + 1}, Epoch {epoch + 1} Avg Validation Accuracy: {avg_val_acc:.4f}")

            if avg_val_acc > best_acc:
                best_acc = avg_val_acc
                best_mask = max(val_results, key=lambda x: x["acc"])['mask']
                torch.save(model.state_dict(), f"checkpoints_cmu/supernet_round{round + 1}.pth")

            with open(f"logs/round_{round + 1}_epoch_{epoch + 1}_val_results.json", "w") as f:
                json.dump(val_results, f, indent=2)

            for res in val_results:
                nas.update_rewards([res["mask"]], res["acc"], res["flops"])

        nas.save_rewards(f"logs/rewards_round{round + 1}.json")

        if best_acc > best_overall_acc:
            best_overall_acc = best_acc
            best_overall_mask = best_mask
            best_model_state_dict = model.state_dict()

    if best_model_state_dict is not None:
        torch.save(best_model_state_dict, "checkpoints_cmu/best_supernet_overall.pth")
        with open("checkpoints_cmu/best_mask.json", "w") as f:
            json.dump(best_overall_mask, f)
        print(f"🏆 Best model saved with acc = {best_overall_acc:.4f} and mask = {best_overall_mask}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="../cmu_dataset")
    parser.add_argument("--prior_path", type=str, default="../cmu_dataset/GAT/supernet_mask_prior.json")
    parser.add_argument("--dim_model", type=int, default=256)
    parser.add_argument("--n_heads", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs_per_round", type=int, default=7)
    parser.add_argument("--nas_rounds", type=int, default=5)
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--mutation", type=float, default=0.2)
    parser.add_argument("--alpha", type=float, default=0.6)
    parser.add_argument("--weight", type=float, default=0.8)
    args = parser.parse_args()
    main(args)
