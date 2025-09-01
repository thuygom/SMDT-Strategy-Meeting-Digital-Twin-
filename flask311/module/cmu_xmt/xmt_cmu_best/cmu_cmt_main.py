import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
import argparse
import csv
import os

from cmu_dataset import CMUMOSEIDataset
from cmu_cmt import CrossModalTransformer


def make_log_dir(path):
    os.makedirs(path, exist_ok=True)


def save_attention_logs(logs, filename):
    fieldnames = [
        'text→audio', 'text→visual',
        'audio→text', 'audio→visual',
        'visual→text', 'visual→audio'
    ]
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for log in logs:
            writer.writerow({k: float(v) for k, v in log.items()})


def cmu_collate_fn(batch):
    texts, audios, visuals, labels7, labels2 = zip(*batch)
    padded_texts = pad_sequence(texts, batch_first=True)
    padded_audios = pad_sequence(audios, batch_first=True)
    padded_visuals = pad_sequence(visuals, batch_first=True)
    labels7 = torch.stack(labels7)
    labels2 = torch.stack(labels2)
    return padded_texts, padded_audios, padded_visuals, labels7, labels2


def train(model, loader, criterion, optimizer, device, log_path=None):
    model.train()
    total_loss, correct7, correct2, total = 0, 0, 0, 0
    attn_logs_list = []

    for text, audio, visual, label7, label2 in tqdm(loader, desc="Train"):
        text, audio, visual = text.to(device), audio.to(device), visual.to(device)
        label7, label2 = label7.to(device), label2.to(device)

        optimizer.zero_grad()
        logits, attn_logs = model(text, audio, visual)
        loss = criterion(logits, label7)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        pred7 = torch.argmax(logits, dim=1)
        pred2 = (pred7 >= 3).long()

        correct7 += (pred7 == label7).sum().item()
        correct2 += (pred2 == label2).sum().item()
        total += label7.size(0)

        attn_logs_list.append(attn_logs)

    if log_path:
        save_attention_logs(attn_logs_list, log_path)

    acc7 = correct7 / total
    acc2 = correct2 / total
    avg_loss = total_loss / len(loader)
    return avg_loss, acc7, acc2


def evaluate(model, loader, criterion, device, log_path=None):
    model.eval()
    total_loss, correct7, correct2, total = 0, 0, 0, 0
    attn_logs_list = []

    with torch.no_grad():
        for text, audio, visual, label7, label2 in tqdm(loader, desc="Eval"):
            text, audio, visual = text.to(device), audio.to(device), visual.to(device)
            label7, label2 = label7.to(device), label2.to(device)

            logits, attn_logs = model(text, audio, visual)
            loss = criterion(logits, label7)

            total_loss += loss.item()
            pred7 = torch.argmax(logits, dim=1)
            pred2 = (pred7 >= 3).long()

            correct7 += (pred7 == label7).sum().item()
            correct2 += (pred2 == label2).sum().item()
            total += label7.size(0)

            attn_logs_list.append(attn_logs)

    if log_path:
        save_attention_logs(attn_logs_list, log_path)

    acc7 = correct7 / total
    acc2 = correct2 / total
    avg_loss = total_loss / len(loader)
    return avg_loss, acc7, acc2


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"✅ Using device: {device}")

    log_train_dir = os.path.join("log", "train")
    log_dev_dir = os.path.join("log", "dev")
    make_log_dir(log_train_dir)
    make_log_dir(log_dev_dir)

    train_dataset = CMUMOSEIDataset(split='train', data_dir=args.data_dir)
    dev_dataset = CMUMOSEIDataset(split='dev', data_dir=args.data_dir)
    test_dataset = CMUMOSEIDataset(split='test', data_dir=args.data_dir)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=cmu_collate_fn)
    dev_loader = DataLoader(dev_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=cmu_collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=cmu_collate_fn)

    model = CrossModalTransformer(
        dim_text=300,
        dim_audio=74,
        dim_visual=35,
        dim_model=args.dim_model,
        n_heads=args.n_heads,
        dropout=args.dropout,
        mask_path=args.mask_path  # ✅ mask 경로 전달
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)

    best_dev_acc = 0

    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")
        train_loss, train_acc7, train_acc2 = train(
            model, train_loader, criterion, optimizer, device,
            log_path=os.path.join(log_train_dir, f"epoch{epoch+1}.csv")
        )
        print(f"Train Loss: {train_loss:.4f} | acc7: {train_acc7:.4f} | acc2: {train_acc2:.4f}")

        dev_loss, dev_acc7, dev_acc2 = evaluate(
            model, dev_loader, criterion, device,
            log_path=os.path.join(log_dev_dir, f"epoch{epoch+1}.csv")
        )
        print(f"Dev Loss:   {dev_loss:.4f} | acc7: {dev_acc7:.4f} | acc2: {dev_acc2:.4f}")

        if dev_acc7 > best_dev_acc:
            best_dev_acc = dev_acc7
            torch.save(model.state_dict(), "best_model_cmu.pth")
            print("✅ Best model saved!")

    print("\n🏁 Training complete! Loading best model...")
    model.load_state_dict(torch.load("best_model_cmu.pth"))
    test_loss, test_acc7, test_acc2 = evaluate(model, test_loader, criterion, device)
    print(f"🎯 Test acc7: {test_acc7:.4f} | acc2: {test_acc2:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="../cmu_dataset")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--dropout", type=float, default=0.4)
    parser.add_argument("--dim_model", type=int, default=256)
    parser.add_argument("--n_heads", type=int, default=8)
    parser.add_argument("--mask_path", type=str, default="../xmt_cmu_supernet/checkpoints_cmu/best_mask.json")  # ✅ 추가

    args = parser.parse_args()
    main(args)
