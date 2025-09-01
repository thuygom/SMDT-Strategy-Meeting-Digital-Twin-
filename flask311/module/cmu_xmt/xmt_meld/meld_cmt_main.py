import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
import argparse
import csv
import os

from meld_dataset import MELDDataset
from meld_cmt import CrossModalTransformer


def make_log_dir(path):
    os.makedirs(path, exist_ok=True)


def save_attention_logs(logs, filename):
    fieldnames = [
        'text->audio', 'text->visual',
        'audio->text', 'audio->visual',
        'visual->text', 'visual->audio'
    ]
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for log in logs:
            writer.writerow({k: float(v) for k, v in log.items()})


def meld_collate_fn(batch):
    texts, audios, visuals, labels = zip(*batch)
    padded_texts = pad_sequence(texts, batch_first=True)
    padded_audios = pad_sequence(audios, batch_first=True)
    padded_visuals = pad_sequence(visuals, batch_first=True)
    labels = torch.stack(labels)
    return padded_texts, padded_audios, padded_visuals, labels


def train(model, loader, criterion, optimizer, device, log_path=None):
    model.train()
    total_loss, correct, total = 0, 0, 0
    attn_logs_list = []

    for text, audio, visual, labels in tqdm(loader, desc="Train"):
        text, audio, visual, labels = text.to(device), audio.to(device), visual.to(device), labels.to(device)

        optimizer.zero_grad()
        logits, attn_logs = model(text, audio, visual)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, preds = torch.max(logits, dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        attn_logs_list.append(attn_logs)

    if log_path:
        save_attention_logs(attn_logs_list, log_path)

    acc = correct / total
    avg_loss = total_loss / len(loader)
    return avg_loss, acc


def evaluate(model, loader, criterion, device, log_path=None):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    attn_logs_list = []

    with torch.no_grad():
        for text, audio, visual, labels in tqdm(loader, desc="Eval"):
            text, audio, visual, labels = text.to(device), audio.to(device), visual.to(device), labels.to(device)

            logits, attn_logs = model(text, audio, visual)
            loss = criterion(logits, labels)

            total_loss += loss.item()
            _, preds = torch.max(logits, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            attn_logs_list.append(attn_logs)

    if log_path:
        save_attention_logs(attn_logs_list, log_path)

    acc = correct / total
    avg_loss = total_loss / len(loader)
    return avg_loss, acc


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"✅ Using device: {device}")

    log_train_dir = os.path.join("log", "train")
    log_dev_dir = os.path.join("log", "dev")
    make_log_dir(log_train_dir)
    make_log_dir(log_dev_dir)

    train_dataset = MELDDataset(split='train', data_dir=args.data_dir)
    dev_dataset = MELDDataset(split='dev', data_dir=args.data_dir)
    test_dataset = MELDDataset(split='test', data_dir=args.data_dir)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=meld_collate_fn)
    dev_loader = DataLoader(dev_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=meld_collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=meld_collate_fn)

    model = CrossModalTransformer(
        dim_text=300,
        dim_audio=32,
        dim_visual=2048,
        dim_model=args.dim_model,
        n_heads=args.n_heads,
        dropout=args.dropout
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)

    best_dev_acc = 0

    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")
        train_loss, train_acc = train(
            model, train_loader, criterion, optimizer, device,
            log_path=os.path.join(log_train_dir, f"epoch{epoch+1}.csv")
        )
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")

        dev_loss, dev_acc = evaluate(
            model, dev_loader, criterion, device,
            log_path=os.path.join(log_dev_dir, f"epoch{epoch+1}.csv")
        )
        print(f"Dev Loss: {dev_loss:.4f} | Dev Acc: {dev_acc:.4f}")

        if dev_acc > best_dev_acc:
            best_dev_acc = dev_acc
            torch.save(model.state_dict(), "best_model.pth")
            print("✅ Best model saved!")

    print("\n🏁 Training complete! Loading best model...")
    model.load_state_dict(torch.load("best_model.pth"))

    print("\n🎯 Final Test Evaluation (All Paths Activated):")
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    print(f"Test Accuracy: {test_acc:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="../meld_dataset")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--dim_model", type=int, default=256)
    parser.add_argument("--n_heads", type=int, default=8)

    args = parser.parse_args()
    main(args)
