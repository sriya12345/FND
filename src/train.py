import random

import numpy as np
import torch
import torch.nn as nn
import wandb
from dataset import get_dataloader
from model import CLIPFakeNewsClassifier
import clip

SEED = 42


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def train(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for batch_idx, batch in enumerate(dataloader):
        image, text, label = batch
        image = image.to(device)
        label = label.float().to(device)
        optimizer.zero_grad()
        output = model(image, text)
        loss = criterion(output.squeeze(1), label)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        if batch_idx % 50 == 0:
            print(f"  Batch {batch_idx}/{len(dataloader)} | Loss: {loss.item():.4f}")

    return total_loss / len(dataloader)     # avg loss per batch    

def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            image, text, label = batch
            image = image.to(device)
            label = label.float().to(device)
            output = model(image, text)
            loss = criterion(output.squeeze(1), label)
            total_loss += loss.item()
            # accuracy 
            preds = (torch.sigmoid(output.squeeze(1))> 0.5).float()
            correct += (preds == label).sum().item()
            total += label.size(0)
            if batch_idx % 50 == 0:
                print(f"  Batch {batch_idx}/{len(dataloader)} | Loss: {loss.item():.4f}")

    return total_loss / len(dataloader), correct / total

def main():
    set_seed(SEED)

    num_epochs = 5
    lr = 1e-3
    batch_size = 32

    wandb.init(project="fakeddit-clip", config={
        "epochs": num_epochs, "lr": lr, "batch_size": batch_size, "seed": SEED,
    })

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CLIPFakeNewsClassifier().to(device)
    _, preprocess = clip.load("ViT-B/32")
    train_data = get_dataloader("data/multimodal_train_subset.tsv", "data/images/train",
                                 preprocess, batch_size=batch_size)
    val_data = get_dataloader("data/multimodal_validate_subset.tsv", "data/images/val",
                               preprocess, batch_size=batch_size)

    optimizer = torch.optim.Adam(model.mlp.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=1)
    criterion = nn.BCEWithLogitsLoss()

    best_val_acc = 0

    for epoch in range(num_epochs):
        train_loss = train(model, train_data, optimizer, criterion, device)
        val_loss, val_acc = evaluate(model, val_data, criterion, device)
        scheduler.step(val_loss)
        print(f"Epoch {epoch+1}/{num_epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

        wandb.log({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "lr": optimizer.param_groups[0]["lr"],
        })

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "checkpoints/best_model.pt")

    wandb.finish()

if __name__ == "__main__":
    main()