import torch
import torch.nn as nn
import wandb
from dataset import get_dataloader
from model import CLIPFakeNewsClassifier

def train(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for batch in dataloader:
        image, text, label = batch
        image = image.to(device)
        label = label.float().to(device)
        optimizer.zero_grad()
        output = model(image, text)
        loss = criterion(output.squeeze(1), label)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)     # avg loss per batch    

def eval(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in dataloader:
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
    return total_loss / len(dataloader), correct / total