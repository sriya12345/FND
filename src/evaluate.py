import torch
import clip
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

from dataset import get_dataloader
from model import CLIPFakeNewsClassifier


def get_predictions(model, dataloader, device):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for image, text, label in dataloader:
            image = image.to(device)
            output = model(image, text)
            preds = (torch.sigmoid(output.squeeze(1)) > 0.5).float()
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(label.float().tolist())
    return all_labels, all_preds


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = CLIPFakeNewsClassifier().to(device)
    model.load_state_dict(torch.load("checkpoints/best_model.pt", map_location=device))

    _, preprocess = clip.load("ViT-B/32")
    test_data = get_dataloader(
        "data/multimodal_test_public_subset.tsv", "data/images/test", preprocess,
        shuffle=False,
    )

    labels, preds = get_predictions(model, test_data, device)

    acc = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="binary")
    cm = confusion_matrix(labels, preds)

    print(f"Test Accuracy:  {acc:.4f}")
    print(f"Precision:      {precision:.4f}")
    print(f"Recall:         {recall:.4f}")
    print(f"F1:             {f1:.4f}")
    print("Confusion matrix [[TN, FP], [FN, TP]]:")
    print(cm)


if __name__ == "__main__":
    main()
