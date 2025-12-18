import torch
from sklearn.metrics import confusion_matrix, classification_report


def get_preds_and_cm(model, dataloader, device):
    """
    Generates predictions and ground truth labels for metrics.
    """
    model.eval()
    preds, labels = [], []

    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            logits = out[0] if isinstance(out, tuple) else out
            pred = logits.argmax(dim=1)

            preds.extend(pred.cpu().tolist())
            labels.extend(y.cpu().tolist())

    cm = confusion_matrix(labels, preds)
    return labels, preds, cm

def print_report(y_true, y_pred, class_names):
    print(classification_report(y_true, y_pred, target_names=class_names))