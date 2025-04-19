from sklearn.metrics import f1_score, roc_auc_score
import torch
import torch.nn.functional as F

def evaluate(model, dataloader, device='cuda', threshold=0.5):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device).float()

            logits = model(images)
            probs = torch.sigmoid(logits)

            all_preds.append(probs.cpu())
            all_labels.append(labels.cpu())

    # append all
    all_preds = torch.cat(all_preds, dim=0).numpy()
    all_labels = torch.cat(all_labels, dim=0).numpy()
    binary_preds = (all_preds > threshold).astype(int)

    # Accuracy
    acc = (binary_preds == all_labels).mean(axis=1).mean()

    # F1 scores
    f1_macro = f1_score(all_labels, binary_preds, average='macro', zero_division=0)
    f1_micro = f1_score(all_labels, binary_preds, average='micro', zero_division=0)

    # AUC scores
    try:
        auc_macro = roc_auc_score(all_labels, all_preds, average='macro')
        auc_micro = roc_auc_score(all_labels, all_preds, average='micro')
    except ValueError:
        auc_macro, auc_micro = float('nan'), float('nan')  # not count all 0

    print(f"\n[Evaluation Results]")
    print(f"Accuracy     : {acc:.4f}")
    print(f"F1 Macro     : {f1_macro:.4f}")
    print(f"F1 Micro     : {f1_micro:.4f}")
    print(f"AUC Macro    : {auc_macro:.4f}")
    print(f"AUC Micro    : {auc_micro:.4f}")

    return {
        'accuracy': acc,
        'f1_macro': f1_macro,
        'f1_micro': f1_micro,
        'auc_macro': auc_macro,
        'auc_micro': auc_micro
    }
