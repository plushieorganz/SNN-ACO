import torch


def accuracy(preds: torch.Tensor, targets: torch.Tensor) -> float:
    correct = (preds == targets).sum().item()
    return correct / max(1, len(targets))


def f1_score(preds: torch.Tensor, targets: torch.Tensor) -> float:
    preds = preds.detach().cpu()
    targets = targets.detach().cpu()
    tp = ((preds == 1) & (targets == 1)).sum().item()
    fp = ((preds == 1) & (targets == 0)).sum().item()
    fn = ((preds == 0) & (targets == 1)).sum().item()
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall + 1e-8)


def compute_classification_metrics(logits: torch.Tensor, targets: torch.Tensor):
    preds = torch.argmax(logits, dim=1)
    return {
        "acc": accuracy(preds, targets),
        "f1": f1_score(preds, targets),
    }
