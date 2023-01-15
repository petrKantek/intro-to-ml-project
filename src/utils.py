from torchmetrics import Accuracy


def get_acc(n_classes: int) -> Accuracy:
    if n_classes > 2:
        return Accuracy(task="multiclass", num_classes=n_classes)
    return Accuracy(task="binary")
