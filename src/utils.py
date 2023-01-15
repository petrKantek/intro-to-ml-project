import matplotlib.pyplot as plt
import numpy as np
import torchvision.transforms.functional as F
from torchmetrics import Accuracy


def get_acc(n_classes: int) -> Accuracy:
    if n_classes > 2:
        return Accuracy(task="multiclass", num_classes=n_classes)
    return Accuracy(task="binary")


def show(imgs):
    if not isinstance(imgs, list):
        imgs = [imgs]
    fig, axs = plt.subplots(ncols=len(imgs), squeeze=False, figsize=(20, 20))
    for i, img in enumerate(imgs):
        img = img.detach()
        img = F.to_pil_image(img)
        axs[0, i].imshow(np.asarray(img))
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
