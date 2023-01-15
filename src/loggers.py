import pytorch_lightning as pl
import torch
import wandb


class ImagePredictionLogger(pl.Callback):
    def __init__(self, val_samples, num_samples=32):
        super().__init__()
        self.val_imgs, self.val_labels = val_samples
        self.val_imgs = self.val_imgs[:num_samples]
        self.val_labels = self.val_labels[:num_samples]

    def on_validation_epoch_end(self, trainer, pl_module):
        val_imgs = self.val_imgs.to(device=pl_module.device)

        logits = pl_module(val_imgs)
        preds = torch.argmax(logits, 1)

        trainer.logger.experiment.log(
            {
                "examples": [
                    wandb.Image(x, caption=self.labels_to_caption(pred, y))
                    for x, pred, y in zip(val_imgs, preds, self.val_labels)
                ],
                "global_step": trainer.global_step,
            }
        )

    def labels_to_caption(self, pred, y):
        mapping = {
            0: "airplane",
            1: "car",
            2: "bird",
            3: "cat",
            4: "deer",
            5: "dog",
            6: "frog",
            7: "horse",
            8: "ship",
            9: "truck",
        }
        pred_name = mapping[pred.item()]
        y_name = mapping[y.item()]
        return f"Pred: {pred_name}, Label: {y_name}"
