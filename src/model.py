import pytorch_lightning as pl
import torch
import wandb

from utils import get_acc


class CIFAR10Model(pl.LightningModule):
    def __init__(
        self,
        model: torch.nn.Module,
        in_dims: tuple[int, int, int, int],
        n_classes: int = 10,
        lr: float = 1e-4,
    ):
        super().__init__()
        self.model = model

        self.save_hyperparameters(ignore=["model"])

        self.train_acc = get_acc(n_classes)
        self.valid_acc = get_acc(n_classes)
        self.test_acc = get_acc(n_classes)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits, loss = self.loss(x, y)
        preds = torch.argmax(logits, 1)

        self.log("train/loss", loss, on_epoch=True)
        self.train_acc(preds, y)
        self.log("train/acc", self.train_acc, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits, loss = self.loss(x, y)
        preds = torch.argmax(logits, 1)

        self.valid_acc(preds, y)
        self.log("valid/loss_epoch", loss)  # default on val/test is on_epoch only
        self.log("valid/acc_epoch", self.valid_acc)

        return logits

    def validation_epoch_end(self, outputs):
        dummy_input = torch.zeros(self.hparams["in_dims"], device=self.device)
        model_filename = f"model_{str(self.global_step).zfill(5)}.onnx"
        torch.onnx.export(self, dummy_input, model_filename, opset_version=11)
        artifact = wandb.Artifact(name="model.ckpt", type="model")
        artifact.add_file(model_filename)
        self.logger.experiment.log_artifact(artifact)

        flattened_logits = torch.flatten(torch.cat(validation_step_outputs))
        self.logger.experiment.log(
            {
                "valid/logits": wandb.Histogram(flattened_logits.to("cpu")),
                "global_step": self.global_step,
            }
        )

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits, loss = self.loss(x, y)
        preds = torch.argmax(logits, 1)

        self.test_acc(preds, y)
        self.log("test/loss_epoch", loss, on_step=False, on_epoch=True)
        self.log("test/acc_epoch", self.test_acc, on_step=False, on_epoch=True)

    def test_epoch_end(self, test_step_outputs):  # args are defined as part of pl API
        dummy_input = torch.zeros(self.hparams["in_dims"], device=self.device)
        model_filename = "model_final.onnx"
        self.to_onnx(model_filename, dummy_input, export_params=True)
        artifact = wandb.Artifact(name="model.ckpt", type="model")
        artifact.add_file(model_filename)
        wandb.log_artifact(artifact)

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits, loss = self.loss(x, y)
        preds = torch.argmax(logits, 1)

        self.valid_acc(preds, y)
        self.log("valid/loss_epoch", loss)
        self.log("valid/acc_epoch", self.valid_acc)

        return logits

    def validation_epoch_end(self, validation_step_outputs):
        dummy_input = torch.zeros(self.hparams["in_dims"], device=self.device)
        model_filename = f"model_{str(self.global_step).zfill(5)}.onnx"
        torch.onnx.export(self, dummy_input, model_filename, opset_version=11)
        artifact = wandb.Artifact(name="model.ckpt", type="model")
        artifact.add_file(model_filename)
        self.logger.experiment.log_artifact(artifact)

        flattened_logits = torch.flatten(torch.cat(validation_step_outputs))
        self.logger.experiment.log(
            {
                "valid/logits": wandb.Histogram(flattened_logits.to("cpu")),
                "global_step": self.global_step,
            }
        )

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams["lr"])

    def loss(self, x, y):
        logits = self(x)
        cel = torch.nn.CrossEntropyLoss()
        loss = cel(logits, y)
        return logits, loss
