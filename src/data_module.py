import pytorch_lightning as pl
import torchvision
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import CIFAR10


class CIFAR10DataModule(pl.LightningDataModule):
    def __init__(
        self,
        transformations: list,
        data_dir: str = "../data",
        batch_size: int = 64,
        num_workers: int = 4,
    ):
        super().__init__()
        self.transform = torchvision.transforms.Compose(transformations)
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers

    def prepare_data(self):
        CIFAR10(self.data_dir, train=True, download=True)
        CIFAR10(self.data_dir, train=False, download=True)

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            cifar10 = CIFAR10(self.data_dir, train=True, transform=self.transform)
            self.cifar_train, self.cifar_val = random_split(cifar10, [45000, 5000])
        if stage == "test" or stage is None:
            self.cifar_test = CIFAR10(
                self.data_dir, train=False, transform=self.transform
            )

    def train_dataloader(self):
        return DataLoader(
            self.cifar_train, batch_size=self.batch_size, num_workers=self.num_workers
        )

    def val_dataloader(self):
        return DataLoader(
            self.cifar_val,
            batch_size=10 * self.batch_size,
            num_workers=self.num_workers,
        )

    def test_dataloader(self):
        return DataLoader(
            self.cifar_test,
            batch_size=10 * self.batch_size,
            num_workers=self.num_workers,
        )
