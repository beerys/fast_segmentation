from typing import Any, Dict, Optional, cast, List

import kornia.augmentation as K
from kornia.filters import canny, gaussian_blur2d

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
from torch import Tensor
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter  # type: ignore[attr-defined]
from torchmetrics import Accuracy, JaccardIndex, MetricCollection


from DataLoaders import CustomTileDataset, GridGeoSampler, RandomGeoSampler

cmap = matplotlib.colors.ListedColormap([
    (0,0,0,0), # Class 0, background
    (0,0,1,1), # Class 1, good water
    (1,0,0,1), # Class 2, bad water
    (0,1,0,1), # Class 3, land
])
# If you have a set of class probabilities then you can matrix multiply them with this
# matrix to get mixed colors
soft_cmap = np.array([
    (0,0,0), # Class 0, background
    (0,0,1), # Class 1, good water
    (1,0,0), # Class 2, bad water
    (0,1,0), # Class 3, land
])
# column normalize to sum to 1
soft_cmap = soft_cmap / soft_cmap.sum(axis=0, keepdims=True)
rasterio_cmap = {
    0: (  0,  0,  0,  0),
    1: (  0,  0,255,255),
    2: (255,  0,  0,255),
    3: (  0,255,  0,255),
}

def preprocess(sample: Dict[str, Any]) -> Dict[str, Any]:
    """Transform a single sample from the Dataset."""
    sample["image"] = sample["image"] / 255.0
    sample["image"] = sample["image"].float()

    if "mask" in sample:
        sample["mask"] = sample["mask"].float()

    return sample

class SegmentationTask(pl.LightningModule):

    def config_task(self) -> None:
        """Configures the task based on kwargs parameters passed to the constructor."""
        if self.hparams["segmentation_model"] == "unet":
            self.model = smp.Unet(
                encoder_name=self.hparams["encoder_name"],
                encoder_weights=self.hparams["encoder_weights"],
                in_channels=3,
                classes=4,
            )
        elif self.hparams["segmentation_model"] == "deeplabv3+":
            self.model = smp.DeepLabV3Plus(
                encoder_name=self.hparams["encoder_name"],
                encoder_weights=self.hparams["encoder_weights"],
                in_channels=3,
                classes=4,
            )
        else:
            raise ValueError(
                f"Model type '{self.hparams['segmentation_model']}' is not valid."
            )

        if self.hparams["loss"] == "ce":
            self.loss = nn.CrossEntropyLoss(  # type: ignore[attr-defined]
                ignore_index=0, reduction="mean"
            )
        elif self.hparams["loss"] == "jaccard":
            self.loss = smp.losses.JaccardLoss(mode="multiclass", classes=[1,2,3])
        elif self.hparams["loss"] == "focal":
            self.loss = smp.losses.FocalLoss(
                "multiclass", ignore_index=0, normalized=True
            )
        else:
            raise ValueError(f"Loss type '{self.hparams['loss']}' is not valid.")

    def __init__(
        self,
        **kwargs: Any,
    ) -> None:
        """Initialize the LightningModule with a model and loss function.

        Keyword Args:
            segmentation_model: Name of the segmentation model type to use
            encoder_name: Name of the encoder model backbone to use
            encoder_weights: None or "imagenet" to use imagenet pretrained weights in
                the encoder model
            loss: Name of the loss function
        """
        super().__init__()
        self.save_hyperparameters()  # creates `self.hparams` from kwargs

        self.config_task()

        self.train_augmentations = K.AugmentationSequential(
            K.RandomRotation(p=0.5, degrees=90),
            K.RandomHorizontalFlip(p=0.5),
            K.RandomVerticalFlip(p=0.5),
            K.RandomSharpness(p=0.5),
            K.ColorJitter(p=0.5, brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
            data_keys=["input", "mask"],
        )

        self.train_metrics = MetricCollection(
            [
                Accuracy(num_classes=4, ignore_index=0),
                JaccardIndex(num_classes=4, ignore_index=0),
            ],
            prefix="train_",
        )
        self.val_metrics = self.train_metrics.clone(prefix="val_")
        self.test_metrics = self.train_metrics.clone(prefix="test_")

        self.loss1 = nn.CrossEntropyLoss(ignore_index=0)
        self.loss2 = smp.losses.TverskyLoss(
            "multiclass", ignore_index=0, beta=1.0
        )

    def forward(self, x: Tensor) -> Any:  # type: ignore[override]
        """Forward pass of the model."""
        return self.model(x)

    def training_step(  # type: ignore[override]
        self, batch: Dict[str, Any], batch_idx: int
    ) -> Tensor:
        """Training step - reports average accuracy and average IoU.

        Args:
            batch: Current batch
            batch_idx: Index of current batch

        Returns:
            training loss
        """
        x = batch["image"]
        y = batch["mask"]
        with torch.no_grad():
            x, y = self.train_augmentations(x, y)
        y = y.long().squeeze()

        y_hat = self.forward(x)
        y_hat_hard = y_hat.argmax(dim=1)

        loss = self.loss1(y_hat, y)# + self.loss2(y_hat, y)

        # pixel_loss = self.loss(y_hat, y)
        # y_binary = (y.unsqueeze(1) > 0).float()
        # edges = canny(y_binary)[1]
        # sigma = 3
        # scale = 10
        # kernel_size = 2*int(3*sigma) + 1
        # weights = gaussian_blur2d(edges, kernel_size=(kernel_size,kernel_size), sigma=(sigma,sigma)) * scale
        # weights = weights.squeeze()
        # loss = torch.mean(pixel_loss * weights)


        # by default, the train step logs every `log_every_n_steps` steps where
        # `log_every_n_steps` is a parameter to the `Trainer` object
        self.log("train_loss", loss, on_step=True, on_epoch=False)
        self.train_metrics(y_hat_hard, y)

        return cast(Tensor, loss)

    def training_epoch_end(self, outputs: Any) -> None:
        """Logs epoch level training metrics.

        Args:
            outputs: list of items returned by training_step
        """
        self.log_dict(self.train_metrics.compute())
        self.train_metrics.reset()

    def validation_step(  # type: ignore[override]
        self, batch: Dict[str, Any], batch_idx: int
    ) -> None:
        """Validation step - reports average accuracy and average IoU.

        Logs the first 10 validation samples to tensorboard as images with 3 subplots
        showing the image, mask, and predictions.

        Args:
            batch: Current batch
            batch_idx: Index of current batch
        """
        x = batch["image"]
        y = batch["mask"].long().squeeze()
        y_hat = self.forward(x)
        y_hat_hard = y_hat.argmax(dim=1)

        loss = self.loss(y_hat, y)

        self.log("val_loss", loss, on_step=False, on_epoch=True)
        self.val_metrics(y_hat_hard, y)

        if batch_idx < 10:
            # Render the image, ground truth mask, and predicted mask for the first
            # image in the batch
            img = np.rollaxis(  # convert image to channels last format
                x[0].cpu().numpy(), 0, 3
            )
            mask = y[0].cpu().numpy()
            pred = y_hat_hard[0].cpu().numpy()
            fig, axs = plt.subplots(1, 3, figsize=(12, 4))
            axs[0].imshow(img)
            axs[0].axis("off")
            axs[1].imshow(mask, vmin=0, vmax=3, cmap=cmap, interpolation="none")
            axs[1].axis("off")
            axs[2].imshow(pred, vmin=0, vmax=3, cmap=cmap, interpolation="none")
            axs[2].axis("off")

            # the SummaryWriter is a tensorboard object, see:
            # https://pytorch.org/docs/stable/tensorboard.html#
            summary_writer: SummaryWriter = self.logger.experiment
            summary_writer.add_figure(
                f"image/{batch_idx}", fig, global_step=self.global_step
            )

            plt.close()

    def validation_epoch_end(self, outputs: Any) -> None:
        """Logs epoch level validation metrics.

        Args:
            outputs: list of items returned by validation_step
        """
        self.log_dict(self.val_metrics.compute())
        self.val_metrics.reset()

    def test_step(  # type: ignore[override]
        self, batch: Dict[str, Any], batch_idx: int
    ) -> None:
        """Test step identical to the validation step.

        Args:
            batch: Current batch
            batch_idx: Index of current batch
        """
        x = batch["image"]
        y = batch["mask"].long().squeeze()
        y_hat = self.forward(x)
        y_hat_hard = y_hat.argmax(dim=1)

        loss = self.loss(y_hat, y)

        # by default, the test and validation steps only log per *epoch*
        self.log("test_loss", loss, on_step=False, on_epoch=True)
        self.test_metrics(y_hat_hard, y)

    def test_epoch_end(self, outputs: Any) -> None:
        """Logs epoch level test metrics.

        Args:
            outputs: list of items returned by test_step
        """
        self.log_dict(self.test_metrics.compute())
        self.test_metrics.reset()

    def configure_optimizers(self) -> Dict[str, Any]:
        """Initialize the optimizer and learning rate scheduler.

        Returns:
            a "lr dict" according to the pytorch lightning documentation --
            https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#configure-optimizers
        """

        if self.hparams["optimizer"] == 'adam':
            optimizer = torch.optim.Adam(
                self.model.parameters(),
                lr=self.hparams["learning_rate"],
                weight_decay=self.hparams["weight_decay"]
            )
        elif self.hparams["optimizer"] == 'rmsprop':
            optimizer = torch.optim.RMSprop(
                self.model.parameters(),
                lr=self.hparams["learning_rate"],
                weight_decay=self.hparams["weight_decay"]
            )
        elif self.hparams["optimizer"] == 'sgd':
            optimizer = torch.optim.SGD(
                self.model.parameters(),
                lr=self.hparams["learning_rate"],
                weight_decay=self.hparams["weight_decay"]
            )
        elif self.hparams["optimizer"] == 'adamw':
            optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=self.hparams["learning_rate"],
                weight_decay=self.hparams["weight_decay"]
            )
        else:
            raise ValueError(f"Optimizer '{self.hparams['optimizer']}' is not supported.")

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": ReduceLROnPlateau(
                    optimizer,
                    patience=self.hparams["learning_rate_schedule_patience"],
                ),
                "monitor": "val_loss",
            },
        }


class SegmentationDataModule(pl.LightningDataModule):

    def __init__(
        self,
        image_fns: Dict[str, List[str]],
        mask_fns: Dict[str, List[str]],
        batch_size: int = 64,
        patch_size: int = 256,
        num_workers: int = 4,
        train_batches_per_epoch=512,
        valid_batches_per_epoch=32,
        **kwargs: Any,
    ) -> None:
        super().__init__()  # type: ignore[no-untyped-call]
        self.image_fns = image_fns
        self.mask_fns = mask_fns
        self.batch_size = batch_size
        self.patch_size = patch_size
        self.num_workers = num_workers
        self.train_patches_per_epoch = train_batches_per_epoch * batch_size
        self.valid_patches_per_epoch = valid_batches_per_epoch * batch_size

    def setup(self, stage: Optional[str] = None) -> None:
        """Initialize the main ``Dataset`` objects.

        This method is called once per GPU per run.
        """
        train_transforms = preprocess
        val_test_transforms = preprocess

        self.train_dataset = CustomTileDataset(
            self.image_fns['train'],
            self.mask_fns['train'],
            transforms=train_transforms,
            sanity_check=True
        )

        self.val_dataset = CustomTileDataset(
            self.image_fns['valid'],
            self.mask_fns['valid'],
            transforms=val_test_transforms,
            sanity_check=True
        )

        self.test_dataset = CustomTileDataset(
            self.image_fns['test'],
            self.mask_fns['test'],
            transforms=val_test_transforms,
            sanity_check=True
        )

    def train_dataloader(self) -> DataLoader[Any]:
        """Return a DataLoader for training."""

        sampler = RandomGeoSampler(
            self.image_fns['train'], self.train_patches_per_epoch, self.patch_size
        )

        return DataLoader(
            self.train_dataset,
            sampler=sampler,
            batch_size=self.batch_size,
            num_workers=self.num_workers
        )

    def val_dataloader(self) -> DataLoader[Any]:
        """Return a DataLoader for validation."""
        sampler = RandomGeoSampler(
            self.image_fns['valid'], self.valid_patches_per_epoch, self.patch_size
        )

        return DataLoader(
            self.val_dataset,
            sampler=sampler,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )

    def test_dataloader(self) -> DataLoader[Any]:
        """Return a DataLoader for testing."""
        sampler = GridGeoSampler(
            self.image_fns['test'], list(range(len(self.image_fns['test']))), 640, 640
        )

        return DataLoader(
            self.test_dataset,
            sampler=sampler,
            batch_size=16,
            num_workers=self.num_workers,
        )
