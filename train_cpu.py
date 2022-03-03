import os
import subprocess

import numpy as np
import matplotlib.pyplot as plt

from trainers import SegmentationTask, SegmentationDataModule
import pytorch_lightning as pl

from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

image_fns = [
    "data/imagery/16_pre_imagery_cropped.tif",
]

mask_fns = [
    "data/masks/16_pre_imagery_cropped_mask_buffered.tif",
]

dm = SegmentationDataModule(
    image_fns={"train": image_fns, "valid": image_fns, "test": image_fns},
    mask_fns={"train": mask_fns, "valid": mask_fns, "test": mask_fns},
    batch_size=24,
    patch_size=512,
    num_workers=6,
    batches_per_epoch=256,
)

task = SegmentationTask(
    segmentation_model="unet",
    encoder_name="resnet18",
    encoder_weights="imagenet", # use None for random weight init
    loss="ce",
    learning_rate=0.001,
    learning_rate_schedule_patience=6,
    optimizer="adamw",
    weight_decay=0.01,
)

log_dir = "output/logs/"
output_dir = "output/runs/"
experiment_name = "unet-resnet18-imagenet-lr_0.001"
experiment_dir = os.path.join(output_dir, experiment_name)

tb_logger = pl_loggers.TensorBoardLogger(log_dir, name=experiment_name)

checkpoint_callback = ModelCheckpoint(
    monitor="val_loss",
    dirpath=experiment_dir,
    save_top_k=12,
    save_last=True,
)
early_stopping_callback = EarlyStopping(
    monitor="val_loss",
    min_delta=0.00,
    patience=18,
)

trainer_args = {
    "callbacks": [checkpoint_callback, early_stopping_callback],
    "logger": tb_logger,
    "default_root_dir": experiment_dir,
    "max_epochs": 15,
}

trainer = pl.Trainer(**trainer_args)

trainer.fit(model=task, datamodule=dm)
