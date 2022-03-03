#!/usr/bin/env python

""""""
import time
import argparse
import os
from unittest.mock import patch

import numpy as np
import rasterio

from DataLoaders import CustomTileDataset, GridGeoSampler
from trainers import SegmentationTask, preprocess, soft_cmap, rasterio_cmap

import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
import tqdm

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

BATCH_SIZE = 2
PATCH_SIZE = 2048
PADDING = 128
assert PADDING % 2 == 0
HALF_PADDING = PADDING // 2
STRIDE = PATCH_SIZE - PADDING
NUM_WORKERS = 6

def set_up_parser() -> argparse.ArgumentParser:
    """Set up the argument parser.

    Returns:
        the argument parser
    """
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--input-model-checkpoint",
        required=True,
        type=str,
        help="model checkpoint (.ckpt format)",
        metavar="CKPT",
    )
    parser.add_argument(
        "--input-image-fn",
        required=True,
        type=str,
        help="input imagery as a geotiff",
        metavar="GEOTIFF"
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        required=True,
        type=str,
        help="directory to write prediction tiles to",
    )
    parser.add_argument(
        "--output-fn",
        required=False,
        default=None,
        type=str,
        help="filename to write prediction tiles to (defaults to name of input file)",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="overwrites the output tiles if they exist",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="print stuff",
    )
    parser.add_argument(
        "--save-soft",
        action="store_true",
        help="save the soft predictions as well",
    )
    parser.add_argument(
        "--gpu",
        required=False,
        type=int,
        help="GPU id to use for inference, CPU is used if not set",
    )

    return parser


def main(args: argparse.Namespace) -> None:
    """Data downloader script.

    Args:
        args: command-line arguments
    """
    #########################
    # Setup inputs
    #########################
    assert os.path.exists(args.input_model_checkpoint)
    assert args.input_model_checkpoint.endswith(".ckpt")
    assert os.path.exists(args.input_image_fn)
    assert args.input_image_fn.endswith(".tif")

    os.makedirs(args.output_dir, exist_ok=True)

    if args.output_fn is None:
        output_soft_predictions_fn = os.path.join(
            args.output_dir,
            os.path.basename(args.input_image_fn).replace(".tif", "_predictions-soft.tif")
        )
        output_hard_predictions_fn = os.path.join(
            args.output_dir,
            os.path.basename(args.input_image_fn).replace(".tif", "_predictions.tif")
        )
    else:
        assert ".tif" in args.output_fn
        output_soft_predictions_fn = os.path.join(
            args.output_dir,
            args.output_fn.replace(".tif", "_predictions-soft.tif")
        )
        output_hard_predictions_fn = os.path.join(
            args.output_dir,
            args.output_fn.replace(".tif", "_predictions.tif")
        )
    if not args.overwrite:
        assert not os.path.exists(output_hard_predictions_fn)
        if args.save_soft:
            assert not os.path.exists(output_soft_predictions_fn)

    device = torch.device(
        f"cuda:{args.gpu}"
        if (args.gpu is not None) and torch.cuda.is_available() else
        "cpu"
    )

    #########################
    # Load task and data
    #########################
    tic = time.time()
    task = SegmentationTask.load_from_checkpoint(args.input_model_checkpoint)
    task.freeze()
    model = task.model
    model = model.eval().to(device)

    dataset = CustomTileDataset([args.input_image_fn], None, transforms=preprocess)
    sampler = GridGeoSampler(
        [args.input_image_fn], image_fn_indices=[0], patch_size=PATCH_SIZE, stride=STRIDE
    )
    dataloader = DataLoader(
        dataset,
        sampler=sampler,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS
    )
    if args.verbose:
        print(
            "Finished loading checkpoint and setting up dataset in"
            f" {time.time()-tic:0.2f} seconds"
        )

    #########################
    # Run inference
    #########################
    tic = time.time()
    with rasterio.open(args.input_image_fn) as f:
        input_height, input_width = f.shape
        profile = f.profile

    if args.verbose:
        print(f"Input size: {input_height} x {input_width}")
    assert PATCH_SIZE <= input_height
    assert PATCH_SIZE <= input_width
    output = np.zeros((4, input_height, input_width), dtype=np.float32)
    kernel = np.ones((PATCH_SIZE, PATCH_SIZE), dtype=np.float32)
    kernel[HALF_PADDING:-HALF_PADDING, HALF_PADDING:-HALF_PADDING] = 5
    counts = np.zeros((input_height, input_width), dtype=np.float32)

    for batch in tqdm.tqdm(dataloader):
        images = batch["image"].to(device)
        y_coords = batch["y"]
        x_coords = batch["x"]

        with torch.no_grad():
            t_batch_output = model(images)
            t_batch_output = F.softmax(t_batch_output, dim=1).cpu().numpy()

        for t_output, y, x in zip(t_batch_output, y_coords, x_coords):
            output[:, y:y+PATCH_SIZE, x:x+PATCH_SIZE] += t_output * kernel
            counts[y:y+PATCH_SIZE, x:x+PATCH_SIZE] += kernel

    #output = output / counts
    output[0,:,:] = 0
    #output = output / output.sum(axis=0, keepdims=True)
    output_hard = output.argmax(axis=0).astype(np.uint8)
    if args.verbose:
        print(f"Finished running model in {time.time()-tic:0.2f} seconds")

    #########################
    # Save predictions
    #########################
    tic = time.time()
    profile["count"] = 1
    profile["dtype"] = "uint8"
    profile["compress"] = "lzw"
    profile["predictor"] = 2
    profile["nodata"] = 0
    with rasterio.open(output_hard_predictions_fn, "w", **profile) as f:
        f.write(output_hard, 1)
        f.write_colormap(1, rasterio_cmap)

    if args.save_soft:
        # Convert the predictions probabilities to RGB colors
        output = np.rollaxis(output, 0, 3)
        output = output @ soft_cmap
        output = (output * 255).astype(np.uint8)
        output = np.rollaxis(output, 2, 0)

        profile["count"] = 3
        profile["photometric"] = "RGB"
        del profile["nodata"]

        with rasterio.open(output_soft_predictions_fn, "w", **profile) as f:
            f.write(output)
    if args.verbose:
        print(f"Finished saving predictions in {time.time()-tic:0.2f} seconds")

if __name__ == "__main__":
    parser = set_up_parser()
    args = parser.parse_args()
    main(args)
