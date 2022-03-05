#!/usr/bin/env python

""""""

import argparse
import os
import subprocess
import time

import rasterio

def set_up_parser() -> argparse.ArgumentParser:
    """Set up the argument parser.

    Returns:
        the argument parser
    """
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "-i",
        "--input-fn",
        required=True,
        type=str,
        help="GeoJSON file containing annotations",
        metavar="GEOJSON",
    )
    parser.add_argument(
        "-t",
        "--target-fn",
        required=True,
        type=str,
        help="GeoTIFF file to match size with",
        metavar="GeoTIFF",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        required=True,
        type=str,
        help="directory to write output tiles to",
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

    return parser


def main(args: argparse.Namespace) -> None:
    """Data downloader script.

    Args:
        args: command-line arguments
    """
    #########################
    # Setup inputs
    #########################
    assert os.path.exists(args.input_fn)
    assert os.path.exists(args.target_fn)
    assert args.input_fn.endswith(".geojson")
    assert os.path.exists(args.output_dir) and not os.path.isfile(args.output_dir)


    base_name = os.path.basename(args.target_fn).replace('.tif','')
    output_mask_fn = os.path.join(
        args.output_dir,
        f"{base_name}_mask.tif"
    )
    if not args.overwrite:
        assert not os.path.exists(output_mask_fn)

    #########################
    # Calculate bounds
    #########################
    with rasterio.open(args.target_fn) as f:
        left, bottom, right, top = f.bounds
        width = f.width
        height = f.height

    #########################
    # Create mask
    #########################
    tic = time.time()
    command = [
        "gdal_rasterize",
        "-q",
        "-ot", "Byte",
        "-a_nodata", "0",
        "-init", "0",
        "-burn", "1",
        "-of", "GTiff",
        "-co", "TILED=YES",
        "-co", "BLOCKXSIZE=512",
        "-co", "BLOCKYSIZE=512",
        "-co", "INTERLEAVE=PIXEL",
        "-where", "\"Species='Background'\"",
        "-te", str(left), str(bottom), str(right), str(top),
        "-ts", str(width), str(height),
        "-co", "COMPRESS=LZW",
        "-co", "PREDICTOR=2",
        "-co", "BIGTIFF=YES",
        args.input_fn,
        output_mask_fn,
    ]
    subprocess.call(" ".join(command), shell=True)

    # Handles the polygons that don't have an explicit label
    command = [
        "gdal_rasterize",
        "-q",
        "-b", "1",
        "-burn", "2",
        "-where", "\"Species='Cashew'\"",
        args.input_fn,
        output_mask_fn,
    ]
    subprocess.call(" ".join(command), shell=True)

    command = [
        "gdal_rasterize",
        "-q",
        "-b", "1",
        "-burn", "3",
        "-where", "\"Species='Other'\"",
        args.input_fn,
        output_mask_fn,
    ]
    subprocess.call(" ".join(command), shell=True)
    rasterize_duration = time.time() - tic
    if args.verbose:
        print(f"Finished rasterizing labels in {rasterize_duration:0.2f} seconds")


if __name__ == "__main__":
    beg = time.time()
    parser = set_up_parser()
    args = parser.parse_args()
    main(args)
    end = time.time()
    print(__file__ + ' finished in ' + str(end-beg))
