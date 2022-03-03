import rasterio
import numpy as np
import cv2
import argparse


def apply_distance_buffer(input_path, output_path, background_class_label,
        target_class_label, buffer_size):

    with rasterio.open(input_path) as f:
        mask = f.read().squeeze()
        mask_profile = f.profile

    nodata_mask = (mask != target_class_label).astype(np.uint8)


    transform = cv2.distanceTransform(
        nodata_mask, distanceType=cv2.DIST_L2, maskSize=3
    )


    # buffer size is in meters -- this is in units of the CRS
    background_mask = (transform > 0) & (transform < buffer_size)


    mask[background_mask] = target_class_label


    with rasterio.open(output_path, "w", **mask_profile) as f:
        f.write(mask, 1)

def parse_args():

    parser = argparse.ArgumentParser(description='Basic statistics on tfrecord files')

    parser.add_argument('--input_path', dest='input_path',
                        required=True)
    parser.add_argument('--output_path', dest='output_path',
                        required=True)
    parser.add_argument('--background_class_label', 
                        dest='background_class_label',
                        default=1)
    parser.add_argument('--target_class_label',
                        dest='target_class_label',
                        default=3)
    parser.add_argument('--buffer_size',
                        dest='buffer_size',
                        default=3)


    parsed_args = parser.parse_args()

    return parsed_args

def main():
    parsed_args = parse_args()

    apply_distance_buffer(parsed_args.input_path, parsed_args.output_path,
                          parsed_args.background_class_label, 
                          parsed_args.target_class_label,
                          parsed_args.buffer_size)

if __name__ == '__main__':
    main()
