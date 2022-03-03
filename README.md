# Building segmentation hackathon

## Visualizer links

- http://13.95.159.165/tool/

## Setup

```
conda config --set channel_priority strict
conda env create --file environment.yml
conda activate torchgeo
```

## Pipeline steps

- Upload geojson labels to `data/labels/`
- Convert labels from EPSG:4326 (lat/lon) to the coordinate system (CRS) of the imagery, in this case, EPSG:32616
  - `ogr2ogr -of GeoJSON -t_srs EPSG:32616 demo_annotations_epsg32616.geojson demo_annotations.geojson`
- Create masks
  - `python create_mask_from_annotations.py --input-fn data/labels/demo_annotations_epsg32616.geojson --target-fn data/imagery/16_pre_imagery_cropped.tif --output-dir data/masks/ --overwrite`
  - NOTE: `create_mask_from_annotations.py` will need to be edited with the class names used in the web-tool
- Buffer the masks
  - Run `Apply distance buffer to mask.ipynb`
- Train models
  - Run `Train.ipynb`
- Inference
  - `python inference.py --input-model-checkpoint output/runs/unet-resnet18-imagenet-lr_0.001/last.ckpt --input-image-fn data/imagery/16_pre_imagery_cropped.tif --output-dir predictions/unet-resnet18-imagenet-lr_0.001/ --overwrite --gpu 1`
