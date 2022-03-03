import numpy as np
import rasterio
import rasterio.io
import rasterio.merge
import rasterio.windows

import torch
from torch.utils.data import Dataset, Sampler


class CustomTileDataset(Dataset):

    def __init__(self, image_fns, mask_fns, transforms=None, sanity_check=False):
        self.image_fns = image_fns
        self.mask_fns = mask_fns
        if self.mask_fns is not None:
            assert len(image_fns) == len(mask_fns)

        # Check to make sure that all the image and mask tile pairs are the same size
        # as a sanity check
        if sanity_check and mask_fns is not None:
            for image_fn, mask_fn in zip(image_fns, mask_fns):
                with rasterio.open(image_fn) as f:
                    image_height, image_width = f.shape
                with rasterio.open(mask_fn) as f:
                    mask_height, mask_width = f.shape
                assert image_height == mask_height
                assert image_width == mask_width

        self.transforms = transforms

    def __getitem__(self, index):

        i, y, x, patch_size = index
        assert 0 <= i < len(self.image_fns)

        sample = {
            "y": y,
            "x": x,
        }

        window = rasterio.windows.Window(
            x, y, patch_size, patch_size
        )

        # Load imagery
        image_fn = self.image_fns[i]
        with rasterio.open(image_fn) as f:
            image = f.read(window=window)
        sample["image"] = torch.from_numpy(image)

        # Load mask
        if self.mask_fns is not None:
            mask_fn = self.mask_fns[i]
            with rasterio.open(mask_fn) as f:
                mask = f.read(window=window)
            sample["mask"] = torch.from_numpy(mask)

        if self.transforms is not None:
            sample = self.transforms(sample)

        return sample


class RandomGeoSampler(Sampler):

    def __init__(self, image_fns, length, patch_size):
        self.tile_sample_weights = []
        self.tile_heights = []
        self.tile_widths = []
        self.length = length
        self.patch_size = patch_size

        for image_fn in image_fns:
            with rasterio.open(image_fn) as f:
                image_height, image_width = f.shape
            self.tile_sample_weights.append(image_height * image_width)
            self.tile_heights.append(image_height)
            self.tile_widths.append(image_width)

        self.tile_sample_weights = np.array(self.tile_sample_weights)
        self.tile_sample_weights = (
            self.tile_sample_weights / self.tile_sample_weights.sum()
        )
        self.num_tiles = len(self.tile_sample_weights)

    def __iter__(self):
        for _ in range(len(self)):
            i = np.random.choice(self.num_tiles, p=self.tile_sample_weights)
            y = np.random.randint(0, self.tile_heights[i] - self.patch_size)
            x = np.random.randint(0, self.tile_widths[i] - self.patch_size)

            yield (i, y, x, self.patch_size)

    def __len__(self):
        return self.length


class GridGeoSampler(Sampler):

    def __init__(
        self,
        image_fns,
        image_fn_indices,
        patch_size=256,
        stride=256,
    ):
        self.image_fn_indices = image_fn_indices
        self.patch_size = patch_size

        # tuples of the form (i, y, x, patch_size) that index into a CustomTileDataset
        self.indices = []
        for i in self.image_fn_indices:
            with rasterio.open(image_fns[i]) as f:
                height, width = f.height, f.width

            for y in list(range(0, height - patch_size, stride)) + [height - patch_size]:
                for x in list(range(0, width - patch_size, stride)) + [width - patch_size]:
                    self.indices.append((i,y,x,self.patch_size))
        self.num_chips = len(self.indices)

    def __iter__(self):
        for index in self.indices:
            yield index

    def __len__(self):
        return self.num_chips
