import pathlib
from datetime import datetime

import torch
from torch import Tensor
from matplotlib import pyplot as plt
import pandas as pd
from matplotlib.figure import Figure
from einops import rearrange
import rasterio

from .geo import RasterDataset
from typing import List
from .utils import (
    GeoSlice,
    Path,
    Sample,
    disambiguate_timestamp
)


class ESDQuantizer():
    r"""Decode ESD-encoded quantized indices into continuous embedding vectors.

    The ESDQuantizerDecoder converts integer quantization indices produced by an
    ESD quantizer into continuous values in the range [-1, 1], representing
    multi-level embeddings of the original input. This enables downstream tasks,
    such as visualization, machine learning, or spatial analysis, to operate
    directly on decoded embeddings without reconstructing the full raw data.

    Key points:

    * Factorized decoding: Each index is split into multiple levels according
      to the quantizer configuration.
    * Continuous mapping: Level indices are rescaled and centered to [-1, 1],
      preserving relative distances in embedding space.
    * Fully vectorized: The decoding is performed on entire tensors at once,
      avoiding slow per-pixel loops and enabling GPU acceleration.
    * Flexible input: Supports arbitrary batch sizes and spatial dimensions
      (..., H, W).

    Usage:

    .. code-block:: python

        decoder = ESDQuantizerDecoder()
        decoded = decoder.apply_transform(torch.from_numpy(ESD_codes.astype(np.int32)))

    .. note::
        The output retains the channel dimension corresponding to embedding levels.
        Users can further convert embeddings to visualizations or aggregate them
        for downstream tasks.
    
    """
    def __init__(
        self,
        levels: List[int] = [8, 8, 8, 5, 5, 5],
    ) -> None:

        levels_tensor = torch.tensor(levels, dtype=torch.int32)
        self._levels = levels_tensor

        basis_tensor = torch.cumprod(
            torch.tensor([1] + levels[:-1], dtype=torch.int32),
            dim=0
        )
        self._basis = basis_tensor

    @staticmethod
    def _exists(v) -> bool:
        return v is not None

    def indices_to_codes(self, indices: Tensor) -> Tensor:
        assert self._exists(indices)

        indices = rearrange(indices, '... -> ... 1')
        level_indices = (indices // self._basis) % self._levels

        half = self._levels // 2
        codes = (level_indices - half) / half

        return codes

    def quantize(
        self,
        input: Tensor,
    ) -> Tensor:
        return self.indices_to_codes(input).movedim(-1, -3)

class EmbeddedSeamlessData(RasterDataset):
    r"""Embedded Seamless Data (ESD).

    The `Embedded Seamless Data (ESD) <https://arxiv.org/abs/2601.11183>`
    is a global, analysis-ready Earth embedding dataset at 30-meter resolution,
    designed to overcome the computational and storage challenges of planetary-scale
    Earth system science. By transforming multi-sensor satellite observations into
    compact, quantized latent vectors, ESD reduces the original data volume (~1 PB for
    a full year of global land surfaces) to approximately 2.4 TB, enabling decadal-scale
    analysis on standard workstations.

    Key features:

    * **Longitudinal Consistency**: Provides a continuous record from 2000 to 2024,
      harmonized from Landsat 5, 7, 8, 9 and MODIS Terra imagery.
    * **High Reconstructive Fidelity**: Achieves a Mean Absolute Error (MAE) of 0.013
      across six spectral bands, ensuring the embeddings retain physically meaningful
      surface information.
    * **Semantic Intelligence**: Captures complex land surface patterns, outperforming
      raw sensor fusion data for land-cover classification (global accuracy 79.74%).
    * **Implicit Denoising**: Filters transient noise such as clouds and shadows via
      the ESDNet architecture, producing clean signals suitable for temporal and
      environmental monitoring.
    * **Few-Shot Proficiency**: Supports robust learning with minimal labeled data,
      ideal for regions with scarce ground-truth measurements.
    * **Compact and Vectorized**: Each 30-meter pixel is represented by a
      high-dimensional embedding vector, which can be aggregated, compared, or analyzed
      efficiently without reconstructing raw imagery.

    The dataset covers terrestrial land surfaces, shallow waters, intertidal and reef
    zones, inland waterways, and coastal regions. Polar coverage is limited by satellite
    orbits and sensor availability.

    Produced by the ESDNet framework, ESD provides an ultra-lightweight, globally
    consistent representation of surface conditions, enabling flexible, high-resolution
    analysis of land surface dynamics over decades.

    If you use this dataset in your research, please cite:

    * Chen, S. et al. (2026). "Democratizing planetary-scale analysis: An ultra-lightweight
      Earth embedding database for accurate and flexible global land monitoring." 
      arXiv preprint arXiv:2601.11183. <https://arxiv.org/abs/2601.11183>

    .. note::
       The dataset and code are available at
       `GitHub: shuangchencc/ESD <https://github.com/shuangchencc/ESD>`.
       dataset : https://data-starcloud.pcl.ac.cn/iearthdata/64

    .. versionadded:: 0.9
    """
    quantizer = ESDQuantizer()

    def __getitem__(self, index: GeoSlice) -> Sample:
        """Retrieve input, target, and/or metadata indexed by spatiotemporal slice.

        Args:
            index: [xmin:xmax:xres, ymin:ymax:yres, tmin:tmax:tres] coordinates to index.

        Returns:
            Sample of input, target, and/or metadata at that index.

        Raises:
            IndexError: If *index* is not found in the dataset.
        """
        x, y, t = self._disambiguate_slice(index)
        interval = pd.Interval(t.start, t.stop)
        df = self.index.iloc[self.index.index.overlaps(interval)]
        df = df.iloc[:: t.step]
        df = df.cx[x.start : x.stop, y.start : y.stop]

        if df.empty:
            raise IndexError(
                f'index: {index} not found in dataset with bounds: {self.bounds}'
            )

        if self.separate_files:
            data_list: list[Tensor] = []
            for band in self.bands:
                band_filepaths = []
                for filepath in df.filepath:
                    filepath = self._update_filepath(band, filepath)
                    band_filepaths.append(filepath)
                data_list.append(self._merge_files(band_filepaths, index))
            data = torch.cat(data_list)
        else:
            data = self._merge_files(df.filepath, index, self.band_indexes)

        transform = rasterio.transform.from_origin(x.start, y.stop, x.step, y.step)
        sample: Sample = {
            'bounds': self._slice_to_tensor(index),
            'transform': torch.tensor(transform),
        }

        data = data.to(self.dtype)
        if self.is_image:
            sample['image'] = data
        else:
            sample['mask'] = data.squeeze(0)
          
        # ESD quantize
        sample['image'] = self.quantizer.quantize(sample['image'])

        if self.transforms is not None:
            sample = self.transforms(sample)

        return sample

    def _filepath_to_timestamp(self, filepath: Path) -> tuple[datetime, datetime]:
        """Extract minimum and maximum timestamps from the filepath.

        Args:
            filepath: Full path to the file.

        Returns:
            (mint, maxt) tuple.
        """
        # Example file paths:
        # * SDC30_EBD_V001: 2024/SDC30_EBD_V001_02VMN_2024.tif

        date_format = '%Y'

        # Iterate over path components from leaf (filename) to root
        for part in pathlib.Path(filepath).parts[::-1]:
            try:
                mint, maxt = disambiguate_timestamp(part, date_format)
                return mint, maxt
            except ValueError:
                continue
        return self.mint, self.maxt
    
    def plot(
        self,
        sample: Sample,
        show_titles: bool = True,
        suptitle: str | None = None,
    ) -> Figure:
        """Visualize ESD embeddings as an RGB image using PyTorch.

        Valid pixels are determined by checking for non-zero vectors.

        Args:
            vectors: ESD embedding tensor of shape (months, channels, height, width)
            show_titles: whether to show a title above the image
            suptitle: optional figure-level title

        Returns:
            A matplotlib Figure containing the RGB(A) visualization.
        """
        vectors = sample['image']
        print(vectors.shape)
        months, channels, H, W = vectors.shape

        # Compute valid mask: any non-zero pixel across channels
        valid_mask = (~torch.isclose(vectors, torch.tensor(0., device=vectors.device), atol=1e-6))
        valid_mask = valid_mask[:12].any(dim=1).any(dim=0)  # combine first 12 months, shape (H, W)

        # Reduce channels to RGB using mean over selected channels
        R = (vectors[:, 5].mean(dim=0) + 1) / 2  # normalize to [0,1]
        G = (vectors[:, 1].mean(dim=0) + 1) / 2
        B = (vectors[:, 2].mean(dim=0) + 1) / 2

        # Clamp to [0,1] and convert to uint8
        disp_img = torch.zeros(H, W, 4, dtype=torch.uint8, device='cpu')
        disp_img[..., 0] = (R.clamp(0, 1) * 255).to(torch.uint8).cpu()
        disp_img[..., 1] = (G.clamp(0, 1) * 255).to(torch.uint8).cpu()
        disp_img[..., 2] = (B.clamp(0, 1) * 255).to(torch.uint8).cpu()
        disp_img[..., 3] = valid_mask.to(torch.uint8) * 255  # alpha channel

        # Plot
        fig, ax = plt.subplots()
        ax.imshow(disp_img.cpu().numpy())
        ax.axis('off')

        if show_titles:
            ax.set_title("ESD Embedding Visualization")

        if suptitle is not None:
            plt.suptitle(suptitle)

        return fig
