import math
import typing
from typing import Dict, List, Optional, Tuple, Union, Any

import zarr
from ome_zarr.format import CurrentFormat
from ome_zarr.io import parse_url
from ome_zarr.scale import Scaler
from ome_zarr.writer import write_image, write_multiscale

from .. import types
from ..metadata import utils
from ..utils import io_utils

# from .writer import Writer


def _compute_scales(
    scale_num_levels: int,
    scale_factor: Tuple[float, float, float],
    pixelsizes: Tuple[float, float, float],
    chunks: Tuple[int, int, int, int, int],
    data_shape: Tuple[int, int, int, int, int],
    translation: Tuple[float, float, float],
    method: str = "nearest",
) -> Tuple[List, List, Scaler]:
    """Generate the list of coordinate transformations and associated chunk options.

    Parameters
    ----------
    scale_num_levels: the number of downsampling levels
    scale_factor: a tuple of scale factors in each spatial dimension (Z, Y, X)
    pixelsizes: a list of pixel sizes in each spatial dimension (Z, Y, X)
    chunks: a 5D tuple of integers with size of each chunk dimension (T, C, Z, Y, X)
    data_shape: a 5D tuple of the full resolution image's shape
    method: scaler method, either "nearest" or "precomputed". Use "precomputed"
            if writing a precomputed pyramid

    Returns
    -------
    A tuple of the coordinate transforms, chunk options, and the Scaler instance
    """
    transforms = [
        [
            # the voxel size for the first scale level
            {
                "type": "scale",
                "scale": [
                    1.0,
                    1.0,
                    pixelsizes[0],
                    pixelsizes[1],
                    pixelsizes[2],
                ],
            },
            {
                "type": "translation",
                "translation": translation
            }
        ]
    ]
    chunk_sizes = []
    lastz = data_shape[2]
    lasty = data_shape[3]
    lastx = data_shape[4]
    opts = dict(
        chunks=(
            1,
            1,
            min(lastz, chunks[2]),
            min(lasty, chunks[3]),
            min(lastx, chunks[4]),
        )
    )
    chunk_sizes.append(opts)
    # TODO scaler might want to use different method for segmentations than raw
    # TODO control how many levels of zarr are created
    if scale_num_levels > 1:
        scaler = Scaler()
        scaler.method = method
        scaler.max_layer = scale_num_levels - 1
        # choose the largest factor, generally either all factors are the same or Z is 1.
        if scaler.method == "nearest":
            scale_factor = list(scale_factor)
            scale_factor[0] = 1.0
        scaler.downscale = max(scale_factor) if scale_factor is not None else 2
        for i in range(scale_num_levels - 1):
            last_transform = transforms[-1][0]
            last_scale = typing.cast(List, last_transform["scale"])
            transforms.append(
                [
                    {
                        "type": "scale",
                        "scale": [
                            1.0,
                            1.0,
                            last_scale[2] * scale_factor[0],
                            last_scale[3] * scale_factor[1],
                            last_scale[4] * scale_factor[2],
                        ],
                    },
                    {
                        "type": "translation",
                        "translation": translation
                    }
                ]
            )
            lastz = int(math.ceil(lastz / scale_factor[0]))
            lasty = int(math.ceil(lasty / scale_factor[1]))
            lastx = int(math.ceil(lastx / scale_factor[2]))
            opts = dict(chunks=(
                1,
                1,
                min(lastz, chunks[2]),
                min(lasty, chunks[3]),
                min(lastx, chunks[4])))
            chunk_sizes.append(opts)
    else:
        scaler = None

    return transforms, chunk_sizes, scaler


def _get_axes_5d(time_unit="millisecond", space_unit="micrometer"):
    axes_5d = [
        {"name": "t", "type": "time", "unit": f"{time_unit}"},
        {"name": "c", "type": "channel"},
        {"name": "z", "type": "space", "unit": f"{space_unit}"},
        {"name": "y", "type": "space", "unit": f"{space_unit}"},
        {"name": "x", "type": "space", "unit": f"{space_unit}"},
    ]
    return axes_5d


def _ensure_storage_options(storage_options):
    if storage_options is None:
        storage_options = {}
    return storage_options


def _ensure_pixel_sizes(physical_pixel_sizes):
    if physical_pixel_sizes is None:
        pixelsizes = (1.0, 1.0, 1.0)
    else:
        pixelsizes = (
            physical_pixel_sizes.Z if physical_pixel_sizes.Z is not None else 1.0,
            physical_pixel_sizes.Y if physical_pixel_sizes.Y is not None else 1.0,
            physical_pixel_sizes.X if physical_pixel_sizes.X is not None else 1.0,
        )
    return pixelsizes


def _ensure_translation(translation, data):
    if translation is None:
        translation = [0,] * data.ndim
    if not len(translation) == data.ndim:
        raise ValueError("Length of translation vector must match data dimensions.")
    return translation


def _ensure_chunks(chunks, data, target_chunk_size=64):
    if chunks is None:
        plane_size = data.shape[3] * data.shape[4] * data.itemsize
        # convert to bytes
        target_chunk_size_bytes = target_chunk_size * (1024 * 1024)
        nplanes_per_chunk = int(math.ceil(target_chunk_size_bytes / plane_size))
        nplanes_per_chunk = min(nplanes_per_chunk, data.shape[2])
        chunks = (
            1,
            1,
            nplanes_per_chunk,
            data.shape[3],
            data.shape[4],
        )
    return chunks


def _ensure_channel_colors(channel_colors, data):
    if channel_colors is None:
        # TODO generate proper colors or confirm that the underlying lib can handle
        # None
        channel_colors = [i for i in range(data.shape[1])]
    return channel_colors


def _ensure_channel_names(channel_names, data, image_name):
    # TODO this isn't generating a very pretty looking name but it will be
    # unique
    if channel_names is None:
        channel_names = [
            utils.generate_ome_channel_id(image_id=image_name, channel_id=i)
            for i in range(data.shape[1])
        ]
    return channel_names


class OmeZarrWriter:
    def __init__(self, uri: types.PathLike):
        """
        Constructor.

        Parameters
        ----------
        uri: types.PathLike
            The URI or local path for where to save the data.
        """
        # Resolve final destination
        fs, path = io_utils.pathlike_to_fs(uri)

        # Save image to zarr store!
        self.store = parse_url(uri, mode="w").store
        self.root_group = zarr.group(store=self.store)

    @staticmethod
    def build_ome(
        data_shape: Tuple[int, ...],
        image_name: str,
        channel_names: List[str],
        channel_colors: List[int],
        channel_minmax: List[Tuple[float, float]],
    ) -> Dict:
        """
        Create the necessary metadata for an OME tiff image

        Parameters
        ----------
        data_shape:
            A 5-d tuple, assumed to be TCZYX order
        image_name:
            The name of the image
        channel_names:
            The names for each channel
        channel_colors:
            List of all channel colors
        channel_minmax:
            List of all (min, max) pairs of channel intensities

        Returns
        -------
        Dict
            An "omero" metadata object suitable for writing to ome-zarr
        """
        ch = []
        for i in range(data_shape[1]):
            ch.append(
                {
                    "active": True,
                    "coefficient": 1,
                    "color": f"{channel_colors[i]:06x}",
                    "family": "linear",
                    "inverted": False,
                    "label": channel_names[i],
                    "window": {
                        "end": float(channel_minmax[i][1]),
                        "max": float(channel_minmax[i][1]),
                        "min": float(channel_minmax[i][0]),
                        "start": float(channel_minmax[i][0]),
                    },
                }
            )

        omero = {
            "id": 1,  # ID in OMERO
            "name": image_name,  # Name as shown in the UI
            "version": "0.4",  # Current version
            "channels": ch,
            "rdefs": {
                "defaultT": 0,  # First timepoint to show the user
                "defaultZ": data_shape[2] // 2,  # First Z section to show the user
                "model": "color",  # "color" or "greyscale"
            },
            # TODO: can we add more metadata here?
            # # from here down this is all extra and not part of the ome-zarr spec
            # "meta": {
            #     "projectDescription": "20+ lines of gene edited cells etc",
            #     "datasetName": "aics_hipsc_v2020.1",
            #     "projectId": 2,
            #     "imageDescription": "foo bar",
            #     "imageTimestamp": 1277977808.0,
            #     "imageId": 12,
            #     "imageAuthor": "danielt",
            #     "imageName": "AICS-12_143.ome.tif",
            #     "datasetDescription": "variance dataset after QC",
            #     "projectName": "aics cell variance project",
            #     "datasetId": 3
            # },
            # no longer needed as this is captured elsewhere?
            # or is this still a convenience for the 3d viewer?
            # "size": {
            #     "width": shape[4],
            #     "c": shape[1],
            #     "z": shape[2],
            #     "t": shape[0],
            #     "height": shape[3]
            # },
        }
        return omero

    def write_multiscale(
        self,
        pyramid: List,
        image_name: str,
        physical_pixel_sizes: Optional[types.PhysicalPixelSizes],
        translation: Optional[List[float]],
        channel_names: Optional[List[str]],
        channel_colors: Optional[List[int]],
        scale_factor: Tuple[float, float, float] = (2.0, 2.0, 2.0),
        chunks: Optional[tuple] = None,
        storage_options: Optional[Dict] = None,
        compute_dask: bool = False,
        **metadata: Union[str, Dict[str, Any], List[Dict[str, Any]]]
    ) -> List:

        image_data = pyramid[0]
        storage_options = _ensure_storage_options(storage_options)
        channel_names = _ensure_channel_names(channel_names, image_data, image_name)
        channel_colors = _ensure_channel_colors(channel_colors, image_data)
        chunks = _ensure_chunks(chunks, image_data, target_chunk_size=64)
        pixelsizes = _ensure_pixel_sizes(physical_pixel_sizes)
        translation = _ensure_translation(translation, image_data)

        # try to construct per-image metadata
        ome_json = OmeZarrWriter.build_ome(
            image_data.shape,
            image_name,
            channel_names=channel_names,  # type: ignore
            channel_colors=channel_colors,  # type: ignore
            # This can be slow if computed here.
            # TODO: Rely on user to supply the per-channel min/max.
            channel_minmax=[(0.0, 1.0) for i in range(image_data.shape[1])],
        )
        # TODO user supplies units?
        axes_5d = _get_axes_5d()

        transforms, chunk_opts, scaler = _compute_scales(
            len(pyramid),
            scale_factor,
            pixelsizes,
            chunks,
            image_data.shape,
            translation,
            method="precomputed",
        )
        for opt in chunk_opts:
            opt.update(storage_options)

        # TODO image name must be unique within this root group
        group = self.root_group.create_group(image_name, overwrite=True)
        group.attrs["omero"] = ome_json
        jobs = write_multiscale(
            pyramid,
            group=group,
            fmt=CurrentFormat(),
            axes=axes_5d,
            coordinate_transformations=transforms,
            storage_options=chunk_opts,
            name=None,
            compute_dask=compute_dask,
            **metadata
        )

        return jobs

    def write_image(
        self,
        image_data: types.ArrayLike,  # each ArrayLike must be 5D TCZYX
        image_name: str,
        physical_pixel_sizes: Optional[types.PhysicalPixelSizes],
        translation: Optional[List[float]],
        channel_names: Optional[List[str]],
        channel_colors: Optional[List[int]],
        scale_num_levels: int = 1,
        scale_factor: float = 2.0,
        chunks: Optional[tuple] = None,
        storage_options: Optional[Dict] = None,
    ) -> None:
        """
        Write a data array to a file.

        Parameters
        ----------
        image_data: types.ArrayLike
            The array of data to store. Data arrays must have 2 to 6 dimensions. If a
            list is provided, then it is understood to be multiple images written to the
            ome-tiff file. All following metadata parameters will be expanded to the
            length of this list.
        image_name: str
            string representing the name of the image
        physical_pixel_sizes: Optional[types.PhysicalPixelSizes]
            PhysicalPixelSizes object representing the physical pixel sizes in Z, Y, X
            in microns.
            Default: None
        channel_names: Optional[List[str]]
            Lists of strings representing the names of the data channels
            Default: None
            If None is given, the list will be generated as a 0-indexed list of strings
            of the form "Channel:image_index:channel_index"
        channel_colors: Optional[List[int]]
            List of rgb color values per channel or a list of lists for each image.
            These must be values compatible with the OME spec.
            Default: None
        scale_num_levels: Optional[int]
            Number of pyramid levels to use for the image.
            Default: 1 (represents no downsampled levels)
        scale_factor: Optional[Tuple[float]]
            The scale factors to use for each spatial dimension the image,
            in Z,Y,X order.
            Only active if scale_num_levels > 1.
            If "nearest" scaling is used (default), the first scale factor is overridden to 1.0
            Default: (1.0, 2.0, 2.0)
        storage_options: Optional[Dict]
            Options to pass to the zarr storage backend, e.g., "compressor"
            Default: None

        Examples
        --------
        Write a TCZYX data set to OME-Zarr

        >>> image = numpy.ndarray([1, 10, 3, 1024, 2048])
        ... writer = OmeZarrWriter("/path/to/file.ome.zarr")
        ... writer.write_image(image)

        Write multi-scene data to OME-Zarr, specifying channel names

        >>> image0 = numpy.ndarray([3, 10, 1024, 2048])
        ... image1 = numpy.ndarray([3, 10, 512, 512])
        ... writer = OmeZarrWriter("/path/to/file.ome.zarr")
        ... writer.write_image(image0, "Image:0", ["C00","C01","C02"])
        ... writer.write_image(image1, "Image:1", ["C10","C11","C12"])
        """
        storage_options = _ensure_storage_options(storage_options)
        channel_names = _ensure_channel_names(channel_names, image_data, image_name)
        channel_colors = _ensure_channel_colors(channel_colors, image_data)
        chunks = _ensure_chunks(chunks, image_data, target_chunk_size=64)
        pixelsizes = _ensure_pixel_sizes(physical_pixel_sizes)
        translation = _ensure_translation(translation, image_data)

        # try to construct per-image metadata
        ome_json = OmeZarrWriter.build_ome(
            image_data.shape,
            image_name,
            channel_names=channel_names,  # type: ignore
            channel_colors=channel_colors,  # type: ignore
            # This can be slow if computed here.
            # TODO: Rely on user to supply the per-channel min/max.
            channel_minmax=[(0.0, 1.0) for i in range(image_data.shape[1])],
        )
        # TODO user supplies units?
        axes_5d = _get_axes_5d()

        transforms, chunk_opts, scaler = _compute_scales(
            scale_num_levels,
            (scale_factor,) * 3,
            pixelsizes,
            chunks,
            image_data.shape,
            translation,
            method="nearest",
        )
        for opt in chunk_opts:
            opt.update(storage_options)

        # TODO image name must be unique within this root group
        group = self.root_group.create_group(image_name, overwrite=True)
        group.attrs["omero"] = ome_json
        write_image(
            image=image_data,
            group=group,
            scaler=scaler,
            axes=axes_5d,
            # For each resolution, we have a List of transformation Dicts (not
            # validated). Each list of dicts are added to each datasets in order.
            coordinate_transformations=transforms,
            # Options to be passed on to the storage backend. A list would need to
            # match the number of datasets in a multiresolution pyramid. One can
            # provide different chunk size for each level of a pyramid using this
            # option.
            storage_options=chunk_opts,
        )
