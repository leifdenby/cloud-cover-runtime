import os
import random
from pathlib import Path
import dask
import dask.array as daskarray

import numpy as np
import pandas as pd
import xarray as xr
from pandas_path import path  # noqa
from loguru import logger
import click
from tqdm import tqdm
import rasterio


def add_paths(df, feature_dir, bands, label_dir=None):
    """
    Given dataframe with a column for chip_id, returns a dataframe with a column
    added indicating the path to each band's TIF image as "{band}_path", eg "B02_path".
    A column is also added to the dataframe with paths to the label TIF, if the
    path to the labels directory is provided.
    """
    for band in bands:
        df[f"{band}_path"] = feature_dir / df["chip_id"] / f"{band}.tif"
        # make sure a random sample of paths exist
        assert df.sample(n=40, random_state=5)[f"{band}_path"].path.exists().all()
    if label_dir is not None:
        df["label_path"] = label_dir / (df["chip_id"] + ".tif")
        # make sure a random sample of paths exist
        assert df.sample(n=40, random_state=5)["label_path"].path.exists().all()

    return df


def _load_dataset_meta(data_path, bands):
    logger.info("start reading csv")

    random.seed(9)  # set a seed for reproducibility

    train_meta = pd.read_csv(Path(data_path) / "train_metadata.csv")
    train_features_path = data_path / "train_features"
    train_labels_path = data_path / "train_labels"

    train_meta = add_paths(
        df=train_meta,
        feature_dir=train_features_path,
        label_dir=train_labels_path,
        bands=bands,
    )

    # put 1/3 of chips into the validation set
    chip_ids = train_meta.chip_id.unique().tolist()
    val_chip_ids = random.sample(chip_ids, round(len(chip_ids) * 0.33))

    val_mask = train_meta.chip_id.isin(val_chip_ids)
    val = train_meta[val_mask].copy().reset_index(drop=True)
    train = train_meta[~val_mask].copy().reset_index(drop=True)

    logger.info("finished reading csv and creating train/val split")

    return train, val


def build_xr_ds(df, bands, transforms=None):
    # https://blog.dask.org/2019/06/20/load-image-data
    samples = []

    def _read_chip(bands, row):
        band_arrs = []
        for band in bands:
            fp = getattr(row, f"{band}_path")
            with rasterio.open(fp) as b:
                band_arr = b.read(1).astype("float32")
                band_arrs.append(band_arr)
        x_arr = np.stack(band_arrs, axis=-1)

        # Apply data augmentations, if provided
        if transforms:
            x_arr = transforms(image=x_arr)["image"]
        x_arr = np.transpose(x_arr, [2, 0, 1])

        return x_arr[..., np.newaxis]

    def _read_label(row):
        with rasterio.open(row.label_path) as lp:
            y_arr = lp.read(1).astype("float32")
        # Apply same data augmentations to the label
        if transforms:
            y_arr = transforms(image=y_arr)["image"]
        return y_arr[..., np.newaxis]

    def _lazy_load(load_fn, **kwargs):
        sample = load_fn(row=df.iloc[0], **kwargs)

        array_lazy = [
            dask.delayed(load_fn)(row=row, **kwargs) for _, row in df.iterrows()
        ]

        array_lazy = [
            daskarray.from_delayed(x, shape=sample.shape, dtype="float32")
            for x in tqdm(array_lazy)
        ]

        return daskarray.concatenate(array_lazy, axis=-1)

    ds = xr.Dataset()
    da_chip = xr.DataArray(
        _lazy_load(_read_chip, bands=bands),
        dims=("bands", "x", "y", "chip_id"),
        coords=dict(bands=bands, chip_id=df.chip_id),
    )
    ds["chip"] = da_chip

    da_label = xr.DataArray(
        _lazy_load(_read_label),
        dims=("x", "y", "chip_id"),
        coords=dict(chip_id=df.chip_id),
    )
    ds["label"] = da_label

    # ds = ds.assign_coords(bands=bands)
    return ds


@click.command()
@click.argument("data_path")
@click.option(
    "--bands", default=["B02", "B03", "B04", "B08"], type=list, show_default=True
)
def main(data_path: os.PathLike, bands):
    """
    From training data CSV create zarr-based datasets for use during training
    for specific set of channels
    """
    from dask.distributed import Client

    client = Client()
    logger.info(client)

    conditions = ["train", "val"]
    f_id = "_".join(sorted(bands))

    fps = [Path(f"{cond}__{f_id}.zarr") for cond in conditions]

    for cond, fp in zip(conditions, fps):
        if fp.exists():
            raise Exception(f"{fp} already exists")

    data_path = Path(os.environ.get("DATAPATH", "data/"))
    df_train, df_val = _load_dataset_meta(data_path=data_path, bands=bands)

    for cond, fp, df in zip(conditions, fps, [df_train, df_val]):
        logger.info(f"create dataset for {cond}")
        ds = build_xr_ds(df=df, bands=bands)

        ds = ds.chunk(dict(chip_id=100))
        ds.to_zarr(fp)
        print(f"saved to {fp}")


if __name__ == "__main__":
    main()
