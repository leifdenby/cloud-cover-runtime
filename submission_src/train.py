import os
import random
from pathlib import Path

import pandas as pd
import pytorch_lightning as pl
from loguru import logger
from pandas_path import path  # noqa


from .cloud_model import CloudModel

BANDS = ["B02", "B03", "B04", "B08"]


def add_paths(df, feature_dir, label_dir=None, bands=BANDS):
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


def _load_dataset_meta(data_path):
    logger.info("start reading data")

    random.seed(9)  # set a seed for reproducibility

    train_meta = pd.read_csv(Path(data_path) / "train_metadata.csv")
    train_features_path = data_path / "train_features"
    train_labels_path = data_path / "train_labels"

    train_meta = add_paths(train_meta, train_features_path, train_labels_path)

    # put 1/3 of chips into the validation set
    chip_ids = train_meta.chip_id.unique().tolist()
    val_chip_ids = random.sample(chip_ids, round(len(chip_ids) * 0.33))

    val_mask = train_meta.chip_id.isin(val_chip_ids)
    val = train_meta[val_mask].copy().reset_index(drop=True)
    train = train_meta[~val_mask].copy().reset_index(drop=True)

    # separate features from labels
    feature_cols = ["chip_id"] + [f"{band}_path" for band in BANDS]

    val_x = val[feature_cols].copy()
    val_y = val[["chip_id", "label_path"]].copy()

    train_x = train[feature_cols].copy()
    train_y = train[["chip_id", "label_path"]].copy()
    logger.info("reading data complete")

    return (train_x, train_y), (val_x, val_y)


def main():
    data_path = Path(os.environ.get("DATAPATH", "data/"))
    (train_x, train_y), (val_x, val_y) = _load_dataset_meta(data_path=data_path)

    # Set up pytorch_lightning.Trainer object
    cloud_model = CloudModel(
        bands=BANDS,
        x_train=train_x,
        y_train=train_y,
        x_val=val_x,
        y_val=val_y,
        hparams={"num_workers": 7, "batch_size": 8},
    )
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        monitor="iou_epoch", mode="max", verbose=True
    )
    early_stopping_callback = pl.callbacks.early_stopping.EarlyStopping(
        monitor="iou_epoch",
        patience=(cloud_model.patience * 3),
        mode="max",
        verbose=True,
    )

    trainer = pl.Trainer(
        gpus=1,
        fast_dev_run=False,
        callbacks=[checkpoint_callback, early_stopping_callback],
    )

    trainer.fit(model=cloud_model)


if __name__ == "__main__":
    main()
