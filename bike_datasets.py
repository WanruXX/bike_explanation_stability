"""Public hourly data preparation; no private imports or automatic downloads."""
import hashlib
import io
import json
from pathlib import Path
from urllib.request import urlopen
from zipfile import ZipFile

import numpy as np
import pandas as pd

from bike_config import FEATURE_VARIANTS


CATEGORICAL_FEATURES = ("season", "yr", "mnth", "hr", "holiday", "weekday",
                        "workingday", "weathersit")
NUMERIC_FEATURES = ("temp", "atemp", "hum", "windspeed")
FEATURE_NAMES = CATEGORICAL_FEATURES + NUMERIC_FEATURES


def download_bike_data(cfg):
    """Download only when explicitly called; refuse to overwrite existing files."""
    folder = Path(cfg.data_dir)
    destinations = [folder / name for name in ("hour.csv", "download.json")]
    if any(path.exists() for path in destinations):
        raise FileExistsError("Download outputs exist; reuse them or choose another data_dir")
    with urlopen(cfg.download_url, timeout=60) as response:
        archive = response.read(5 * 1024 * 1024 + 1)
    if len(archive) > 5 * 1024 * 1024:
        raise ValueError("Dataset archive exceeds the 5 MiB download limit")
    with ZipFile(io.BytesIO(archive)) as zipped:
        matches = [entry for entry in zipped.infolist() if Path(entry.filename).name == "hour.csv"]
        if len(matches) != 1 or matches[0].file_size > 5 * 1024 * 1024:
            raise ValueError("Expected exactly one small hour.csv in the archive")
        content = zipped.read(matches[0])
    pd.read_csv(io.BytesIO(content), nrows=1, usecols=list(FEATURE_NAMES) + ["cnt", "dteday"])
    folder.mkdir(parents=True, exist_ok=True)
    with destinations[0].open("xb") as stream:
        stream.write(content)
    provenance = dict(url=cfg.download_url, dataset_doi="10.24432/C5W894",
                      license="CC BY 4.0", archive_sha256=hashlib.sha256(archive).hexdigest(),
                      hour_csv_sha256=hashlib.sha256(content).hexdigest())
    with destinations[1].open("x", encoding="utf-8") as stream:
        json.dump(provenance, stream, indent=2, allow_nan=False)
    return provenance


class BikeDataset:
    def __init__(self, frame, feature_names, split):
        self.split = split
        self.feature_names = tuple(feature_names)
        self.inputs = frame.loc[:, list(feature_names)].astype(np.float64).reset_index(drop=True)
        self.targets = frame["cnt"].to_numpy(dtype=np.float64, copy=True)
        self.row_ids = frame["instant"].to_numpy(dtype=np.int64, copy=True)
        self.timestamps = pd.DatetimeIndex(frame["timestamp"])
        if not len(self.targets):
            raise ValueError(f"Empty {split} dataset; adjust chronological boundaries")
        if not np.isfinite(self.inputs.to_numpy()).all() or not np.isfinite(self.targets).all():
            raise ValueError(f"Nonfinite inputs or targets in {split}")

    def __len__(self):
        return len(self.targets)


def feature_groups(feature_names):
    groups = dict(calendar=("season", "yr", "mnth"),
                  daily_schedule=("hr", "holiday", "weekday", "workingday"),
                  temperature=("temp", "atemp"),
                  other_weather=("weathersit", "hum", "windspeed"))
    return {name: tuple(feature for feature in values if feature in feature_names)
            for name, values in groups.items()}


def prepare_bike_datasets(cfg, feature_variant="full"):
    if feature_variant not in FEATURE_VARIANTS:
        raise ValueError("Unsupported feature variant")
    path = Path(cfg.data_dir) / "hour.csv"
    if not path.is_file():
        raise FileNotFoundError(f"Missing {path}; call download_bike_data explicitly first")
    frame = pd.read_csv(path)
    required = set(FEATURE_NAMES) | {"instant", "dteday", "cnt", "casual", "registered"}
    if not required.issubset(frame.columns):
        raise ValueError(f"Missing columns: {sorted(required - set(frame.columns))}")
    values = frame.loc[:, sorted(required - {"dteday"})].to_numpy(dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError("Source numeric data must be finite")
    if frame["instant"].duplicated().any() or (frame["instant"] % 1 != 0).any():
        raise ValueError("instant must contain unique integral row IDs")
    if (frame["hr"] % 1 != 0).any() or not frame["hr"].between(0, 23).all():
        raise ValueError("hr must contain integral hours between 0 and 23")
    if (frame[["cnt", "casual", "registered"]] < 0).any().any():
        raise ValueError("Rental counts must be nonnegative")
    if not np.array_equal(frame["casual"] + frame["registered"], frame["cnt"]):
        raise ValueError("Target components do not sum to cnt")
    frame["timestamp"] = pd.to_datetime(frame["dteday"], format="%Y-%m-%d", errors="raise") + pd.to_timedelta(frame["hr"], unit="h")
    if frame["timestamp"].isna().any() or frame["timestamp"].duplicated().any():
        raise ValueError("Timestamps must be present and unique")
    frame = frame.sort_values("timestamp").reset_index(drop=True)
    train_end, validation_end = pd.Timestamp(cfg.train_end), pd.Timestamp(cfg.validation_end)
    masks = dict(train=frame["timestamp"] < train_end,
                 validation=(frame["timestamp"] >= train_end) & (frame["timestamp"] < validation_end),
                 test=frame["timestamp"] >= validation_end)
    features = tuple(feature for feature in FEATURE_NAMES
                     if not (feature_variant == "temperature_only" and feature == "atemp")
                     and not (feature_variant == "apparent_temperature_only" and feature == "temp"))
    datasets = {split: BikeDataset(frame.loc[mask], features, split) for split, mask in masks.items()}
    csv_hash = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest_path = path.parent / "download.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else None
    if manifest is not None and manifest.get("hour_csv_sha256") != csv_hash:
        raise ValueError("hour.csv differs from its download manifest")
    metadata = dict(feature_variant=feature_variant, feature_names=list(features),
                    feature_groups=feature_groups(features), csv_sha256=csv_hash,
                    download_manifest=manifest, task="retrospective hourly demand regression",
                    excluded_predictors=["instant", "dteday", "cnt", "casual", "registered"],
                    support={split: dict(rows=len(data), first_timestamp=str(data.timestamps.min()),
                                         last_timestamp=str(data.timestamps.max()))
                             for split, data in datasets.items()})
    return dict(datasets=datasets, metadata=metadata)