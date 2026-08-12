"""LMDB-backed healthy-slice dataset adapter."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset
from torchvision import transforms


class LMDBSliceDataset(Dataset):
    """Read the original ANDi healthy-slice LMDB format."""

    def __init__(self, directory: str | Path, image_size: int | None = None):
        try:
            import lmdb
        except ImportError as exc:
            raise ImportError("LMDBSliceDataset requires the optional 'lmdb' package.") from exc

        self._lmdb = lmdb
        self.directory = str(directory)
        self.image_size = image_size
        env = self._lmdb.open(
            self.directory,
            max_readers=1,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )
        with env.begin(write=False) as txn:
            self.length = txn.stat()["entries"]
        env.close()

    def _open_lmdb(self) -> None:
        self.env = self._lmdb.open(
            self.directory,
            max_readers=1,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )
        self.txn = self.env.begin(write=False)

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> torch.Tensor:
        if not hasattr(self, "txn"):
            self._open_lmdb()

        byteflow = self.txn.get(f"{index:08}".encode("ascii"))
        if byteflow is None:
            raise IndexError(index)

        tensor = torch.from_numpy(pickle.loads(byteflow)).float()
        if self.image_size is not None and tensor.shape[-1] != self.image_size:
            tensor = transforms.Resize(self.image_size, antialias=True)(tensor)
        return tensor


def build_lmdb_dataset(config: dict[str, Any]) -> LMDBSliceDataset:
    image_size = int(config.get("image_size", 128))
    if "path" not in config:
        raise ValueError("data.path is required when data.type is 'lmdb'.")
    return LMDBSliceDataset(config["path"], image_size=image_size)
