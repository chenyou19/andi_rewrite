"""LMDB-backed healthy-slice dataset adapter."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, Sequence

import torch
from torch.utils.data import Dataset
from torchvision import transforms


class LMDBSliceDataset(Dataset):
    """Read the original ANDi healthy-slice LMDB format."""

    def __init__(
        self,
        directory: str | Path,
        image_size: int | None = None,
        channel_indices: Sequence[int] | None = None,
    ):
        try:
            import lmdb
        except ImportError as exc:
            raise ImportError("LMDBSliceDataset requires the optional 'lmdb' package.") from exc

        self._lmdb = lmdb
        self.directory = str(directory)
        self.image_size = image_size
        if channel_indices is None:
            self.channel_indices = None
        else:
            values = list(channel_indices)
            if not values:
                raise ValueError("LMDB channel_indices must not be empty.")
            if any(isinstance(value, bool) or not isinstance(value, int) for value in values):
                raise TypeError("LMDB channel_indices must contain only integers (not bool).")
            if len(set(values)) != len(values):
                raise ValueError("LMDB channel_indices must be unique.")
            if any(value < 0 for value in values):
                raise IndexError("LMDB channel_indices must be non-negative.")
            self.channel_indices = values
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
        if self.channel_indices is not None:
            invalid = [index for index in self.channel_indices if index >= tensor.shape[0]]
            if invalid:
                raise IndexError(
                    f"LMDB channel_indices out of range for source C={tensor.shape[0]}: {invalid}."
                )
            tensor = torch.index_select(
                tensor,
                dim=0,
                index=torch.tensor(self.channel_indices, dtype=torch.long),
            )
        if self.image_size is not None and tensor.shape[-1] != self.image_size:
            tensor = transforms.Resize(self.image_size, antialias=True)(tensor)
        return tensor


def build_lmdb_dataset(config: dict[str, Any]) -> LMDBSliceDataset:
    image_size = int(config.get("image_size", 128))
    if "path" not in config:
        raise ValueError("data.path is required when data.type is 'lmdb'.")
    if "channel_indices" not in config:
        # Preserve the historical constructor call exactly for old configs.
        return LMDBSliceDataset(config["path"], image_size=image_size)
    return LMDBSliceDataset(
        config["path"], image_size=image_size, channel_indices=config["channel_indices"]
    )
