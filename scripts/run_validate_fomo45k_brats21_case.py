"""JSON-safe entry point for the FOMO45K BraTS21 case validator."""

from __future__ import annotations

import json

import numpy as np


_ORIGINAL_JSON_DEFAULT = json.JSONEncoder.default


def _numpy_json_default(self, value):
    if isinstance(value, np.generic):
        return value.item()
    return _ORIGINAL_JSON_DEFAULT(self, value)


json.JSONEncoder.default = _numpy_json_default

from validate_fomo45k_brats21_case import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
