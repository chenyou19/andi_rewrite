"""輕量 config schema / validation。

只用標準庫（typing / 內建型別），不引入 pydantic / hydra / omegaconf。
目標不是把整個系統 dataclass 化，而是提供一個清楚的 validation entry point：

1. 確認 top-level sections 是 dict
2. 檢查明顯錯誤型別與必要欄位
3. 保留所有未知欄位（未知 top-level section 不報錯）
4. 不檢查路徑是否實體存在
5. 不轉型覆蓋原值，只驗證「可轉型」

validate_config() 回傳的仍是原本那個 dict，呼叫端可以照舊當作 dict 使用。
"""

from __future__ import annotations

from typing import Any

# 已知 section；若存在則必須是 dict。未列出的 top-level key 不會報錯（保留）。
_DICT_SECTIONS = (
    "experiment",
    "runtime",
    "data",
    "model",
    "diffusion",
    "noise",
    "training",
    "anomaly",
    "metrics",
    "evaluation",
)


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _min_suffix(min_value: Any, exclusive: bool) -> str:
    if min_value is None:
        return ""
    return f" {'>' if exclusive else '>='} {min_value}"


def ensure_mapping(config: dict[str, Any], key: str) -> dict[str, Any] | None:
    """若 key 存在但不是 dict 則 raise；回傳該 section 或 None。"""

    if key not in config:
        return None
    section = config[key]
    if not isinstance(section, dict):
        raise ValueError(f"Config section '{key}' must be a mapping/dict.")
    return section


def require_keys(section: dict[str, Any], keys: list[str], section_name: str) -> None:
    for key in keys:
        if key not in section:
            raise ValueError(f"Missing required config key: {section_name}.{key}")


def as_int(value: Any, path: str, *, min_value: int | None = None, exclusive: bool = True) -> int:
    suffix = _min_suffix(min_value, exclusive)
    try:
        ivalue = int(value)
    except (TypeError, ValueError):
        raise ValueError(f"Config key '{path}' must be an integer{suffix}.")
    if min_value is not None and not (ivalue > min_value if exclusive else ivalue >= min_value):
        raise ValueError(f"Config key '{path}' must be an integer{suffix}.")
    return ivalue


def as_float(value: Any, path: str, *, min_value: float | None = None, exclusive: bool = True) -> float:
    suffix = _min_suffix(min_value, exclusive)
    try:
        fvalue = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"Config key '{path}' must be a float{suffix}.")
    if min_value is not None and not (fvalue > min_value if exclusive else fvalue >= min_value):
        raise ValueError(f"Config key '{path}' must be a float{suffix}.")
    return fvalue


def as_bool(value: Any, path: str) -> bool:
    # downstream 用 bool(value)，幾乎不會 raise；這裡只擋掉明顯不合理的容器型別。
    if isinstance(value, (dict, list, tuple, set)):
        raise ValueError(f"Config key '{path}' must be a boolean.")
    return bool(value)


def validate_optional_int(section: dict[str, Any], key: str, path_prefix: str, *,
                          min_value: int | None = None, exclusive: bool = True) -> None:
    if key in section and section[key] is not None:
        as_int(section[key], f"{path_prefix}.{key}", min_value=min_value, exclusive=exclusive)


def validate_optional_float(section: dict[str, Any], key: str, path_prefix: str, *,
                            min_value: float | None = None, exclusive: bool = True) -> None:
    if key in section and section[key] is not None:
        as_float(section[key], f"{path_prefix}.{key}", min_value=min_value, exclusive=exclusive)


def validate_optional_bool(section: dict[str, Any], key: str, path_prefix: str) -> None:
    if key in section and section[key] is not None:
        as_bool(section[key], f"{path_prefix}.{key}")


def _ensure_nested_mapping(section: dict[str, Any], key: str, path_prefix: str) -> dict[str, Any] | None:
    value = section.get(key)
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ValueError(f"Config section '{path_prefix}.{key}' must be a mapping/dict.")
    return value


# --------------------------------------------------------------------------- #
# per-section validators
# --------------------------------------------------------------------------- #
def _validate_runtime(config: dict[str, Any]) -> None:
    runtime = config.get("runtime")
    if not isinstance(runtime, dict):
        return
    validate_optional_int(runtime, "seed", "runtime")
    if "device" in runtime and runtime["device"] is not None and not isinstance(runtime["device"], str):
        raise ValueError("Config key 'runtime.device' must be a string.")
    validate_optional_bool(runtime, "accelerate", "runtime")
    validate_optional_bool(runtime, "distributed", "runtime")
    validate_optional_bool(runtime, "find_unused_parameters", "runtime")
    validate_optional_bool(runtime, "cudnn_benchmark", "runtime")


def _validate_data(config: dict[str, Any], mode: str | None) -> None:
    data = config.get("data")
    if not isinstance(data, dict):
        return
    validate_optional_int(data, "batch_size", "data", min_value=0, exclusive=True)
    validate_optional_int(data, "workers", "data", min_value=0, exclusive=False)
    validate_optional_int(data, "image_size", "data", min_value=0, exclusive=True)
    validate_optional_int(data, "channels", "data", min_value=0, exclusive=True)
    validate_optional_int(data, "size_splits", "data", min_value=0, exclusive=True)

    # 只在能確定是 train/eval 時檢查 dataset 必要欄位；只檢查 key，不檢查檔案存在。
    if mode in ("train", "eval"):
        data_type = str(data.get("type", "lmdb")).lower()
        if data_type == "lmdb":
            require_keys(data, ["path"], "data")
        elif data_type in {"brats_healthy_slices", "healthy_slices"}:
            require_keys(data, ["path_to_csv", "dataset_path"], "data")
        elif data_type in {"volume", "mri_volume", "brats_volume"}:
            require_keys(data, ["path_to_csv", "dataset_path"], "data")


def _validate_model(config: dict[str, Any]) -> None:
    model = config.get("model")
    if not isinstance(model, dict):
        return
    for key in ("image_size", "in_channels", "out_channels", "channels", "time_dim", "base_channels", "num_blocks"):
        validate_optional_int(model, key, "model", min_value=0, exclusive=True)
    validate_optional_float(model, "dropout", "model", min_value=0, exclusive=False)


def _validate_diffusion(config: dict[str, Any]) -> None:
    diffusion = config.get("diffusion")
    if not isinstance(diffusion, dict):
        return
    validate_optional_int(diffusion, "steps", "diffusion", min_value=0, exclusive=True)
    validate_optional_float(diffusion, "beta_start", "diffusion", min_value=0, exclusive=True)
    validate_optional_float(diffusion, "beta_end", "diffusion", min_value=0, exclusive=True)
    beta_start = diffusion.get("beta_start")
    beta_end = diffusion.get("beta_end")
    if beta_start is not None and beta_end is not None and not float(beta_end) > float(beta_start):
        raise ValueError("Config requires diffusion.beta_end > diffusion.beta_start.")


def _validate_sampler(sampler: Any, path: str) -> None:
    if not isinstance(sampler, dict):
        raise ValueError(f"Config key '{path}' must be a mapping/dict.")
    if str(sampler.get("type", "")).lower() == "hybrid":
        components = sampler.get("components")
        if not isinstance(components, list):
            raise ValueError(f"Config key '{path}.components' must be a list.")
        for index, component in enumerate(components):
            component_path = f"{path}.components[{index}]"
            if not isinstance(component, dict):
                raise ValueError(f"Config key '{component_path}' must be a mapping/dict.")
            if "weight" in component and component["weight"] is not None:
                as_float(component["weight"], f"{component_path}.weight")
            _validate_sampler(component, component_path)  # 巢狀 hybrid 也檢查


def _validate_noise(config: dict[str, Any]) -> None:
    noise = config.get("noise")
    if noise is None:
        return
    if not isinstance(noise, dict):
        raise ValueError("Config section 'noise' must be a mapping/dict.")
    schedule = noise.get("schedule")
    if schedule is not None:
        if not isinstance(schedule, dict):
            raise ValueError("Config key 'noise.schedule' must be a mapping/dict.")
        if str(schedule.get("type", "")).lower() == "epoch_switch":
            for key in ("before", "after"):
                sub = schedule.get(key)
                if not isinstance(sub, dict):
                    raise ValueError(f"Config key 'noise.schedule.{key}' must be a mapping/dict.")
                _validate_sampler(sub, f"noise.schedule.{key}")
        if schedule.get("sampler") is not None:
            _validate_sampler(schedule["sampler"], "noise.schedule.sampler")
    elif noise.get("sampler") is not None:
        _validate_sampler(noise["sampler"], "noise.sampler")
    else:
        # 扁平形式：noise 本身就是 sampler config（含 hybrid 的情況）。
        _validate_sampler(noise, "noise")


def _validate_training(config: dict[str, Any]) -> None:
    training = config.get("training")
    if not isinstance(training, dict):
        return
    validate_optional_int(training, "epochs", "training", min_value=0, exclusive=True)
    validate_optional_float(training, "start_lr", "training", min_value=0, exclusive=True)
    validate_optional_float(training, "target_lr", "training", min_value=0, exclusive=True)
    validate_optional_bool(training, "normalize_input", "training")

    for nested in ("scheduler", "ema", "checkpoint", "samples", "progress"):
        _ensure_nested_mapping(training, nested, "training")

    ema = training.get("ema")
    if isinstance(ema, dict):
        validate_optional_bool(ema, "enabled", "training.ema")
        validate_optional_float(ema, "decay", "training.ema")

    checkpoint = training.get("checkpoint")
    if isinstance(checkpoint, dict):
        validate_optional_int(checkpoint, "save_every_epochs", "training.checkpoint", min_value=0, exclusive=True)

    samples = training.get("samples")
    if isinstance(samples, dict):
        validate_optional_bool(samples, "enabled", "training.samples")
        validate_optional_int(samples, "num_images", "training.samples", min_value=0, exclusive=True)
        validate_optional_int(samples, "channels", "training.samples", min_value=0, exclusive=True)
        validate_optional_int(samples, "image_size", "training.samples", min_value=0, exclusive=True)


def _validate_anomaly(config: dict[str, Any]) -> None:
    anomaly = config.get("anomaly")
    if not isinstance(anomaly, dict):
        return
    for key in ("t_lower", "start", "t_upper", "stop"):
        validate_optional_int(anomaly, key, "anomaly")
    t_lower = anomaly.get("t_lower", anomaly.get("start"))
    t_upper = anomaly.get("t_upper", anomaly.get("stop"))
    if t_lower is not None and t_upper is not None and not int(t_upper) > int(t_lower):
        raise ValueError("Config requires anomaly.t_upper > anomaly.t_lower.")

    median_filter = _ensure_nested_mapping(anomaly, "median_filter", "anomaly")
    if median_filter is not None:
        validate_optional_bool(median_filter, "enabled", "anomaly.median_filter")
        validate_optional_int(median_filter, "kernel_size", "anomaly.median_filter", min_value=0, exclusive=True)

    for key in ("aggregation", "modality_pool"):
        if key in anomaly and anomaly[key] is not None and not isinstance(anomaly[key], (str, dict)):
            raise ValueError(f"Config key 'anomaly.{key}' must be a string or mapping/dict.")

    if anomaly.get("postprocess") is not None and not isinstance(anomaly["postprocess"], dict):
        raise ValueError("Config section 'anomaly.postprocess' must be a mapping/dict.")


def _validate_metrics(config: dict[str, Any]) -> None:
    metrics = config.get("metrics")
    if not isinstance(metrics, dict):
        return
    validate_optional_float(metrics, "thr_start", "metrics")
    validate_optional_float(metrics, "threshold_start", "metrics")
    validate_optional_float(metrics, "thr_end", "metrics")
    validate_optional_float(metrics, "threshold_end", "metrics")
    validate_optional_float(metrics, "thr_step", "metrics", min_value=0, exclusive=True)
    validate_optional_float(metrics, "threshold_step", "metrics", min_value=0, exclusive=True)
    validate_optional_float(metrics, "threshold", "metrics")
    validate_optional_int(metrics, "size_splits", "metrics", min_value=0, exclusive=True)
    validate_optional_int(metrics, "kernel_size", "metrics", min_value=0, exclusive=True)
    validate_optional_int(metrics, "rank", "metrics", min_value=0, exclusive=True)
    validate_optional_int(metrics, "connectivity", "metrics", min_value=0, exclusive=True)

    if metrics.get("median_filter") is not None and not isinstance(metrics["median_filter"], dict):
        raise ValueError("Config section 'metrics.median_filter' must be a mapping/dict.")
    if metrics.get("postprocess") is not None and not isinstance(metrics["postprocess"], dict):
        raise ValueError("Config section 'metrics.postprocess' must be a mapping/dict.")

    start = metrics.get("thr_start", metrics.get("threshold_start"))
    end = metrics.get("thr_end", metrics.get("threshold_end"))
    step = metrics.get("thr_step", metrics.get("threshold_step"))
    if start is not None and end is not None and step is not None and not float(end) > float(start):
        raise ValueError("Config requires metrics.threshold_end > metrics.threshold_start.")


def _validate_evaluation(config: dict[str, Any]) -> None:
    evaluation = config.get("evaluation")
    if not isinstance(evaluation, dict):
        return
    progress = evaluation.get("progress")
    if progress is not None and not isinstance(progress, (bool, dict)):
        raise ValueError("Config key 'evaluation.progress' must be a boolean or mapping/dict.")


# --------------------------------------------------------------------------- #
# public API
# --------------------------------------------------------------------------- #
def infer_config_mode(config: dict[str, Any]) -> str | None:
    """依 config 內容推測 train / eval；無法判斷時回傳 None。"""

    if "training" in config:
        return "train"
    if "metrics" in config or "evaluation" in config or "anomaly" in config:
        return "eval"
    return None


def validate_config(config: dict[str, Any], *, mode: str | None = None) -> dict[str, Any]:
    """輕量驗證 config 並原樣回傳（不轉型、不丟棄未知欄位）。"""

    if not isinstance(config, dict):
        raise ValueError("Config root must be a mapping/dict.")

    for key in _DICT_SECTIONS:
        ensure_mapping(config, key)

    resolved_mode = mode if mode is not None else infer_config_mode(config)

    _validate_runtime(config)
    _validate_data(config, resolved_mode)
    _validate_model(config)
    _validate_diffusion(config)
    _validate_noise(config)
    _validate_training(config)
    _validate_anomaly(config)
    _validate_metrics(config)
    _validate_evaluation(config)
    return config
