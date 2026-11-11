"""Names and output paths shared by the inference entry points."""

from pathlib import Path


INFERENCE_PROFILES = ("original", "engineering125", "fast12")


def resolve_inference_profile(name: str | None) -> str:
    if name is None:
        return "original"
    if name not in INFERENCE_PROFILES:
        raise ValueError(
            f"Unknown evaluation.inference_profile: {name!r}. "
            f"Choose from {', '.join(INFERENCE_PROFILES)}."
        )
    return name


def profile_directory(directory: Path, profile: str) -> Path:
    return directory if profile == "original" else directory / profile


def profile_csv(path: Path, profile: str) -> Path:
    return profile_directory(path.parent, profile) / path.name
