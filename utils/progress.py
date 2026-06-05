from __future__ import annotations

import sys
import time
from typing import Any


def format_duration(seconds: float | None) -> str:
    if seconds is None or seconds < 0:
        return "--:--"
    seconds = int(round(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours:d}:{minutes:02d}:{seconds:02d}"
    return f"{minutes:02d}:{seconds:02d}"


class ProgressReporter:
    """Small tqdm-backed progress reporter with a no-dependency fallback."""

    def __init__(
        self,
        total: int,
        description: str,
        enabled: bool = True,
        unit: str = "item",
        leave: bool = True,
    ):
        self.total = int(total)
        self.description = description
        self.enabled = bool(enabled and self.total > 0)
        self.unit = unit
        self.leave = leave
        self.current = 0
        self.started_at = time.monotonic()
        self._last_width = 0
        self._tqdm = None
        self._postfix = ""
        if not self.enabled:
            return
        try:
            from tqdm.auto import tqdm

            self._tqdm = tqdm(total=self.total, desc=description, unit=unit, leave=leave)
        except ImportError:
            self._render()

    def update(self, count: int = 1, postfix: dict[str, Any] | str | None = None) -> None:
        if not self.enabled:
            return
        if postfix is not None:
            self.set_postfix(postfix)
        self.current += int(count)
        if self._tqdm is not None:
            self._tqdm.update(count)
            return
        self._render()

    def set_postfix(self, postfix: dict[str, Any] | str) -> None:
        if not self.enabled:
            return
        if self._tqdm is not None:
            if isinstance(postfix, dict):
                self._tqdm.set_postfix(postfix)
            else:
                self._tqdm.set_postfix_str(str(postfix))
            return
        if isinstance(postfix, dict):
            self._postfix = " ".join(f"{key}={value}" for key, value in postfix.items())
        else:
            self._postfix = str(postfix)
        self._render()

    def close(self) -> None:
        if not self.enabled:
            return
        if self._tqdm is not None:
            self._tqdm.close()
            return
        sys.stderr.write("\n")
        sys.stderr.flush()

    def _render(self) -> None:
        elapsed = time.monotonic() - self.started_at
        rate = self.current / elapsed if elapsed > 0 else 0.0
        remaining = (self.total - self.current) / rate if rate > 0 else None
        percent = self.current / self.total
        bar_width = 30
        filled = min(bar_width, int(round(bar_width * percent)))
        bar = "#" * filled + "-" * (bar_width - filled)
        postfix = f" {self._postfix}" if self._postfix else ""
        message = (
            f"\r{self.description}: |{bar}| {self.current}/{self.total} "
            f"{percent * 100:5.1f}% elapsed {format_duration(elapsed)} "
            f"ETA {format_duration(remaining)}{postfix}"
        )
        padding = " " * max(0, self._last_width - len(message))
        sys.stderr.write(message + padding)
        sys.stderr.flush()
        self._last_width = len(message)
