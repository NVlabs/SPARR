# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the NVIDIA License.
# See the LICENSE file in the repository root for the full license text.

"""TensorBoard logging for the residual learner.

Three groups of scalars:

``train/*``     every metric ``train_step`` returns.
``buffer/*``    online and demo buffer sizes, promotion count and median.
``episode/*``   per-episode actor outcome and residual statistics, plus a
                rolling success rate, logged against the learner step.

A missing TensorBoard backend disables logging instead of failing.
"""

from __future__ import annotations

from collections import deque
from pathlib import Path
from typing import Any, Mapping, Optional

#: Rolling window for the success-rate curve, in episodes.
DEFAULT_WINDOW = 20

#: The residual's six action dimensions, in pipeline order.
AXIS_NAMES = ("dx", "dy", "dz", "rx", "ry", "rz")


class MetricsLogger:
    """TensorBoard writer with a no-op fallback."""

    def __init__(
        self,
        logdir: Optional[str | Path],
        *,
        window: int = DEFAULT_WINDOW,
        enabled: bool = True,
    ) -> None:
        self._writer: Any = None
        self._window: deque[int] = deque(maxlen=window)
        self._episodes = 0
        self._successes = 0
        self._step = 0
        self.logdir: Optional[Path] = None
        if not enabled or logdir is None:
            return
        try:
            from torch.utils.tensorboard import SummaryWriter
        except Exception as exc:  # noqa: BLE001 - logging must never be fatal
            print(f"tensorboard unavailable ({type(exc).__name__}: {exc}); "
                  "metrics will not be logged")
            return
        self.logdir = Path(logdir)
        self.logdir.mkdir(parents=True, exist_ok=True)
        self._writer = SummaryWriter(str(self.logdir))

    @property
    def active(self) -> bool:
        return self._writer is not None

    def _scalar(self, tag: str, value: Any, step: int) -> None:
        if self._writer is None:
            return
        try:
            self._writer.add_scalar(tag, float(value), step)
        except (TypeError, ValueError):
            pass  # non-scalar metrics are skipped, not fatal

    def log_train(self, step: int, metrics: Mapping[str, Any]) -> None:
        """Every metric train_step returned, under train/."""
        self._step = step
        for key, value in metrics.items():
            self._scalar(f"train/{key}", value, step)

    def log_buffers(
        self,
        step: int,
        *,
        online: Optional[int] = None,
        demo: Optional[int] = None,
        promoted: Optional[int] = None,
        promotion_median: Optional[float] = None,
    ) -> None:
        for tag, value in (
            ("buffer/online_size", online),
            ("buffer/demo_size", demo),
            ("buffer/promoted_total", promoted),
            ("buffer/promotion_median", promotion_median),
        ):
            if value is not None:
                self._scalar(tag, value, step)

    def log_episode(self, payload: Mapping[str, Any]) -> None:
        """One finished episode from the actor, logged at the current learner step."""
        step = self._step
        succeeded = bool(payload.get("succeed", False))
        self._episodes += 1
        self._successes += int(succeeded)
        self._window.append(int(succeeded))

        self._scalar("episode/succeeded", int(succeeded), step)
        self._scalar("episode/length", payload.get("length", 0), step)
        self._scalar("episode/promoted", int(bool(payload.get("promoted", False))), step)
        residual_ok = payload.get("residual_ok_fraction")
        if residual_ok is not None:
            self._scalar("episode/residual_ok_fraction", residual_ok, step)
        for key in (
            "residual_abs_mean",
            "residual_abs_mean_scaled",
            "residual_max_abs",
            "residual_saturation",
            "residual_steps",
        ):
            if key in payload:
                self._scalar(f"episode/{key}", payload[key], step)
        for key, prefix in (
            ("residual_mean_axes", "episode/residual_mean"),
            ("residual_abs_mean_axes", "episode/residual_abs_mean"),
        ):
            for axis, value in enumerate(payload.get(key) or ()):
                self._scalar(f"{prefix}/{AXIS_NAMES[axis]}", value, step)

        self._scalar("episode/count", self._episodes, step)
        self._scalar(
            "episode/success_rate_cumulative", self._successes / self._episodes, step
        )
        if self._window:
            self._scalar(
                f"episode/success_rate_last_{self._window.maxlen}",
                sum(self._window) / len(self._window),
                step,
            )

    def log_checkpoint(self, step: int) -> None:
        """Mark where a checkpoint was written, so curves can be read against it."""
        self._scalar("checkpoint/written", 1, step)
        if self._episodes:
            self._scalar(
                "checkpoint/success_rate_at_write",
                self._successes / self._episodes,
                step,
            )

    def log_hparams(self, params: Mapping[str, Any]) -> None:
        if self._writer is None:
            return
        text = "\n".join(f"- **{k}**: `{v}`" for k, v in sorted(params.items()))
        try:
            self._writer.add_text("config", text, 0)
        except Exception:  # noqa: BLE001
            pass

    def flush(self) -> None:
        if self._writer is not None:
            self._writer.flush()

    def close(self) -> None:
        if self._writer is not None:
            self._writer.flush()
            self._writer.close()
            self._writer = None
