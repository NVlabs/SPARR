# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the NVIDIA OneWay Noncommercial License (March 22, 2022).
# See the LICENSE file in the repository root for the full license text:
# https://developer.download.nvidia.com/licenses/NVIDIA-OneWay-Noncommercial-License-22Mar2022.pdf

"""Bounded single-threaded HDF5 command queue for the 30 Hz control path."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from queue import Empty, Full, Queue
from threading import Event, Lock, Thread
from typing import Any, Mapping, Sequence

from .hdf5_schema import SUPPORTED_CAMERA_NAMES
from .hdf5_writer import HDF5SessionWriter, TrialWriter


class WriterBackpressureError(RuntimeError):
    """Raised when the control path would otherwise block on disk I/O."""


class WriterWorkerError(RuntimeError):
    """Raised after the background writer encounters a fatal exception."""


@dataclass
class _Reply:
    event: Event = field(default_factory=Event)
    value: Any = None
    error: BaseException | None = None

    def wait(self, timeout_s: float = 30.0) -> Any:
        if not self.event.wait(timeout_s):
            raise TimeoutError("timed out waiting for HDF5 writer")
        if self.error is not None:
            raise WriterWorkerError(str(self.error)) from self.error
        return self.value


@dataclass(frozen=True)
class _Command:
    kind: str
    payload: Mapping[str, Any] | None = None
    reply: _Reply | None = None


class AsyncHDF5SessionWriter:
    """Serialize all h5py operations away from the ROS control callback.

    Append calls are non-blocking. A full queue is an explicit collection
    failure rather than an unbounded latency spike; the caller must abort the
    trial and mark its label invalid.
    """

    def __init__(
        self,
        final_path: str | Path,
        *,
        queue_size: int,
        metadata: Mapping[str, Any] | None = None,
        calibration: Mapping[str, Any] | None = None,
        root_attrs: Mapping[str, Any] | None = None,
        image_shape: tuple[int, int, int] = (128, 128, 3),
        enabled_cameras: Sequence[str] = SUPPORTED_CAMERA_NAMES,
        training_action_clip: float,
        flush_every: int = 30,
        overwrite: bool = False,
    ) -> None:
        if queue_size < 1:
            raise ValueError("queue_size must be positive")
        self._session = HDF5SessionWriter(
            final_path,
            metadata=metadata,
            calibration=calibration,
            root_attrs=root_attrs,
            image_shape=image_shape,
            enabled_cameras=enabled_cameras,
            training_action_clip=training_action_clip,
            flush_every=flush_every,
            overwrite=overwrite,
        )
        self._queue: Queue[_Command] = Queue(maxsize=int(queue_size))
        self._trial: TrialWriter | None = None
        self._fatal: BaseException | None = None
        self._fatal_lock = Lock()
        self._closed = False
        self._thread = Thread(
            target=self._run,
            name="sparr-phase1-hdf5-writer",
            daemon=False,
        )
        self._thread.start()

    @property
    def final_path(self) -> Path:
        return self._session.final_path

    @property
    def partial_path(self) -> Path:
        return self._session.partial_path

    @property
    def queue_depth(self) -> int:
        return self._queue.qsize()

    @property
    def error(self) -> BaseException | None:
        with self._fatal_lock:
            return self._fatal

    def check_health(self) -> None:
        error = self.error
        if error is not None:
            raise WriterWorkerError(str(error)) from error
        if self._closed:
            raise WriterWorkerError("writer is closed")

    def start_trial(
        self,
        *,
        attempt_index: int,
        attrs: Mapping[str, Any] | None = None,
        trial_id: str | None = None,
    ) -> str:
        reply = _Reply()
        self._put(
            _Command(
                "start",
                {
                    "attempt_index": int(attempt_index),
                    "attrs": dict(attrs or {}),
                    "trial_id": trial_id,
                },
                reply,
            ),
            block=True,
        )
        return str(reply.wait())

    def append_observation(self, record: Mapping[str, Any]) -> None:
        self._put(_Command("observation", record), block=False)

    def append_action(self, record: Mapping[str, Any]) -> None:
        self._put(_Command("action", record), block=False)

    def finish_trial(self, **outcome: Any) -> None:
        reply = _Reply()
        self._put(_Command("finish", outcome, reply), block=True)
        reply.wait()

    def invalidate_pending_action(self, *, guard_bitmask: int = 0) -> bool:
        reply = _Reply()
        self._put(
            _Command("invalidate", {"guard_bitmask": int(guard_bitmask)}, reply),
            block=True,
        )
        return bool(reply.wait())

    def flush(self) -> None:
        reply = _Reply()
        self._put(_Command("flush", reply=reply), block=True)
        reply.wait()

    def close(self, *, clean: bool) -> Path:
        if self._closed:
            return self.final_path if self.final_path.exists() else self.partial_path
        reply = _Reply()
        self._put(_Command("close", {"clean": bool(clean)}, reply), block=True, allow_fatal=True)
        try:
            path = Path(reply.wait())
        finally:
            self._thread.join(timeout=30.0)
            self._closed = True
        return path

    def _put(self, command: _Command, *, block: bool, allow_fatal: bool = False) -> None:
        if not allow_fatal:
            self.check_health()
        try:
            if block:
                self._queue.put(command, timeout=30.0)
            else:
                self._queue.put_nowait(command)
        except Full as exc:
            raise WriterBackpressureError(
                f"HDF5 writer queue reached capacity {self._queue.maxsize}"
            ) from exc

    def _set_fatal(self, error: BaseException) -> None:
        with self._fatal_lock:
            if self._fatal is None:
                self._fatal = error

    def _run(self) -> None:
        while True:
            try:
                command = self._queue.get(timeout=0.25)
            except Empty:
                continue
            try:
                if self.error is not None and command.kind != "close":
                    raise WriterWorkerError(str(self.error))
                value = self._execute(command)
                if command.reply is not None:
                    command.reply.value = value
            except BaseException as exc:  # preserve original error for the ROS thread
                self._set_fatal(exc)
                if command.reply is not None:
                    command.reply.error = exc
            finally:
                if command.reply is not None:
                    command.reply.event.set()
                self._queue.task_done()
            if command.kind == "close":
                return

    def _execute(self, command: _Command) -> Any:
        payload = dict(command.payload or {})
        if command.kind == "start":
            self._trial = self._session.start_trial(**payload)
            return self._trial.trial_uuid
        if command.kind == "observation":
            self._require_trial().append_observation(payload)
            return None
        if command.kind == "action":
            self._require_trial().append_action(payload)
            return None
        if command.kind == "finish":
            self._require_trial().finish(**payload)
            self._trial = None
            return None
        if command.kind == "invalidate":
            return self._require_trial().invalidate_pending_action(**payload)
        if command.kind == "flush":
            self._session.flush()
            return None
        if command.kind == "close":
            requested_clean = bool(payload.get("clean", False))
            return self._session.close(clean=requested_clean and self.error is None)
        raise ValueError(f"unknown writer command {command.kind!r}")

    def _require_trial(self) -> TrialWriter:
        if self._trial is None:
            raise RuntimeError("no HDF5 trial is active")
        return self._trial


__all__ = [
    "AsyncHDF5SessionWriter",
    "WriterBackpressureError",
    "WriterWorkerError",
]
