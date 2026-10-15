"""Convert Phase 1 HDF5 sessions into an RLPD demonstration buffer.

Writes the transition-dict list (pickle) that ``serl_launcher``'s replay
buffers consume. Steps after the trailing stationary run of the TCP begins are
discarded. The base-policy mean exists per action row only, so the terminal
observation reuses the previous mean; that row is never bootstrapped from.
"""

from __future__ import annotations

import argparse
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import h5py
import numpy as np

from .episode import (
    DEFAULT_STATIONARY_THRESHOLD_M,
    terminal_run_start,
)
from .observation_spec import (
    ResidualObservationSpec,
    assemble_state,
)

#: Alias kept for tests; the implementation lives in episode.py.
_terminal_run_start = terminal_run_start


class DemoExportError(RuntimeError):
    """Raised when a session cannot be converted into demonstrations."""


@dataclass
class ExportStats:
    sessions: int = 0
    trials_seen: int = 0
    trials_exported: int = 0
    transitions: int = 0
    truncated_steps: int = 0
    backfilled_steps: int = 0
    skipped: list[str] = field(default_factory=list)


def _dataset(trial: h5py.Group, path: str) -> h5py.Dataset:
    node = trial.get(path)
    if not isinstance(node, h5py.Dataset):
        raise DemoExportError(f"{trial.name} is missing required dataset {path!r}")
    return node


def _observation_means(trial: h5py.Group) -> np.ndarray:
    """Per-observation base means, with the terminal row carried forward."""
    means = np.asarray(_dataset(trial, "actions/policy/mu_raw_normalized")[:], np.float32)
    if means.ndim != 2 or means.shape[1] != 6:
        raise DemoExportError(f"{trial.name}: base mean must be [M, 6], got {means.shape}")
    if len(means) == 0:
        raise DemoExportError(f"{trial.name}: no actions recorded")
    mapping = np.asarray(_dataset(trial, "actions/mapping/observation_index")[:], np.int64)
    if not np.array_equal(mapping, np.arange(len(means), dtype=np.int64)):
        raise DemoExportError(
            f"{trial.name}: action-to-observation mapping is not identity; "
            "the per-observation mean cannot be aligned safely"
        )
    return np.concatenate([means, means[-1:]], axis=0)


_TERM_DATASETS = {
    "flange_pose": "observations/robot/flange_pose_world_xyzw",
    "wrench": "observations/robot/ext_wrench_tcp",
    "socket_pose": "observations/task/socket_root_exact_world_xyzw",
}


def _state_matrix(trial: h5py.Group, spec: ResidualObservationSpec) -> np.ndarray:
    """Per-observation state rows, built with ``assemble_state`` like the online runtime."""
    sources: dict[str, np.ndarray] = {}
    for name, _ in spec.terms():
        if name == "base_mean":
            sources[name] = _observation_means(trial)
        else:
            sources[name] = np.asarray(_dataset(trial, _TERM_DATASETS[name])[:])
    columns = list(sources.values())
    lengths = {len(c) for c in columns}
    if len(lengths) != 1:
        raise DemoExportError(f"{trial.name}: state columns disagree in length: {lengths}")
    rows = len(columns[0])
    try:
        state = np.concatenate(
            [assemble_state(spec, {n: v[i] for n, v in sources.items()}) for i in range(rows)],
            axis=0,
        )
    except ValueError as exc:  # ResidualObservationError subclasses ValueError
        raise DemoExportError(f"{trial.name}: {exc}") from exc
    return state


def _images(trial: h5py.Group, spec: ResidualObservationSpec) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    for key in spec.image_keys:
        data = _dataset(trial, f"observations/camera/{key}/rgb")
        arr = np.asarray(data[:])
        if arr.dtype != np.uint8 or arr.ndim != 4 or arr.shape[-1] != 3:
            raise DemoExportError(
                f"{trial.name}/{key}: expected uint8 [N,H,W,3], got {arr.dtype} {arr.shape}"
            )
        out[key] = arr
    return out


def _trial_transitions(
    trial: h5py.Group,
    spec: ResidualObservationSpec,
    *,
    stationary_threshold_m: float,
    stats: ExportStats,
) -> list[dict[str, Any]]:
    accepted = bool(trial.attrs.get("accepted", False))
    state = _state_matrix(trial, spec)
    images = _images(trial, spec)
    actions = np.asarray(
        _dataset(trial, "labels/residual_effective_software_action_minus_raw_mean")[:],
        np.float32,
    )
    valid = np.asarray(_dataset(trial, "labels/command_valid")[:], bool)
    tcp = np.asarray(_dataset(trial, "observations/robot/tcp_pose_world_xyzw")[:])[:, :3]

    n_actions = len(actions)
    if n_actions == 0:
        raise DemoExportError(f"{trial.name}: no actions to export")
    if len(state) != n_actions + 1:
        raise DemoExportError(
            f"{trial.name}: expected {n_actions + 1} observations, found {len(state)}"
        )
    if not valid.all():
        raise DemoExportError(
            f"{trial.name}: {int((~valid).sum())} action rows are not training-eligible"
        )

    run_start = _terminal_run_start(tcp[: n_actions + 1], stationary_threshold_m)
    if run_start == 0:
        raise DemoExportError(
            f"{trial.name}: the whole episode is stationary within "
            f"{stationary_threshold_m} m; nothing meaningful to learn from"
        )
    keep = min(run_start + 1, n_actions)
    # Only an accepted trial reached a true terminal state.
    truncated = not accepted
    if truncated and keep == n_actions:
        # The final observation's carried-forward mean must not be bootstrapped from.
        keep -= 1
        if keep == 0:
            raise DemoExportError(
                f"{trial.name}: truncated episode has no transition with a real "
                "terminal next_observation"
            )
    stats.truncated_steps += n_actions - keep
    stats.backfilled_steps += keep - run_start

    reward = 1.0 if accepted else 0.0
    transitions: list[dict[str, Any]] = []
    for i in range(keep):
        terminal = i == keep - 1
        obs = {"state": state[i][None, :]}
        nxt = {"state": state[i + 1][None, :]}
        for key, frames in images.items():
            obs[key] = frames[i][None, ...]
            nxt[key] = frames[i + 1][None, ...]
        transitions.append(
            {
                "observations": obs,
                "actions": actions[i],
                "next_observations": nxt,
                # Reward only on the terminal transition (first step of the
                # trailing stationary run).
                "rewards": reward if terminal else 0.0,
                # dones marks the episode boundary; masks gates bootstrapping.
                "masks": 0.0 if (terminal and not truncated) else 1.0,
                "dones": bool(terminal),
                "infos": {
                    "succeed": bool(accepted and terminal),
                    "truncated": bool(terminal and truncated),
                    "trial_uuid": str(trial.attrs.get("trial_uuid", "")),
                    "source_step": int(i),
                },
            }
        )

    # The carried-forward terminal mean must never reach a bootstrap.
    if spec.base_mean and not truncated:
        assert transitions[-1]["masks"] == 0.0, "terminal transition must be masked"
    return transitions


def _postprocess_source(path: Path) -> str | None:
    """The recording this one was derived from, if it is a derived recording."""
    try:
        with h5py.File(path, "r") as handle:
            return str(handle.attrs.get("postprocess_source_file", "")) or None
    except OSError:
        return None


def resolve_sessions(paths: Sequence[Path]) -> tuple[list[Path], list[str]]:
    """Expand directories to the .h5 files to read, and say what was dropped.

    Within a directory, a raw recording is skipped when a postprocessed one
    names it as its source, so trials are not exported twice. Explicitly named
    files are always read.
    """
    out: list[Path] = []
    notes: list[str] = []
    for path in paths:
        if not path.is_dir():
            out.append(path)
            continue
        found = sorted(
            p for p in path.glob("*.h5") if not p.name.endswith(".partial.h5")
        )
        superseded: dict[str, str] = {}
        for candidate in found:
            source = _postprocess_source(candidate)
            if source:
                superseded[source] = candidate.name
        for candidate in found:
            if candidate.name in superseded:
                notes.append(
                    f"{candidate.name}: superseded by {superseded[candidate.name]}, "
                    "which was derived from it"
                )
                continue
            out.append(candidate)
    return out, notes


def export_demos(
    sessions: Sequence[Path],
    spec: ResidualObservationSpec,
    *,
    accepted_only: bool = True,
    stationary_threshold_m: float = DEFAULT_STATIONARY_THRESHOLD_M,
) -> tuple[list[dict[str, Any]], ExportStats]:
    stats = ExportStats()
    out: list[dict[str, Any]] = []
    paths, notes = resolve_sessions(sessions)
    stats.skipped.extend(notes)
    #: camera -> (shape, where it was first seen); one buffer cannot mix
    #: resolutions.
    image_shapes: dict[str, tuple[Any, str]] = {}
    for path in paths:
        with h5py.File(path, "r") as handle:
            if not bool(handle.attrs.get("closed_cleanly", False)):
                stats.skipped.append(f"{path.name}: not closed cleanly")
                continue
            stats.sessions += 1
            for name in sorted(handle["trials"].keys()):
                trial = handle["trials"][name]
                stats.trials_seen += 1
                if not bool(trial.attrs.get("complete", False)):
                    stats.skipped.append(f"{path.name}/{name}: incomplete")
                    continue
                if not bool(trial.attrs.get("export_eligible", False)):
                    stats.skipped.append(f"{path.name}/{name}: not export-eligible")
                    continue
                if accepted_only and not bool(trial.attrs.get("accepted", False)):
                    stats.skipped.append(f"{path.name}/{name}: not accepted")
                    continue
                try:
                    transitions = _trial_transitions(
                        trial, spec,
                        stationary_threshold_m=stationary_threshold_m, stats=stats,
                    )
                except DemoExportError as exc:
                    stats.skipped.append(str(exc))
                    continue
                for key in spec.image_keys:
                    shape = np.shape(transitions[0]["observations"][key])
                    seen = image_shapes.setdefault(key, (shape, f"{path.name}/{name}"))
                    if seen[0] != shape:
                        raise DemoExportError(
                            f"camera {key!r} is {shape} in {path.name}/{name} but "
                            f"{seen[0]} in {seen[1]}; one buffer cannot hold both. "
                            "Export the recordings separately, or postprocess them "
                            "to a common resolution."
                        )
                out.extend(transitions)
                stats.trials_exported += 1
                stats.transitions += len(transitions)
    return out, stats


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sessions", nargs="+", type=Path, help="Phase 1 .h5 files or directories")
    parser.add_argument("--output", required=True, type=Path, help="destination .pkl")
    parser.add_argument("--include-unaccepted", action="store_true",
                        help="also export trials the operator did not accept")
    parser.add_argument("--with-socket-pose", action="store_true",
                        help="ablation: include the socket pose in the residual state")
    parser.add_argument("--no-wrench", action="store_true",
                        help="ablation: exclude force/torque from the residual state")
    parser.add_argument(
        "--config", type=Path, default=None,
        help="phase2_live.yaml to take the observation layout from. Without it "
             "the export defaults to a single wrist_1 image regardless of what "
             "the learner and collector are configured for, which silently "
             "produces demonstrations the runtime cannot load.",
    )
    parser.add_argument("--image-keys", nargs="+", default=None,
                        help="override the config's cameras")
    parser.add_argument("--stationary-threshold-m", type=float,
                        default=DEFAULT_STATIONARY_THRESHOLD_M)
    args = parser.parse_args(argv)

    # Explicit flags override the config's observation layout.
    if args.config is not None:
        from .config import Phase2Config

        base = Phase2Config.load(args.config).observation
    else:
        base = ResidualObservationSpec()
    spec = ResidualObservationSpec(
        flange_pose=base.flange_pose,
        wrench=False if args.no_wrench else base.wrench,
        base_mean=base.base_mean,
        socket_pose=True if args.with_socket_pose else base.socket_pose,
        image_keys=(tuple(args.image_keys) if args.image_keys else base.image_keys),
    )
    demos, stats = export_demos(
        args.sessions, spec,
        accepted_only=not args.include_unaccepted,
        stationary_threshold_m=args.stationary_threshold_m,
    )
    print(f"  state layout ({spec.state_dim()}-D): {spec.describe()}")
    print(f"  images: {', '.join(spec.image_keys)}")
    print(f"  sessions={stats.sessions} trials={stats.trials_exported}/{stats.trials_seen} "
          f"transitions={stats.transitions}")
    print(f"  backfilled={stats.backfilled_steps} truncated={stats.truncated_steps}")
    for note in stats.skipped:
        print(f"  skipped: {note}")
    if not demos:
        print("  ERROR: no demonstrations exported", flush=True)
        return 2
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "wb") as handle:
        pickle.dump(demos, handle, protocol=4)
    print(f"  wrote {len(demos)} transitions -> {args.output}")
    from .promotion import episode_lengths

    # Paste into residual.runtime.promotion_seed_lengths in phase2_live.yaml.
    print(f"  promotion_seed_lengths: {sorted(episode_lengths(demos))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
