"""Console entry point for the residual learner.

Runs without ROS or a robot. Needs demonstrations, a checkpoint directory, and
either a connected actor, ``--replay-online`` data, or ``--offline`` for a dry run.
"""

from __future__ import annotations

import argparse
import re
import shutil
import signal
import sys
from pathlib import Path
from typing import Optional, Sequence

from .config import TRAINING_CONFIG_FILENAME, Phase2Config
from .learner import LearnerConfig, LearnerError, ResidualLearner
from .metrics_log import MetricsLogger
from .observation_spec import ResidualObservationSpec


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="sparr_phase2_learner",
        description="RLPD residual learner (robot-free, ROS-free)",
    )
    parser.add_argument(
        "--demos", nargs="+", default=None, metavar="PKL",
        help="demonstration pickles from sparr_phase2_make_demos (required "
             "except when only exporting)",
    )
    parser.add_argument(
        "--checkpoint-dir", required=True, type=Path,
        help="directory for agent checkpoints and learner state",
    )
    parser.add_argument(
        "--config", type=Path, default=None,
        help="phase2_live.yaml, for the observation layout (defaults to the "
             "built-in 19-D layout)",
    )
    parser.add_argument("--steps", type=int, default=None,
                        help="stop after this many steps (default: max_steps)")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--cta-ratio", type=int, default=None)
    parser.add_argument("--training-starts", type=int, default=None)
    parser.add_argument("--max-promotions", type=int, default=None)
    parser.add_argument("--steps-per-update", type=int, default=None,
                        help="learner steps between parameter broadcasts to the "
                             "actor (default %d)" % LearnerConfig.steps_per_update)
    parser.add_argument("--checkpoint-period", type=int, default=None,
                        help="learner steps between checkpoints (default %d)"
                             % LearnerConfig.checkpoint_period)
    parser.add_argument("--log-period", type=int, default=None,
                        help="learner steps between metric logs (default %d)"
                             % LearnerConfig.log_period)
    parser.add_argument("--checkpoint-step", type=int, default=None,
                        help="restore this checkpoint step instead of the newest "
                             "(used by --resume and --export-actor-params)")
    parser.add_argument("--logdir", default=None,
                        help="TensorBoard directory (default: <checkpoint-dir>/tb)")
    parser.add_argument("--no-tensorboard", action="store_true",
                        help="disable TensorBoard logging entirely")
    parser.add_argument(
        "--export-actor-params", metavar="PATH", nargs="?", const="",
        help="restore --checkpoint-dir, write the actor-params pickle the robot "
             "loads via residual.runtime.params_path, and exit. Defaults to "
             "<checkpoint-dir>/actor_params.pkl.",
    )
    parser.add_argument(
        "--export-steps", metavar="STEPS", default=None,
        help="export several checkpoints at once and exit: 'all', or a comma "
             "list such as 12500,25000. Each is written to "
             "<checkpoint-dir>/actor_params_step<N>.pkl.",
    )
    parser.add_argument(
        "--image-size", type=int, default=None,
        help="encoder input, square. Must match the resolution the "
             "demonstrations were exported at (default: %d)"
             % LearnerConfig.image_size,
    )
    parser.add_argument("--resume", action="store_true",
                        help="restore from --checkpoint-dir before training")
    # -- offline replay of a finished session ----------------------------
    parser.add_argument(
        "--replay-online", nargs="+", metavar="PKL", default=None,
        help="online episodes from a finished Phase 2 session, exported with "
             "sparr_phase2_make_demos --include-unaccepted (failures included; "
             "a buffer of successes only teaches the critic nothing). Enables "
             "offline replay instead of waiting for a live actor.",
    )
    parser.add_argument(
        "--replay-mode", choices=("all", "schedule", "interval"), default="all",
        help="all: every episode present from step 0 -- cheap, but the policy "
             "never sees the early-data regime, so it does not predict a live "
             "result. schedule: release episodes at the steps they actually "
             "arrived at, read from --replay-schedule-tb. interval: release "
             "one every --replay-every steps.",
    )
    parser.add_argument(
        "--replay-schedule-tb", type=Path, default=None,
        help="the TensorBoard directory of the run being reproduced, e.g. "
             "<previous-checkpoint-dir>/tb. Required by --replay-mode schedule.",
    )
    parser.add_argument(
        "--replay-every", type=int, default=None,
        help="learner steps between episode arrivals, for --replay-mode interval",
    )
    parser.add_argument(
        "--offline", action="store_true",
        help="train on demonstrations alone, without waiting for an actor. "
             "A dry run: RLPD is designed around a live online buffer.",
    )
    parser.add_argument("--host", default="0.0.0.0",
                        help="interface to serve the actor link on")
    parser.add_argument("--port", type=int, default=5588)
    parser.add_argument("--broadcast-port", type=int, default=5589)
    return parser


def _checkpoint_steps(directory: Path) -> list[int]:
    """Steps of the flax checkpoints under ``directory``, ascending."""
    steps = []
    for entry in directory.glob("checkpoint_*"):
        match = re.fullmatch(r"checkpoint_(\d+)", entry.name)
        if match:
            steps.append(int(match.group(1)))
    return sorted(steps)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    exporting = args.export_actor_params is not None or args.export_steps is not None
    if not exporting and not args.demos:
        print("error: --demos is required for training", file=sys.stderr)
        return 2

    # When exporting, prefer the run's recorded config so the network shape matches.
    trained_config = Path(args.checkpoint_dir) / TRAINING_CONFIG_FILENAME
    config_path = (
        trained_config if exporting and trained_config.is_file() else args.config
    )
    spec = (
        Phase2Config.load(config_path).observation
        if config_path
        else ResidualObservationSpec()
    )
    overrides = {
        name: getattr(args, name)
        for name in ("batch_size", "seed", "cta_ratio", "training_starts",
                     "max_promotions", "image_size", "steps_per_update",
                     "checkpoint_period", "log_period")
        if getattr(args, name) is not None
    }
    config = LearnerConfig(**overrides)

    if args.replay_online and args.offline:
        print("error: --replay-online and --offline are different things and "
              "cannot be combined. --offline re-trains on the demonstrations "
              "themselves; --replay-online replays a recorded session's online "
              "data.", file=sys.stderr)
        return 2
    if args.replay_mode != "all" and not args.replay_online:
        print("error: --replay-mode needs --replay-online", file=sys.stderr)
        return 2

    learner = ResidualLearner(spec, config, checkpoint_dir=args.checkpoint_dir)

    if args.export_steps is not None:
        available = _checkpoint_steps(Path(args.checkpoint_dir))
        if args.export_steps.strip() == "all":
            wanted = available
        else:
            wanted = [int(x) for x in args.export_steps.split(",") if x.strip()]
        missing = sorted(set(wanted) - set(available))
        if missing or not wanted:
            print(f"error: no checkpoint for step(s) {missing or wanted} under "
                  f"{args.checkpoint_dir}; available: {available}", file=sys.stderr)
            return 2
        for step in wanted:
            learner.restore_checkpoint(step=step)
            out = learner.save_actor_params(
                Path(args.checkpoint_dir) / f"actor_params_step{step}.pkl"
            )
            print(f"step {step} -> {out}")
        return 0

    if args.export_actor_params is not None:
        step = learner.restore_checkpoint(step=args.checkpoint_step)
        # With --checkpoint-step the sidecar (newest checkpoint only) is skipped,
        # so report the requested step.
        restored = args.checkpoint_step if args.checkpoint_step is not None else step
        out = learner.save_actor_params(args.export_actor_params or None)
        print(f"restored checkpoint step {restored}; wrote actor params to {out}")
        return 0
    logdir = args.logdir or (
        str(Path(args.checkpoint_dir) / "tb") if args.checkpoint_dir else None
    )
    metrics = MetricsLogger(logdir, enabled=not args.no_tensorboard)
    if metrics.active:
        print(f"tensorboard: {metrics.logdir}")
        metrics.log_hparams({
            "demos": ",".join(str(d) for d in args.demos),
            "checkpoint_dir": str(args.checkpoint_dir),
            "state_dim": spec.state_dim(),
            "image_keys": ",".join(spec.image_keys),
            **{k: getattr(config, k) for k in (
                "batch_size", "cta_ratio", "training_starts", "discount",
                "demo_fraction", "seed", "image_size", "steps_per_update",
                "checkpoint_period", "replay_buffer_capacity")},
        })

    if args.config:
        # Lets evaluation recover observation layout and action scale from the params path.
        Path(args.checkpoint_dir).mkdir(parents=True, exist_ok=True)
        if Path(args.config).resolve() != trained_config.resolve():
            shutil.copyfile(args.config, trained_config)

    print(f"observation: {spec.state_dim()}-D [{spec.describe()}]")
    print(f"images: {', '.join(spec.image_keys)}")

    inserted = learner.load_demonstrations(args.demos)
    print(f"loaded {inserted} demonstration transitions from {len(args.demos)} file(s)")
    print(f"promotion median seeded at {learner.promoter.median}")

    if args.resume:
        step = learner.restore_checkpoint(step=args.checkpoint_step)
        print(f"resumed from step {step}")

    server = None
    replay = None
    if args.replay_online:
        from .buffers import load_demo_transitions
        from .offline_replay import ReplayDriver, build_schedule, split_episodes

        episodes = split_episodes(load_demo_transitions(args.replay_online))
        if not episodes:
            print("error: --replay-online contains no complete episode "
                  "(no transition has dones=True)", file=sys.stderr)
            return 2
        schedule = build_schedule(
            args.replay_mode,
            len(episodes),
            logdir=args.replay_schedule_tb,
            interval=args.replay_every,
        )
        replay = ReplayDriver(
            episodes=episodes,
            schedule=schedule,
            insert=lambda ep, promoted: learner.add_replayed_episode(
                ep, promoted=promoted
            ),
        )
        print(f"OFFLINE REPLAY ({args.replay_mode}): {replay.describe()}")
        if args.replay_mode == "all":
            print("  NOTE: every episode is present from step 0, so this run "
                  "does not reproduce the early-data regime a live session "
                  "trains through.")
        # Admit the step-0 arrivals, then as many more as readiness needs, as
        # a live run blocks in wait_until_ready before step 0.
        replay.release_due(0)
        while not learner.ready and replay.release_one():
            pass
        if not learner.ready:
            print(f"error: replayed every episode and still cannot fill a "
                  f"batch: {learner.waiting_on()}", file=sys.stderr)
            return 2
        print(f"  released {replay.released} episode(s) to reach readiness; "
              f"{replay.pending} pending")
    elif args.offline:
        # Seed the online buffer from the demonstrations so both halves can sample.
        print("OFFLINE DRY RUN: online buffer seeded from demonstrations; "
              "this is not a training run that produces a usable policy")
        from .buffers import load_demo_transitions, populate

        populate(learner.online_store, load_demo_transitions(args.demos), spec)
    else:
        from .transport import start_learner_server

        def on_stats(payload: dict) -> None:
            print(f"actor stats: {payload}")
            metrics.log_episode(payload)

        server = start_learner_server(
            learner.online_store,
            learner.demo_store,
            port=args.port,
            broadcast_port=args.broadcast_port,
            on_stats=on_stats,
        )
        print(f"serving actor link on {args.host}:{args.port} "
              f"(broadcast {args.broadcast_port})")
        print(f"waiting for data: {learner.waiting_on()}")
        learner.wait_until_ready(
            on_wait=lambda why: print(f"still waiting: {why}", flush=True)
        )

    stopping = False

    def request_stop(signum, frame):
        nonlocal stopping
        if stopping:
            # Second Ctrl-C: abort immediately.
            signal.signal(signal.SIGINT, signal.SIG_DFL)
            raise KeyboardInterrupt
        stopping = True
        print("\nstopping after the current step; press Ctrl-C again to abort")
        learner.config = LearnerConfig(**{**overrides, "max_steps": learner.state.step})

    signal.signal(signal.SIGINT, request_stop)

    def on_log(step: int, info: dict) -> None:
        losses = {k: round(v, 4) for k, v in info.items() if "loss" in k}
        print(f"step {step}: {losses}")
        metrics.log_train(step, info)
        metrics.log_buffers(
            step,
            online=len(learner.online_store),
            demo=len(learner.demo_store),
            promoted=learner.promoter.promoted_count,
            promotion_median=learner.promoter.median,
        )

    publish = server.publish_network if server is not None else None
    try:
        learner.on_checkpoint = metrics.log_checkpoint
        reached = learner.run(
            steps=args.steps,
            on_publish=publish,
            on_log=on_log,
            release=(replay.release_due if replay is not None else None),
        )
        print(f"stopped at step {reached}")
    except KeyboardInterrupt:
        print("\naborted")
    finally:
        metrics.close()
        try:
            path = learner.save_checkpoint()
            print(f"checkpoint written to {path}")
        except LearnerError as exc:
            print(f"could not checkpoint: {exc}", file=sys.stderr)
        if server is not None:
            from .transport import stop_quietly

            for problem in stop_quietly(server):
                print(f"warning: {problem}", file=sys.stderr)
    print(learner.describe())
    return 0


if __name__ == "__main__":
    sys.exit(main())
