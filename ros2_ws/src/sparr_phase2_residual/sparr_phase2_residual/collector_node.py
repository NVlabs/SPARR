# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the NVIDIA OneWay Noncommercial License (March 22, 2022).
# See the LICENSE file in the repository root for the full license text:
# https://developer.download.nvidia.com/licenses/NVIDIA-OneWay-Noncommercial-License-22Mar2022.pdf

"""The Phase 2 collector: Phase 1's node with a residual on the action.

Overrides only action composition and the recording hooks that capture episodes
for the learner; everything else is inherited from :class:`Phase1CollectorNode`.

Safety: in the default shadow mode the commanded action is exactly Phase 1's.
A failed residual evaluation falls back to the Phase 1 action; only
``max_consecutive_failures`` in a row abort the trial.
"""

from __future__ import annotations

import os
from dataclasses import replace
from pathlib import Path
from typing import Any, Optional

import numpy as np
from sparr_phase1_collection.action_pipeline import ActionTrace
from sparr_phase1_collection.collector_node import Phase1CollectorNode

from .config import TRAINING_CONFIG_FILENAME, Phase2Config, evaluation_config
from .episode import EpisodeAccumulator, EpisodeError
from .promotion import DemoPromoter
from .observation_spec import ResidualObservationSpec
from .residual_policy import ResidualDecision, ResidualOutputStats, ResidualPolicy
from .retry import RetryDecision, RetryPolicy


#: Trial endings that say nothing about whether the policy can do the task.
_NO_VERDICT_REASONS = frozenset(
    {"operator_discarded", "operator_aborted_session", "node_shutdown"}
)

#: Returned for no-verdict endings, where the retry policy is not consulted.
_NO_RETRY = RetryDecision(
    repeat=False,
    force_promote=False,
    reason="trial discarded",
    success_rate=None,
    repeats_used=0,
)


class Phase2CollectorNode(Phase1CollectorNode):
    """Phase 1 collection with a learned residual on the composed action."""

    def __init__(self) -> None:
        super().__init__()
        self.declare_parameter("phase2_config_path", "")
        # Fixed-checkpoint evaluation.
        self.declare_parameter("eval_params_path", "")
        self.declare_parameter("eval_stochastic", False)
        self._phase2 = self._load_phase2_config()
        self._residual_failures = 0
        self._residual_decisions = 0
        self._residual_ok = 0
        #: Per-episode (reset in _finish_trial); the counters above are per session.
        self._residual_stats = ResidualOutputStats()
        self._residual: Optional[ResidualPolicy] = None
        self._build_residual()

        # Fed the same records as the HDF5 writer, so online transitions match
        # what demo_export reads back from the file.
        self._episode = EpisodeAccumulator(self._phase2.observation)
        self._last_mean: Optional[np.ndarray] = None
        self._pending_observation = False
        self._client = None
        self._actor_online = None
        self._actor_demo = None
        self._pending_record: Any = None
        self._promoter = (
            DemoPromoter(lengths=list(self._phase2.runtime.promotion_seed_lengths))
            if self._phase2.runtime.promote_episodes
            else None
        )
        self._retry = RetryPolicy(
            enabled=self._phase2.retry.enabled,
            success_rate_threshold=self._phase2.retry.success_rate_threshold,
            window=self._phase2.retry.window,
            max_repeats=self._phase2.retry.max_repeats,
            promote_on_success=self._phase2.retry.promote_on_success,
        )
        self._episodes_sent = 0
        #: Weights received mid-episode, applied at the next boundary.
        self._pending_params: Any = None
        self._connect_learner()

        runtime = self._phase2.runtime
        self.get_logger().info(
            f"Phase 2 residual: mode={runtime.mode} "
            f"scale={runtime.action_scale} "
            f"state={self._phase2.observation.state_dim()}-D "
            f"[{self._phase2.observation.describe()}]"
        )
        if self._phase2.retry.enabled:
            self.get_logger().info(self._retry.describe())
        if not runtime.commands_residual:
            self.get_logger().info(
                "residual shadow mode: the residual is recorded but never commanded"
            )
        if self._mode == "shadow" and runtime.learner_host:
            self.get_logger().warning(
                f"a learner is configured at {runtime.learner_host} but "
                "execution_mode is 'shadow', so NO transitions will be sent: "
                "Phase 1 marks actions non-training-eligible when no command "
                "reaches the robot. Use 'collect live' to feed the learner; the "
                "residual stays uncommanded as long as residual.runtime.mode is "
                "'shadow'."
            )

    # -- setup -----------------------------------------------------------

    def _load_phase2_config(self) -> Phase2Config:
        configured = str(self.get_parameter("phase2_config_path").value).strip()
        if not configured:
            configured = os.environ.get("SPARR_PHASE2_CONFIG", "")
        if not configured:
            # Default: beside the Phase 1 config.
            candidate = Path(self._config_path).with_name("phase2_live.yaml")
            if not candidate.is_file():
                raise FileNotFoundError(
                    "no Phase 2 configuration: set the phase2_config_path "
                    "parameter, the SPARR_PHASE2_CONFIG environment variable, "
                    f"or place phase2_live.yaml beside {self._config_path}"
                )
            configured = str(candidate)
        config = Phase2Config.load(configured)
        eval_params = str(self.get_parameter("eval_params_path").value).strip()
        if not eval_params:
            return config
        params = Path(eval_params)
        if not params.is_absolute():
            params = Path(self._repository_root) / params
        config, sidecar = evaluation_config(
            config, params, stochastic=bool(self.get_parameter("eval_stochastic").value)
        )
        self.get_logger().warning(
            f"EVALUATION of {params}: residual active, "
            f"{'stochastic' if config.runtime.deterministic is False else 'deterministic'}, "
            "no learner, no promotion, no retry"
        )
        if sidecar is not None:
            self.get_logger().info(f"observation and action_scale from {sidecar}")
        else:
            self.get_logger().warning(
                f"no {TRAINING_CONFIG_FILENAME} beside the params file (a run "
                f"from before it was recorded): using residual.image_keys, "
                f"state_terms and action_scale from {configured} -- they MUST "
                "match what this checkpoint was trained with"
            )
        return config

    def _build_residual(self) -> None:
        from .residual_agent import load_actor_params, make_residual_agent

        spec: ResidualObservationSpec = self._phase2.observation
        runtime = self._phase2.runtime
        agent = make_residual_agent(spec, seed=runtime.seed)
        if runtime.params_path:
            agent = agent.replace(
                state=agent.state.replace(params=load_actor_params(runtime.params_path))
            )
            self.get_logger().info(f"residual params loaded from {runtime.params_path}")
        elif runtime.commands_residual:
            # An untrained residual may only be commanded with a reachable
            # learner, or it would stay random indefinitely.
            if not (runtime.learner_host and runtime.wait_for_learner):
                raise ValueError(
                    "residual mode is 'active' with no params_path, so the "
                    "residual starts from its initialisation and relies on the "
                    "learner's broadcast. That needs a learner the actor is "
                    "known to have reached: set residual.runtime.learner_host "
                    "and wait_for_learner: true, or point params_path at an "
                    "exported actor_params.pkl."
                )
            self.get_logger().warning(
                "residual is ACTIVE with no params_path: it starts from its "
                "initialisation and commands the robot until the learner's "
                "first parameter broadcast arrives. action_scale="
                f"{runtime.action_scale} bounds how far it can push the base "
                "policy until then."
            )
        self._residual = ResidualPolicy(
            agent,
            spec,
            seed=runtime.seed,
            deterministic=runtime.deterministic,
            action_scale=runtime.action_scale,
        )

    def _connect_learner(self) -> None:
        """Open the actor side of the learner link, if configured; failure is not fatal."""
        runtime = self._phase2.runtime
        if not runtime.learner_host:
            self.get_logger().info(
                "no learner configured (residual.runtime.learner_host is unset); "
                "episodes will be recorded but not sent anywhere"
            )
            return
        try:
            from agentlace.data.data_store import QueuedDataStore

            from .transport import connect_actor_client

            self._actor_online = QueuedDataStore(runtime.actor_queue_capacity)
            self._actor_demo = QueuedDataStore(runtime.actor_queue_capacity)
            self._client = connect_actor_client(
                self._actor_online,
                self._actor_demo,
                host=runtime.learner_host,
                port=runtime.learner_port,
                broadcast_port=runtime.learner_broadcast_port,
                on_params=self._on_new_params,
                wait_for_server=runtime.wait_for_learner,
                timeout_ms=runtime.learner_timeout_ms,
            )
            self.get_logger().info(
                f"connected to learner at {runtime.learner_host}:{runtime.learner_port}"
            )
        except Exception as exc:  # noqa: BLE001 - never block a trial on this
            self._client = None
            self.get_logger().warning(
                f"could not reach the learner ({type(exc).__name__}: {exc}); "
                "continuing without it -- episodes will be recorded only"
            )

    def _extra_root_attrs(self) -> dict[str, Any]:
        """HDF5 root attributes identifying the residual that produced this session."""
        runtime = self._phase2.runtime
        return {
            "residual_mode": str(runtime.mode),
            "residual_params_path": str(runtime.params_path or ""),
            "residual_action_scale": float(runtime.action_scale),
            "residual_deterministic": bool(runtime.deterministic),
            "residual_seed": int(runtime.seed),
            "residual_image_keys": ",".join(self._phase2.observation.image_keys),
            "residual_state_dim": int(self._phase2.observation.state_dim()),
        }

    def _on_new_params(self, params: Any) -> None:
        """Hold learner-published weights until the next episode boundary.

        Keeps each rollout on a single parameter version; the newest push wins.
        """
        if self._residual is None:
            return
        self._pending_params = params
        if self._apply_pending_params(reason="no episode in progress"):
            return
        self.get_logger().info(
            "residual parameters received; holding until the episode ends"
        )

    def _episode_in_progress(self) -> bool:
        """True once a rollout has recorded anything the learner will see."""
        episode = getattr(self, "_episode", None)
        return bool(episode is not None and episode.states)

    def _apply_pending_params(self, *, reason: str) -> bool:
        """Swap in buffered weights if it is safe to. Returns whether it did."""
        if self._pending_params is None or self._residual is None:
            return False
        if self._episode_in_progress():
            return False
        params, self._pending_params = self._pending_params, None
        try:
            version = self._residual.update_params(params)
            self.get_logger().info(
                f"residual parameters updated (version {version}; {reason})"
            )
            return True
        except Exception as exc:  # noqa: BLE001 - a bad push must not kill the loop
            self.get_logger().error(
                f"rejected learner parameters ({type(exc).__name__}: {exc}); "
                "continuing on the previous ones"
            )
            return False

    def _compose_action(self, sample: Any, bundle: Any) -> ActionTrace:
        decision = self._evaluate_residual(sample, bundle)

        if decision is None or not decision.ok:
            # Fall back to Phase 1's composition, recorded as base-only.
            return super()._compose_action(sample, bundle)

        if self._phase2.runtime.commands_residual:
            from sparr_phase1_collection.action_pipeline import build_action_trace

            return build_action_trace(
                raw_mean=sample.raw_mean,
                raw_sample=sample.raw_sample,
                observed_flange_pose=bundle.snapshot.anchor_pose,
                flange_to_tcp=self._flange_to_tcp,
                config=self._action_config,
                deterministic=self._deterministic_inference,
                residual=decision.residual,
                residual_params_version=decision.params_version,
            )

        # Shadow: command Phase 1's action, record the requested residual.
        return replace(
            super()._compose_action(sample, bundle),
            requested_residual=np.asarray(decision.residual, dtype=np.float64),
            residual_applied=False,
            residual_params_version=decision.params_version,
        )

    def _evaluate_residual(self, sample: Any, bundle: Any) -> Optional[ResidualDecision]:
        if self._residual is None:
            return None
        try:
            terms = self._residual_terms(sample, bundle)
            images = self._residual_images(bundle)
        except Exception as exc:  # noqa: BLE001 - never break the tick
            self._note_residual_failure(f"observation unavailable: {exc}")
            return None

        decision = self._residual.evaluate(terms=terms, images=images)
        self._residual_decisions += 1
        if decision.ok:
            self._residual_ok += 1
            self._residual_failures = 0
            # Raw (unscaled) output, tracked in shadow mode too.
            self._residual_stats.observe(decision.raw_residual)
        else:
            self._note_residual_failure(decision.error)
        return decision

    def _note_residual_failure(self, reason: str) -> None:
        self._residual_failures += 1
        limit = self._phase2.runtime.max_consecutive_failures
        if self._residual_failures == 1 or self._residual_failures % 10 == 0:
            self.get_logger().warning(
                f"residual evaluation failed ({self._residual_failures}/{limit}): {reason}"
            )
        if self._residual_failures >= limit:
            self._reject_trial(
                f"residual_unavailable_for_{self._residual_failures}_steps", 0
            )

    #: Record paths of each state term; must match the HDF5 paths ``demo_export`` reads.
    _TERM_SOURCES = {
        "flange_pose": ("robot", "flange_pose_world_xyzw"),
        "wrench": ("robot", "ext_wrench_tcp"),
        "socket_pose": ("task", "socket_root_exact_world_xyzw"),
    }

    def _residual_terms(self, sample: Any, bundle: Any) -> dict[str, np.ndarray]:
        """The residual's state terms, read from the record being written.

        Uses recorded values rather than the snapshot so offline replay sees the
        identical vector.
        """
        record = bundle.record
        terms: dict[str, np.ndarray] = {}
        for name, _ in self._phase2.observation.terms():
            if name == "base_mean":
                # demo_export reads this from the action row aligned to this observation.
                terms[name] = np.asarray(sample.raw_mean, dtype=np.float64).reshape(-1)
                continue
            group, key = self._TERM_SOURCES[name]
            value = np.asarray(record[group][key], dtype=np.float64)
            if not np.all(np.isfinite(value)):
                raise ValueError(f"recorded {group}/{key} is not finite")
            terms[name] = value
        return terms

    def _residual_images(self, bundle: Any) -> dict[str, np.ndarray]:
        """The same cropped frames the recorder stores for this step."""
        cameras = bundle.record["camera"]
        frames: dict[str, np.ndarray] = {}
        for key in self._phase2.observation.image_keys:
            if key not in cameras:
                raise KeyError(f"no synchronized frame for camera {key!r}")
            frames[key] = np.asarray(cameras[key]["rgb"])
        return frames

    def _append_observation(self, record: Any) -> None:
        super()._append_observation(record)
        if self._last_mean is None:
            # The trial's first observation precedes inference; hold it until
            # its base mean arrives with the action.
            self._pending_record = record
            self._pending_observation = True
            return
        self._capture_observation(record, self._last_mean)

    def _append_action(self, record: Any) -> None:
        super()._append_action(record)
        mean = np.asarray(record["policy"]["mu_raw_normalized"], np.float64)

        if self._pending_observation:
            self._capture_observation(self._pending_record, mean)
            self._pending_observation = False
        self._last_mean = mean

        # Realized post-clip residual, as in demo_export. NaN marks a
        # non-training-eligible step.
        label = np.asarray(
            record["labels"]["residual_effective_software_action_minus_raw_mean"],
            np.float64,
        )
        if not np.all(np.isfinite(label)):
            # Drop the paired observation to keep N observations == M actions + 1.
            if self._episode.states:
                self._episode.states.pop()
                self._episode.images.pop()
                self._episode.tcp_positions.pop()
            return
        self._episode.act(label)

    def _capture_observation(self, record: Any, mean: np.ndarray) -> None:
        # Episode start: apply pending weights.
        if not self._episode.states:
            self._apply_pending_params(reason="episode start")
        try:
            self._episode.observe(record, mean)
        except Exception as exc:  # noqa: BLE001 - recording must not stop a trial
            self.get_logger().warning(
                f"could not capture an observation for training "
                f"({type(exc).__name__}: {exc}); this episode will be dropped"
            )
            self._episode.reset()

    def _finish_trial(self, *, accepted: bool, **kwargs: Any) -> Any:
        # Order matters: retry affects promotion, and Phase 1's _finish_trial
        # reads _repeat_targets and resets trial state.
        decision = self._decide_retry(accepted=accepted, **kwargs)
        self._ship_episode(accepted=accepted, force_promote=decision.force_promote)
        self._last_mean = None
        self._pending_observation = False
        self._residual_stats.reset()
        return super()._finish_trial(accepted=accepted, **kwargs)

    def _ship_episode(self, *, accepted: bool, force_promote: bool = False) -> None:
        """Send the finished episode to the learner; promoted episodes go on both streams."""
        length = self._episode.steps
        if length == 0:
            if self._mode == "shadow":
                self.get_logger().warning(
                    "no transitions to send: execution_mode is 'shadow', so no "
                    "command reached the robot and every action is marked "
                    "non-training-eligible. The residual's own shadow mode is a "
                    "SEPARATE setting (residual.runtime.mode) and is what you "
                    "want for a safe run: collect live with the residual in "
                    "shadow drives the robot with the base policy exactly as "
                    "Phase 1 does, while recording the residual without applying it."
                )
            else:
                self.get_logger().warning(
                    "no transitions to send: the episode had no training-eligible "
                    "steps. Check labels/command_valid in the recording."
                )
            self._episode.reset()
            return
        try:
            transitions = self._episode.finish(
                accepted=accepted,
                info={"trial": self._episode_label()},
            )
        except EpisodeError as exc:
            self.get_logger().warning(f"episode not usable for training: {exc}")
            self._episode.reset()
            return

        promoted = accepted and self._should_promote(
            len(transitions), force=force_promote
        )
        if self._actor_online is not None:
            for transition in transitions:
                self._actor_online.insert(transition)
                if promoted:
                    self._actor_demo.insert(transition)
        self._episode.reset()
        self._apply_pending_params(reason="episode boundary")

        if self._client is None:
            self.get_logger().info(
                f"episode complete: {len(transitions)} transitions "
                f"(from {length} steps), not sent -- no learner connected"
                f"{self._residual_summary_text()}"
            )
            return
        try:
            self._client.update()
            self._episodes_sent += 1
            self.get_logger().info(
                f"sent {len(transitions)} transitions to the learner"
                f"{' (promoted to demonstrations)' if promoted else ''}"
                f"{self._residual_summary_text()}"
            )
            self._client.request(
                "send-stats",
                {
                    "succeed": bool(accepted),
                    "length": len(transitions),
                    "promoted": bool(promoted),
                    "residual_ok_fraction": (
                        self._residual_ok / self._residual_decisions
                        if self._residual_decisions
                        else 0.0
                    ),
                    **self._residual_stats.summary(
                        action_scale=self._phase2.runtime.action_scale
                    ),
                },
            )
        except Exception as exc:  # noqa: BLE001 - a dead learner must not stop us
            self.get_logger().warning(
                f"could not reach the learner ({type(exc).__name__}: {exc}); "
                "the episode is still recorded to HDF5"
            )

    def _decide_retry(self, *, accepted: bool, **kwargs: Any) -> Any:
        """Ask the retry policy what to do, and arm Phase 1 to replay if so.

        ``_repeat_targets`` makes the next attempt reuse this trial's socket
        offset and start verbatim, bypassing the sampler.
        """
        # Rejected before its rollout: the same attempt runs again, so leave the
        # retry state (including an armed replay) exactly as it is.
        if self._rejected_before_rollout:
            return _NO_RETRY
        # No-verdict endings neither move the success window nor consume a repeat.
        if str(kwargs.get("termination_reason", "")) in _NO_VERDICT_REASONS:
            self._retry.abandon()
            self._repeat_targets = None
            return _NO_RETRY

        decision = self._retry.consider(succeeded=bool(accepted))
        if decision.repeat and self._targets is not None:
            self._repeat_targets = self._targets
            self.get_logger().info(f"retry: {decision}")
        else:
            if decision.force_promote or self._repeat_targets is not None:
                self.get_logger().info(f"retry: {decision}")
            self._repeat_targets = None
        return decision

    def _residual_summary_text(self) -> str:
        """One-line residual summary for the episode log, "" when there is none."""
        summary = self._residual_stats.summary(
            action_scale=self._phase2.runtime.action_scale
        )
        if not summary:
            return ""
        return (
            f"  |  residual |mean| {summary['residual_abs_mean']:.4f} raw / "
            f"{summary['residual_abs_mean_scaled']:.4f} scaled, "
            f"max {summary['residual_max_abs']:.3f}, "
            f"saturation {summary['residual_saturation']:.1%}"
        )

    def _should_promote(self, length: int, *, force: bool = False) -> bool:
        """Running-median promotion; ``force`` skips the length criterion."""
        if self._promoter is None:
            return False
        return self._promoter.consider(
            [{"rewards": 1.0, "dones": True}] * length, succeeded=True, force=force
        ).promoted

    def _episode_label(self) -> str:
        return f"{self.get_name()}#{self._episodes_sent}"


__all__ = ["Phase2CollectorNode"]
