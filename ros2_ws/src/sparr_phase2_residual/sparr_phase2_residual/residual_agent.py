"""Construction and checkpointing of the residual SAC agent.

Builds the agent with ``serl_launcher``'s ``make_sac_pixel_agent``, makes the
pretrained encoder available offline, and saves/loads actor parameters.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Any, Optional

import numpy as np

from .observation_spec import ResidualObservationSpec
from .serl_compat import resnet10_pickle_compat

#: Where ``serl_launcher.utils.train_utils.load_resnet10_params`` looks.
SERL_CACHE = Path("~/.serl").expanduser()
RESNET10_FILENAME = "resnet10_params.pkl"

DEFAULT_IMAGE_SIZE = 256


class ResidualAgentError(RuntimeError):
    """Raised when the residual agent cannot be built or restored."""


def ensure_resnet10_cached() -> Path:
    """Link ``$SPARR_RESNET10_PARAMS`` into ``~/.serl``, where serl_launcher looks.

    If neither exists, serl_launcher downloads the encoder on demand.
    """
    target = SERL_CACHE / RESNET10_FILENAME
    if target.exists():
        return target
    baked = os.environ.get("SPARR_RESNET10_PARAMS")
    if not baked or not Path(baked).is_file():
        return target  # absent: serl_launcher will download on demand
    SERL_CACHE.mkdir(parents=True, exist_ok=True)
    try:
        target.symlink_to(baked)
    except OSError:
        shutil.copyfile(baked, target)
    return target


def make_residual_agent(
    spec: ResidualObservationSpec,
    *,
    seed: int = 0,
    image_size: int = DEFAULT_IMAGE_SIZE,
    encoder_type: str = "resnet-pretrained",
    discount: float = 0.97,
    reward_bias: float = 0.0,
    target_entropy: Optional[float] = None,
) -> Any:
    """Build the residual SAC agent for a given observation layout.

    ``encoder_type="resnet"`` is rejected: serl_launcher's ``ResNetEncoder``
    does not accept the ``encode`` keyword ``EncodingWrapper`` passes.
    """
    if encoder_type == "resnet":
        raise ResidualAgentError(
            "encoder_type='resnet' is broken in the vendored serl_launcher "
            "(ResNetEncoder.__call__ rejects the 'encode' kwarg EncodingWrapper "
            "passes it); use 'resnet-pretrained'"
        )
    ensure_resnet10_cached()
    from serl_launcher.utils.launcher import make_sac_pixel_agent

    # Leading axis is serl's frame-stack axis (T=1), not a batch axis.
    sample_obs: dict[str, np.ndarray] = {
        "state": np.zeros((1, spec.state_dim()), np.float32)
    }
    for key in spec.image_keys:
        sample_obs[key] = np.zeros((1, image_size, image_size, 3), np.uint8)

    with resnet10_pickle_compat():
        return make_sac_pixel_agent(
            seed,
            sample_obs,
            np.zeros((6,), np.float32),
            image_keys=tuple(spec.image_keys),
            encoder_type=encoder_type,
            discount=discount,
            reward_bias=reward_bias,
            target_entropy=target_entropy,
        )


def save_actor_params(agent: Any, path: str | Path) -> Path:
    """Write actor parameters atomically; the robot may read the path concurrently."""
    import pickle

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = destination.with_suffix(destination.suffix + ".partial")
    import jax

    with staging.open("wb") as handle:
        pickle.dump(jax.device_get(agent.state.params), handle)
    staging.replace(destination)
    return destination


def load_actor_params(path: str | Path) -> Any:
    """Read parameters written by :func:`save_actor_params`."""
    import pickle

    source = Path(path)
    if not source.is_file():
        raise ResidualAgentError(f"no residual parameters at {source}")
    with resnet10_pickle_compat(), source.open("rb") as handle:
        return pickle.load(handle)
