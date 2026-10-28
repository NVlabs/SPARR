# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the NVIDIA OneWay Noncommercial License (March 22, 2022).
# See the LICENSE file in the repository root for the full license text:
# https://developer.download.nvidia.com/licenses/NVIDIA-OneWay-Noncommercial-License-22Mar2022.pdf

"""The actor/learner wire contract: constants both ends of the link agree on.

The socket-level loopback test lives in ``docker/phase2/verify-transport.py``
because agentlace never terminates its ZeroMQ contexts, which would hang pytest.
"""

from __future__ import annotations

import pytest

pytest.importorskip("agentlace")
pytest.importorskip("gymnasium")

from sparr_phase2_residual.transport import (  # noqa: E402
    DEFAULT_BROADCAST_PORT,
    DEFAULT_PORT,
    DEMO_STORE,
    ONLINE_STORE,
    stop_quietly,
    trainer_config,
)

def test_the_two_ends_agree_on_the_wire_contract():
    """agentlace refuses to connect if these differ, so pin them."""
    config = trainer_config()
    assert config.port_number == 5588
    assert config.broadcast_port == 5589
    assert config.request_types == ["send-stats"]
    assert ONLINE_STORE == "actor_env"
    assert DEMO_STORE == "actor_env_ideal"


def test_stop_quietly_tolerates_an_endpoint_that_never_returns():
    """A learner that has written its checkpoint must not hang on shutdown."""
    import threading

    class Wedged:
        def stop(self):
            threading.Event().wait()  # never returns

    problems = stop_quietly(Wedged(), timeout_s=0.2)
    assert len(problems) == 1 and "did not return" in problems[0]


def test_stop_quietly_reports_a_raising_endpoint():
    class Broken:
        def stop(self):
            raise OSError("socket already closed")

    problems = stop_quietly(Broken(), timeout_s=1.0)
    assert len(problems) == 1 and "OSError" in problems[0]


def test_stop_quietly_is_clean_for_well_behaved_endpoints():
    class Fine:
        def __init__(self):
            self.stopped = False

        def stop(self):
            self.stopped = True

    endpoint = Fine()
    assert stop_quietly(endpoint, None) == []
    assert endpoint.stopped
