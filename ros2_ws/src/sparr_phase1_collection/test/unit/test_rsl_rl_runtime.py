# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the NVIDIA License.
# See the LICENSE file in the repository root for the full license text.

from types import SimpleNamespace

import numpy as np
import pytest

from sparr_phase1_collection.rsl_rl_runtime import (
    PolicyLoadError,
    _clear_recurrent_state,
    _flatten_recurrent_state,
    _load_weights,
)


class _FakeTensor:
    def __init__(self, values):
        self._values = np.asarray(values, dtype=np.float32)

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self._values


def test_missing_post_reset_lstm_state_is_recorded_as_fixed_size_zeros():
    model = SimpleNamespace(memory_a=SimpleNamespace(hidden_state=None))

    state = _flatten_recurrent_state(model, expected_size=12)

    np.testing.assert_array_equal(state, np.zeros(12, dtype=np.float32))


def test_recurrent_state_size_drift_is_rejected_before_hdf5_append():
    model = SimpleNamespace(
        memory_a=SimpleNamespace(
            hidden_state=(_FakeTensor(np.zeros(4)), _FakeTensor(np.zeros(4)))
        )
    )

    with pytest.raises(PolicyLoadError, match="8 values; expected 12"):
        _flatten_recurrent_state(model, expected_size=12)


class _BoolLoadResultModel:
    def __init__(self, keys, *, accepted=True):
        self._keys = tuple(keys)
        self._accepted = accepted
        self.loaded = None

    def load_state_dict(self, values, strict=True):
        self.loaded = (dict(values), strict)
        return self._accepted

    def state_dict(self):
        return {key: object() for key in self._keys}


def test_rsl_rl_three_boolean_load_result_preserves_actor_key_validation():
    model = _BoolLoadResultModel(
        ("actor.weight", "memory_a.rnn.weight", "critic.weight")
    )

    _load_weights(
        model,
        {
            "actor_state_dict": {
                "mlp.weight": object(),
                "rnn.rnn.weight": object(),
            }
        },
    )

    assert model.loaded is not None
    assert model.loaded[1] is False


def test_rsl_rl_three_boolean_load_result_rejects_missing_actor_key():
    model = _BoolLoadResultModel(("actor.weight", "memory_a.rnn.weight"))

    with pytest.raises(PolicyLoadError, match="missing actor checkpoint keys"):
        _load_weights(
            model,
            {"actor_state_dict": {"mlp.weight": object()}},
        )


def test_rsl_rl_three_false_load_result_is_rejected():
    model = _BoolLoadResultModel(("actor.weight",), accepted=False)

    with pytest.raises(PolicyLoadError, match="rejected the actor checkpoint"):
        _load_weights(
            model,
            {"actor_state_dict": {"mlp.weight": object()}},
        )


def test_warm_up_state_is_dropped_so_the_next_sample_starts_from_zeros():
    model = SimpleNamespace(
        memory_a=SimpleNamespace(
            hidden_state=(_FakeTensor(np.ones(6)), _FakeTensor(np.ones(6)))
        )
    )

    _clear_recurrent_state(model)

    assert model.memory_a.hidden_state is None
    np.testing.assert_array_equal(
        _flatten_recurrent_state(model, expected_size=12),
        np.zeros(12, dtype=np.float32),
    )
