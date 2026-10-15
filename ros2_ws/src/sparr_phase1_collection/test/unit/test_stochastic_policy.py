import pytest

from sparr_phase1_collection.stochastic_policy import RecurrentStochasticPolicy

torch = pytest.importorskip("torch")


class _CountingRecurrentActor:
    def __init__(self):
        self.forward_count = 0
        self.hidden_state = 0
        self.reset_count = 0
        self.action_mean = None
        self.action_std = None

    def act(self, observation):
        self.forward_count += 1
        self.hidden_state += 1
        # Mean depends on recurrent state, making a second advance visible.
        self.action_mean = observation + float(self.hidden_state)
        self.action_std = torch.tensor([0.5] * 6, device=observation.device)
        noise = torch.tensor(
            [[4.0, -4.0, 0.5, 1.0, -1.0, 0.0]], device=observation.device
        )
        return self.action_mean + self.action_std * noise

    def reset(self, dones=None):
        self.reset_count += 1
        self.hidden_state = 0


def test_one_act_call_exposes_unclipped_mean_std_and_sample_without_double_lstm_advance():
    actor = _CountingRecurrentActor()
    policy = RecurrentStochasticPolicy(actor)
    observation = torch.zeros((1, 6))

    trace = policy.sample_once(observation)

    assert actor.forward_count == 1
    assert actor.hidden_state == 1
    torch.testing.assert_close(trace.raw_mean, torch.ones((1, 6)))
    torch.testing.assert_close(trace.raw_std, torch.full((1, 6), 0.5))
    torch.testing.assert_close(
        trace.raw_sample,
        torch.tensor([[3.0, -1.0, 1.25, 1.5, 0.5, 1.0]]),
    )
    # The adapter does not clip the sample; clipping belongs to action_pipeline.
    assert trace.raw_sample[0, 0] > 1.0


def test_reset_clears_recurrent_state_once_per_episode_boundary():
    actor = _CountingRecurrentActor()
    policy = RecurrentStochasticPolicy(actor)
    policy.sample_once(torch.zeros((1, 6)))

    policy.reset()

    assert actor.reset_count == 1
    assert actor.hidden_state == 0
