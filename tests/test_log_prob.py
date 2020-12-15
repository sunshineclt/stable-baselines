import pytest
import numpy as np
from gym.spaces.box import Box

from stable_baselines import A2C, PPO1, PPO2, TRPO
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines.common.identity_env import IdentityEnvBox


class Helper:
    @staticmethod
    def proba_vals(obs, state, mask):
        return np.array([-0.4]), np.array([[0.1]])


@pytest.fixture
def helpers():
    return Helper


@pytest.mark.parametrize("model_class", [A2C, PPO1, PPO2, TRPO])
def test_log_prob_calcuation(model_class, helpers):
    env = DummyVecEnv([lambda: IdentityEnvBox()])
    model = model_class('MlpPolicy', env)

    model.proba_step = helpers.proba_vals

    logprob = model.action_probability(np.array([[0.5],[0.5]]), 1, [False], 0.2, True)
    assert np.all(logprob == np.array([-16.616353440210627])), \
        "Calculation failed for {}".format(model_class)