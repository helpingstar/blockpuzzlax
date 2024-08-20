import jax

from blockpuzzlax.environments.blockpuzzle.reward import WoodokuReward

import pytest


@pytest.mark.parametrize(
    "combo, streak, n_block_cells, expected_reward",
    [
        (0, 0, 2, 2),
        (1, 1, 1, 19),
        (1, 1, 4, 22),
        (1, 1, 5, 23),
        (1, 2, 2, 30),
        (1, 3, 4, 42),
        (2, 1, 3, 49),
        (2, 1, 5, 51),
        (1, 2, 4, 32),
        (2, 2, 4, 60),
    ],
)
def test_woodoku_reward(combo, streak, n_block_cells, expected_reward):
    woodoku_reward = jax.jit(WoodokuReward())
    reward = woodoku_reward(combo, streak, n_block_cells)
    assert reward == expected_reward
