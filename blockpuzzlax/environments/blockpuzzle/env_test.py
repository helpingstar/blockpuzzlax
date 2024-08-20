import pytest
import chex
import jax
import jax.numpy as jnp
from blockpuzzlax.environments.blockpuzzle.env import BlockPuzzle
from blockpuzzlax.environments.blockpuzzle.types import State
from blockpuzzlax.environments.blockpuzzle.constants import GRID_SIZE
from jumanji.testing.pytrees import assert_is_jax_array_tree
from jumanji.testing.env_not_smoke import (
    check_env_does_not_smoke,
    check_env_specs_does_not_smoke,
)
from jumanji.types import TimeStep
from blockpuzzlax.environments.blockpuzzle.test_data import step_terminate_dataset
from blockpuzzlax.environments.blockpuzzle import utils


@pytest.fixture
def block_puzzle() -> BlockPuzzle:
    return BlockPuzzle()


def test_block_puzzle__reset(block_puzzle: BlockPuzzle) -> None:
    chex.clear_trace_counter()
    reset_fn = jax.jit(chex.assert_max_traces(block_puzzle.reset, n=1))
    key = jax.random.PRNGKey(0)
    state, timestep = reset_fn(key)

    assert isinstance(timestep, TimeStep)
    assert isinstance(state, State)
    assert state.streak == 0
    assert state.score == 0
    assert jnp.array_equal(state.grid_padded[:GRID_SIZE, :GRID_SIZE], timestep.observation.grid)
    assert not jnp.array_equal(state.blocks[0], state.blocks[1])
    assert not jnp.array_equal(state.blocks[0], state.blocks[2])
    assert not jnp.array_equal(state.blocks[1], state.blocks[2])

    assert_is_jax_array_tree(state)
    assert_is_jax_array_tree(timestep)

    state, timestep = reset_fn(key)
    assert isinstance(timestep, TimeStep)
    assert isinstance(state, State)


@pytest.mark.parametrize("grid_padded, blocks, action", step_terminate_dataset())
def test_block_puzzle__step_terminate(
    grid_padded: chex.Array, blocks: chex.Array, action: chex.Array, block_puzzle: BlockPuzzle
) -> None:
    key = jax.random.key(0)
    # TODO: any idea to handle assert_max_traces? 8 is hardcoded for now
    step_fn = jax.jit(chex.assert_max_traces(block_puzzle.step, n=8))
    action_mask = jax.vmap(utils.block_action_mask, in_axes=(0, None))(blocks, grid_padded)
    state = State(grid_padded=grid_padded, action_mask=action_mask, key=key, blocks=blocks, streak=0, score=0)
    next_state, timestep = step_fn(state, action)

    assert jnp.array_equal(next_state.blocks, state.blocks.at[action[0]].set(jnp.zeros_like(state.blocks[action[0]])))
    assert timestep.last()


def test_block_puzzle__does_not_smoke(block_puzzle: BlockPuzzle) -> None:
    check_env_does_not_smoke(block_puzzle)


def test_block_puzzle__specs_does_not_smoke(block_puzzle: BlockPuzzle) -> None:
    check_env_specs_does_not_smoke(block_puzzle)
