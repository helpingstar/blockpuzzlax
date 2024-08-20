import jax

import jax.numpy as jnp
from blockpuzzlax.environments.blockpuzzle.constants import BLOCKS_LIST, MAX_NUM_BLOCKS, BLOCK_WIDTH, GRID_SIZE
from blockpuzzlax.environments.blockpuzzle import utils
import chex
import pytest

from blockpuzzlax.environments.blockpuzzle.test_data import (
    grid_mask_and_combo_dataset,
    block_action_mask_dataset,
    apply_valid_action_dataset,
)


def test_sample_block_list() -> None:
    blocks_list = jnp.array(BLOCKS_LIST)
    sample_block_list_fn = jax.jit(utils.sample_block_list)
    key = jax.random.key(0)
    blocks = sample_block_list_fn(key, blocks_list)
    assert blocks.shape == (MAX_NUM_BLOCKS, BLOCK_WIDTH, BLOCK_WIDTH)
    for idx in range(MAX_NUM_BLOCKS):
        assert blocks[idx].sum() in range(1, 5 + 1)


def test_empty_block_action_mask() -> None:
    dummy_grid = jnp.zeros((GRID_SIZE, GRID_SIZE), dtype=jnp.int32)
    empty_block = jnp.zeros((5, 5), dtype=jnp.int32)
    block_action_mask_fn = jax.jit(utils.block_action_mask)
    block_action_mask = block_action_mask_fn(empty_block, dummy_grid)
    assert not jnp.any(block_action_mask)


@pytest.mark.parametrize("grid_padded, solution", block_action_mask_dataset())
def test_block_action_mask(grid_padded: chex.Array, solution: chex.Array, block: chex.Array) -> None:
    block_action_mask_fn = jax.jit(utils.block_action_mask)
    block_action_mask = block_action_mask_fn(block, grid_padded)
    assert jnp.all(block_action_mask == solution)


@pytest.mark.parametrize("sample, grid_solution, combo_solution", grid_mask_and_combo_dataset())
def test_grid_mask_and_combo(sample: chex.Array, grid_solution: chex.Array, combo_solution: int) -> None:
    grid_mask_and_combo_fn = jax.jit(utils.grid_mask_and_combo)
    grid_mask, combo = grid_mask_and_combo_fn(sample)
    assert jnp.all(grid_mask == grid_solution)
    assert combo == combo_solution


@pytest.mark.parametrize("grid_padded, action, grid_padded_solution, combo_solution", apply_valid_action_dataset())
def test_apply_valid_action(
    grid_padded: chex.Array,
    action: chex.Array,
    grid_padded_solution: chex.Array,
    combo_solution: chex.Array,
    block: chex.Array,
) -> None:
    apply_valid_action_fn = jax.jit(utils.apply_valid_action)
    updated_grid_padded, combo = apply_valid_action_fn(grid_padded, block, action)
    assert jnp.all(updated_grid_padded == grid_padded_solution)
    assert combo == combo_solution
