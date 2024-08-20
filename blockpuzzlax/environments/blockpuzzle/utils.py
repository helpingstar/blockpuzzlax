from typing import Tuple
import chex
import jax
import jax.numpy as jnp
from blockpuzzlax.environments.blockpuzzle.constants import GRID_SIZE, BOX_IDX, BLOCK_WIDTH
from blockpuzzlax.environments.blockpuzzle.types import GridPadded, Grid, Blocks, Block
import functools


def sample_block_list(key: chex.PRNGKey, block_list: chex.Array) -> Blocks:
    indices = jnp.arange(len(block_list))
    block_indices = jax.random.choice(key, indices, shape=(3,), replace=False)
    blocks = block_list[block_indices]
    return blocks


def block_action_mask(block: chex.Array, padded_grid: chex.Array) -> Grid:
    def empty_block_action_mask(block: chex.Array):
        del block
        return jnp.full((GRID_SIZE, GRID_SIZE), False)

    def not_empty_block_action_mask(block: chex.Array):
        def is_valid(r, c):
            cropped_grid = jax.lax.dynamic_slice(padded_grid, (r, c), block.shape)
            return jax.lax.select(jnp.any(cropped_grid + block >= 2), False, True)

        r_index = jnp.arange(9)
        c_index = jnp.arange(9)

        action_mask = jax.vmap(jax.vmap(is_valid, in_axes=(None, 0)), in_axes=(0, None))(r_index, c_index)
        return action_mask

    one_block_action_mask = jax.lax.cond(
        jnp.sum(block) == 0, empty_block_action_mask, not_empty_block_action_mask, block
    )

    return one_block_action_mask


def grid_mask_and_combo(grid: Grid) -> Tuple[Grid, int]:
    def _is_filled(row: chex.Array):
        return jnp.sum(row) == GRID_SIZE

    def _next_row(row: chex.Array, is_filled: chex.Array, is_checked: chex.Array):
        return jax.lax.select(is_filled & ~is_checked, jnp.ones_like(row), row)

    def _get_combo(is_filled: chex.Array, is_checked: chex.Array):
        return jnp.sum(is_filled & ~is_checked)

    grid_transform_fn_list = (
        lambda x: x,
        lambda x: x.T,
        lambda x: jnp.take(x, jnp.asarray(BOX_IDX)),
    )

    def _check_transformed_grid(grid_masked: chex.Array, fn_index: int):
        grid_transform_fn = functools.partial(jax.lax.switch, fn_index, grid_transform_fn_list)
        filled_indices = jax.vmap(_is_filled)(grid_transform_fn(grid))
        checked_indices = jax.vmap(_is_filled)(grid_transform_fn(grid_masked))
        combo = _get_combo(filled_indices, checked_indices)
        grid_masked = jax.vmap(_next_row)(grid_transform_fn(grid_masked), filled_indices, checked_indices)
        grid_masked = grid_transform_fn(grid_masked)
        return grid_masked, combo

    grid_masked = jnp.zeros_like(grid)
    grid_masked, combo_accum = jax.lax.scan(
        _check_transformed_grid, grid_masked, jnp.arange(len(grid_transform_fn_list))
    )
    return grid_masked, jnp.sum(combo_accum)


def place_block(padded_grid: GridPadded, block: Block, r: int, c: int) -> GridPadded:
    cropped_padded_grid = jax.lax.dynamic_slice(padded_grid, (r, c), (BLOCK_WIDTH, BLOCK_WIDTH))
    placed_cropped_padded_grid = cropped_padded_grid + block
    placed_padded_grid = jax.lax.dynamic_update_slice(padded_grid, placed_cropped_padded_grid, (r, c))
    return placed_padded_grid


# TODO : combo is duplicated
def apply_valid_action(padded_grid: GridPadded, block: Block, action: chex.Array) -> Tuple[GridPadded, int]:
    """
    Assume receiving a valid action, apply the action, and if there are blocks to be cleaned, clean them and return the updated grid along with the combo.
    """
    r, c = action[1], action[2]
    block_placed_padded_grid = place_block(padded_grid, block, r, c)
    block_placed_grid = block_placed_padded_grid[:GRID_SIZE, :GRID_SIZE]
    grid_masked, combo = grid_mask_and_combo(block_placed_grid)
    updated_grid = block_placed_grid - grid_masked
    updated_padded_grid = jax.lax.dynamic_update_slice(block_placed_padded_grid, updated_grid, (0, 0))
    return updated_padded_grid, combo
