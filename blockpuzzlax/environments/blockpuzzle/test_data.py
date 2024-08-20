import chex
import jax.numpy as jnp
from typing import List, Tuple, Iterator
from blockpuzzlax.environments.blockpuzzle.constants import BLOCK_WIDTH
from blockpuzzlax.environments.blockpuzzle.types import Grid, GridPadded, Blocks


def pad_grid(grid_list: List[Grid]) -> List[GridPadded]:
    grid_padded_list = []
    for grid in grid_list:
        grid_padded = jnp.pad(
            grid, pad_width=((0, BLOCK_WIDTH - 1), (0, BLOCK_WIDTH - 1)), mode="constant", constant_values=1
        )
        grid_padded_list.append(grid_padded)
    return grid_padded_list


def grid_random_samples() -> List[Grid]:
    grid_0 = jnp.array(
        [
            [1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1],
        ]
    )

    grid_1 = jnp.array(
        [
            [1, 0, 1, 0, 1, 0, 1, 0, 1],
            [0, 1, 0, 1, 0, 1, 0, 1, 0],
            [1, 0, 1, 0, 1, 0, 1, 0, 1],
            [0, 1, 0, 1, 0, 1, 0, 1, 0],
            [1, 0, 1, 0, 1, 0, 1, 0, 1],
            [0, 1, 0, 1, 0, 1, 0, 1, 0],
            [1, 0, 1, 0, 1, 0, 1, 0, 1],
            [0, 1, 0, 1, 0, 1, 0, 1, 0],
            [1, 0, 1, 0, 1, 0, 1, 0, 1],
        ]
    )

    grid_2 = jnp.array(
        [
            [1, 0, 1, 0, 1, 0, 1, 0, 1],
            [0, 1, 0, 1, 0, 1, 0, 1, 0],
            [1, 0, 1, 0, 1, 0, 1, 0, 1],
            [0, 1, 0, 1, 0, 1, 0, 1, 0],
            [1, 0, 1, 0, 1, 0, 1, 0, 1],
            [0, 1, 0, 1, 0, 1, 0, 1, 0],
            [1, 0, 1, 1, 1, 1, 1, 0, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 0, 1, 1, 1, 1, 1, 0, 1],
        ]
    )

    grid_3 = jnp.array(
        [
            [1, 0, 1, 0, 1, 0, 1, 0, 1],
            [0, 1, 0, 1, 0, 1, 0, 1, 0],
            [1, 0, 1, 0, 1, 0, 1, 1, 1],
            [0, 1, 0, 1, 0, 1, 1, 1, 1],
            [1, 0, 1, 0, 1, 0, 1, 1, 1],
            [0, 1, 0, 1, 0, 1, 0, 1, 0],
            [1, 1, 1, 1, 1, 1, 1, 1, 0],
            [0, 1, 1, 1, 0, 1, 1, 1, 0],
            [1, 1, 1, 1, 1, 1, 1, 1, 0],
        ]
    )

    grid_4 = jnp.array(
        [
            [1, 1, 1, 1, 0, 1, 1, 1, 1],
            [1, 0, 1, 1, 0, 1, 1, 0, 1],
            [1, 1, 1, 1, 0, 1, 1, 1, 1],
            [1, 1, 0, 1, 0, 1, 0, 1, 1],
            [1, 1, 1, 1, 0, 1, 1, 1, 1],
            [1, 1, 1, 1, 0, 1, 1, 1, 1],
            [0, 1, 1, 1, 0, 1, 1, 1, 1],
            [1, 1, 1, 1, 0, 1, 1, 1, 1],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
        ]
    )

    grid_5 = jnp.array(
        [
            [1, 1, 0, 0, 1, 0, 0, 0, 0],
            [1, 1, 0, 0, 1, 0, 0, 0, 0],
            [1, 1, 0, 0, 1, 0, 0, 0, 0],
            [1, 1, 0, 0, 1, 0, 1, 1, 1],
            [1, 1, 1, 1, 1, 0, 0, 0, 0],
            [1, 1, 0, 0, 1, 0, 0, 0, 0],
            [1, 1, 0, 0, 0, 0, 0, 0, 0],
            [1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1],
        ]
    )

    grid_6 = jnp.array(
        [
            [0, 0, 1, 0, 1, 0, 0, 0, 0],
            [1, 0, 1, 1, 0, 1, 0, 0, 1],
            [0, 0, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 1, 0, 1],
            [1, 1, 1, 0, 0, 1, 0, 0, 1],
            [1, 0, 0, 0, 1, 0, 1, 1, 0],
            [0, 0, 0, 1, 0, 0, 1, 1, 0],
            [1, 1, 0, 0, 1, 0, 0, 1, 1],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
        ]
    )

    grid_sample_list = []
    grid_sample_list.append(grid_0)
    grid_sample_list.append(grid_1)
    grid_sample_list.append(grid_2)
    grid_sample_list.append(grid_3)
    grid_sample_list.append(grid_4)
    grid_sample_list.append(grid_5)
    grid_sample_list.append(grid_6)
    return grid_sample_list


def grid_random_samples_for_apply_valid_action() -> List[Grid]:
    grid_0 = jnp.array(
        [
            [1, 1, 0, 1, 1, 1, 0, 1, 0],
            [1, 1, 0, 1, 0, 1, 0, 0, 0],
            [1, 0, 1, 1, 0, 0, 1, 1, 1],
            [1, 0, 1, 1, 0, 0, 1, 0, 0],
            [1, 1, 1, 0, 1, 1, 1, 1, 1],
            [1, 1, 0, 1, 0, 0, 1, 0, 0],
            [0, 0, 1, 0, 1, 1, 1, 0, 0],
            [1, 0, 0, 1, 1, 1, 0, 1, 1],
            [0, 0, 1, 0, 0, 1, 0, 1, 1],
        ]
    )

    grid_1 = jnp.array(
        [
            [0, 1, 0, 1, 1, 0, 0, 1, 0],
            [0, 1, 0, 1, 1, 0, 1, 1, 1],
            [1, 1, 1, 0, 0, 1, 0, 0, 0],
            [1, 0, 0, 1, 1, 0, 1, 0, 0],
            [0, 0, 0, 1, 0, 1, 1, 1, 0],
            [1, 1, 0, 0, 1, 0, 1, 0, 0],
            [0, 1, 0, 1, 0, 0, 0, 0, 1],
            [0, 0, 1, 0, 0, 1, 0, 0, 1],
            [0, 1, 0, 0, 0, 1, 0, 1, 1],
        ]
    )

    grid_2 = jnp.array(
        [
            [1, 1, 0, 0, 1, 1, 1, 1, 1],
            [1, 1, 0, 1, 1, 1, 1, 0, 0],
            [1, 0, 1, 1, 1, 0, 1, 1, 1],
            [1, 0, 1, 1, 1, 0, 1, 1, 0],
            [0, 1, 1, 1, 0, 1, 0, 1, 0],
            [0, 0, 0, 1, 1, 0, 0, 0, 0],
            [0, 1, 0, 0, 1, 1, 1, 0, 1],
            [0, 0, 0, 1, 1, 0, 1, 1, 1],
            [1, 0, 0, 0, 1, 1, 1, 1, 0],
        ]
    )

    grid_3 = jnp.array(
        [
            [0, 0, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 1, 0, 1, 0, 0, 0, 0],
            [1, 1, 0, 0, 1, 1, 1, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 1, 1, 0, 1, 1, 1],
            [0, 1, 1, 0, 1, 0, 1, 0, 1],
            [1, 0, 0, 0, 1, 1, 0, 1, 0],
            [1, 0, 0, 1, 1, 0, 0, 1, 0],
            [1, 0, 1, 1, 0, 0, 1, 0, 1],
        ]
    )

    grid_4 = jnp.array(
        [
            [1, 1, 0, 1, 0, 0, 1, 0, 0],
            [0, 1, 0, 0, 0, 1, 1, 0, 0],
            [0, 1, 0, 1, 1, 0, 1, 0, 0],
            [1, 0, 0, 0, 0, 0, 0, 1, 0],
            [0, 1, 0, 0, 0, 1, 0, 0, 1],
            [0, 0, 1, 0, 0, 1, 0, 1, 0],
            [1, 0, 0, 1, 1, 1, 1, 0, 0],
            [1, 1, 0, 0, 0, 1, 1, 0, 0],
            [0, 1, 0, 1, 0, 1, 1, 0, 1],
        ]
    )

    grid_5 = jnp.array(
        [
            [0, 1, 1, 0, 1, 0, 0, 1, 1],
            [0, 1, 1, 1, 1, 0, 1, 0, 0],
            [0, 0, 0, 0, 1, 0, 1, 0, 0],
            [0, 1, 0, 0, 0, 1, 0, 0, 1],
            [0, 0, 0, 1, 1, 0, 1, 0, 1],
            [0, 0, 1, 0, 1, 0, 0, 0, 0],
            [1, 0, 0, 0, 0, 1, 0, 0, 1],
            [1, 1, 1, 1, 0, 1, 1, 1, 1],
            [0, 0, 0, 0, 0, 1, 0, 0, 0],
        ]
    )

    grid_6 = jnp.array(
        [
            [0, 1, 1, 0, 0, 0, 0, 1, 0],
            [1, 1, 1, 0, 1, 0, 0, 0, 1],
            [1, 0, 0, 1, 0, 0, 0, 0, 0],
            [0, 1, 1, 1, 1, 0, 1, 1, 1],
            [1, 0, 1, 1, 1, 0, 0, 1, 1],
            [1, 1, 0, 0, 0, 0, 0, 1, 0],
            [1, 0, 1, 1, 1, 0, 0, 1, 1],
            [1, 1, 0, 1, 1, 1, 1, 1, 1],
            [1, 1, 0, 1, 1, 1, 0, 1, 1],
        ]
    )

    grid_sample_list = []
    grid_sample_list.append(grid_0)
    grid_sample_list.append(grid_1)
    grid_sample_list.append(grid_2)
    grid_sample_list.append(grid_3)
    grid_sample_list.append(grid_4)
    grid_sample_list.append(grid_5)
    grid_sample_list.append(grid_6)
    return grid_sample_list


def step_terminate_data_grid() -> List[Grid]:
    grid_0 = jnp.array(
        [
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 1, 1, 1, 1, 1, 1, 1],
            [0, 1, 1, 0, 1, 1, 1, 0, 1],
            [0, 0, 1, 1, 0, 0, 1, 0, 0],
            [0, 1, 1, 1, 1, 1, 1, 0, 0],
            [1, 1, 1, 1, 1, 1, 0, 0, 0],
            [0, 1, 0, 1, 0, 1, 1, 0, 0],
            [0, 0, 0, 1, 1, 1, 1, 0, 1],
        ]
    )

    grid_1 = jnp.array(
        [
            [0, 1, 1, 1, 1, 1, 1, 0, 0],
            [1, 1, 1, 0, 1, 0, 1, 1, 1],
            [1, 0, 1, 1, 1, 0, 1, 0, 0],
            [1, 1, 1, 1, 1, 1, 1, 0, 0],
            [1, 0, 1, 0, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 0],
            [1, 0, 0, 1, 1, 1, 0, 0, 1],
            [1, 1, 1, 1, 0, 1, 1, 1, 0],
            [0, 0, 1, 1, 1, 0, 1, 1, 0],
        ]
    )
    grid_2 = jnp.array(
        [
            [0, 0, 1, 0, 0, 1, 0, 1, 0],
            [1, 1, 1, 1, 0, 1, 1, 1, 1],
            [0, 0, 0, 0, 0, 1, 1, 0, 0],
            [1, 1, 0, 1, 1, 1, 0, 1, 0],
            [0, 0, 0, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 0, 0, 1, 0],
            [0, 1, 1, 0, 1, 1, 1, 0, 0],
            [0, 0, 1, 1, 1, 1, 1, 1, 1],
            [0, 1, 1, 0, 0, 1, 0, 0, 0],
        ]
    )
    grid_3 = jnp.array(
        [
            [0, 1, 0, 0, 1, 1, 1, 0, 1],
            [0, 1, 1, 1, 0, 0, 1, 0, 1],
            [0, 1, 1, 1, 1, 1, 0, 0, 1],
            [1, 1, 1, 1, 0, 1, 0, 1, 1],
            [1, 1, 1, 1, 1, 1, 0, 1, 1],
            [0, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 0, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 0, 1, 0, 1],
            [0, 0, 1, 1, 1, 1, 1, 0, 0],
        ]
    )
    grid_4 = jnp.array(
        [
            [1, 1, 0, 0, 0, 0, 0, 1, 0],
            [1, 1, 1, 0, 1, 1, 0, 1, 1],
            [1, 1, 1, 1, 1, 1, 0, 1, 0],
            [0, 0, 0, 0, 1, 1, 1, 0, 0],
            [1, 1, 1, 0, 1, 1, 1, 1, 0],
            [0, 1, 1, 1, 1, 1, 0, 0, 1],
            [0, 1, 1, 0, 0, 1, 0, 1, 1],
            [0, 0, 1, 0, 1, 1, 0, 0, 1],
            [0, 1, 0, 1, 1, 1, 0, 0, 1],
        ]
    )
    grid_5 = jnp.array(
        [
            [0, 0, 0, 0, 0, 1, 1, 1, 1],
            [1, 0, 0, 1, 0, 1, 0, 1, 0],
            [0, 1, 0, 1, 0, 0, 0, 1, 1],
            [1, 1, 1, 0, 0, 0, 0, 0, 0],
            [1, 0, 0, 1, 1, 1, 1, 1, 0],
            [1, 0, 1, 1, 1, 0, 1, 1, 1],
            [1, 0, 1, 1, 1, 1, 1, 0, 1],
            [1, 1, 1, 1, 0, 1, 1, 1, 1],
            [0, 0, 0, 1, 1, 1, 1, 1, 1],
        ]
    )
    grid_6 = jnp.array(
        [
            [0, 0, 0, 1, 0, 1, 1, 1, 1],
            [0, 1, 1, 1, 1, 0, 0, 1, 1],
            [1, 1, 1, 1, 1, 1, 0, 1, 1],
            [1, 0, 1, 0, 0, 1, 0, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 0, 1],
            [1, 1, 1, 1, 1, 1, 1, 0, 1],
            [0, 1, 1, 0, 1, 1, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 0, 0],
            [0, 1, 1, 1, 0, 1, 1, 1, 0],
        ]
    )
    grid_7 = jnp.array(
        [
            [1, 1, 1, 1, 1, 1, 0, 0, 0],
            [0, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 0, 1, 1, 1, 1],
            [0, 0, 1, 1, 1, 1, 1, 0, 1],
            [1, 0, 1, 0, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 0, 0, 1, 1, 1],
            [1, 1, 1, 0, 1, 1, 1, 1, 0],
            [0, 1, 1, 1, 1, 1, 1, 1, 1],
            [0, 0, 0, 0, 0, 1, 1, 1, 0],
        ]
    )

    grid_sample_list = []
    grid_sample_list.append(grid_0)
    grid_sample_list.append(grid_1)
    grid_sample_list.append(grid_2)
    grid_sample_list.append(grid_3)
    grid_sample_list.append(grid_4)
    grid_sample_list.append(grid_5)
    grid_sample_list.append(grid_6)
    grid_sample_list.append(grid_7)
    return grid_sample_list


def step_terminate_data_blocks() -> List[Blocks]:
    blocks_0 = jnp.array(
        [
            [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]],
            [[1, 1, 0, 0, 0], [1, 0, 0, 0, 0], [1, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]],
            [[0, 0, 1, 0, 0], [0, 0, 1, 0, 0], [1, 1, 1, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]],
        ],
        dtype=jnp.int32,
    )
    blocks_1 = jnp.array(
        [
            [[1, 0, 0, 0, 0], [1, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]],
            [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]],
            [[1, 0, 0, 0, 0], [1, 1, 0, 0, 0], [1, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]],
        ],
        dtype=jnp.int32,
    )

    blocks_2 = jnp.array(
        [
            [[0, 1, 0, 0, 0], [0, 1, 0, 0, 0], [1, 1, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]],
            [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]],
            [[1, 1, 0, 0, 0], [1, 1, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]],
        ],
        dtype=jnp.int32,
    )

    blocks_3 = jnp.array(
        [
            [[0, 1, 0, 0, 0], [1, 1, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]],
            [[1, 0, 0, 0, 0], [1, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]],
            [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]],
        ],
        dtype=jnp.int32,
    )

    blocks_4 = jnp.array(
        [
            [[1, 1, 1, 0, 0], [1, 0, 1, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]],
            [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]],
            [[0, 1, 0, 0, 0], [1, 1, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]],
        ],
        dtype=jnp.int32,
    )

    blocks_5 = jnp.array(
        [
            [[1, 1, 1, 0, 0], [0, 0, 1, 0, 0], [0, 0, 1, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]],
            [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]],
            [[0, 1, 0, 0, 0], [1, 1, 1, 0, 0], [0, 1, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]],
        ],
        dtype=jnp.int32,
    )

    blocks_6 = jnp.array(
        [
            [[1, 1, 0, 0, 0], [1, 1, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]],
            [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]],
            [[1, 1, 1, 0, 0], [0, 1, 0, 0, 0], [0, 1, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]],
        ],
        dtype=jnp.int32,
    )

    blocks_7 = jnp.array(
        [
            [[1, 0, 0, 0, 0], [1, 1, 1, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]],
            [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]],
            [[1, 1, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]],
        ],
        dtype=jnp.int32,
    )

    blocks_sample_list = []
    blocks_sample_list.append(blocks_0)
    blocks_sample_list.append(blocks_1)
    blocks_sample_list.append(blocks_2)
    blocks_sample_list.append(blocks_3)
    blocks_sample_list.append(blocks_4)
    blocks_sample_list.append(blocks_5)
    blocks_sample_list.append(blocks_6)
    blocks_sample_list.append(blocks_7)
    return blocks_sample_list


def step_terminate_data_action() -> List[chex.Array]:
    action_0 = jnp.array([1, 5, 7])
    action_1 = jnp.array([0, 2, 7])
    action_2 = jnp.array([0, 2, 1])
    action_3 = jnp.array([1, 2, 6])
    action_4 = jnp.array([2, 7, 6])
    action_5 = jnp.array([0, 0, 2])
    action_6 = jnp.array([0, 6, 7])
    action_7 = jnp.array([2, 8, 0])

    action_sample_list = []
    action_sample_list.append(action_0)
    action_sample_list.append(action_1)
    action_sample_list.append(action_2)
    action_sample_list.append(action_3)
    action_sample_list.append(action_4)
    action_sample_list.append(action_5)
    action_sample_list.append(action_6)
    action_sample_list.append(action_7)
    return action_sample_list


def position_random_samples() -> List[Tuple[int, int]]:
    pos_0 = (0, 0)
    pos_1 = (0, 1)
    pos_2 = (0, 2)
    pos_3 = (0, 3)
    pos_4 = (0, 4)
    pos_5 = (1, 2)
    pos_6 = (6, 1)

    position_sample_list = []
    position_sample_list.append(pos_0)
    position_sample_list.append(pos_1)
    position_sample_list.append(pos_2)
    position_sample_list.append(pos_3)
    position_sample_list.append(pos_4)
    position_sample_list.append(pos_5)
    position_sample_list.append(pos_6)
    return position_sample_list


def apply_valid_action_solution() -> Tuple[List[GridPadded], List[int]]:
    grid_0 = jnp.array(
        [
            [0, 1, 0, 1, 1, 1, 0, 1, 0],
            [0, 1, 0, 1, 0, 1, 0, 0, 0],
            [0, 0, 1, 1, 0, 0, 1, 1, 1],
            [0, 0, 1, 1, 0, 0, 1, 0, 0],
            [0, 1, 1, 0, 1, 1, 1, 1, 1],
            [0, 1, 0, 1, 0, 0, 1, 0, 0],
            [0, 1, 1, 0, 1, 1, 1, 0, 0],
            [0, 1, 0, 1, 1, 1, 0, 1, 1],
            [0, 1, 1, 0, 0, 1, 0, 1, 1],
        ]
    )

    grid_1 = jnp.array(
        [
            [0, 1, 0, 1, 1, 0, 0, 1, 0],
            [0, 1, 0, 1, 1, 0, 1, 1, 1],
            [1, 1, 1, 0, 0, 1, 0, 0, 0],
            [1, 0, 0, 1, 1, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 1, 0, 0, 0],
            [1, 1, 0, 0, 1, 0, 0, 0, 0],
            [0, 1, 0, 1, 0, 0, 0, 0, 1],
            [0, 0, 1, 0, 0, 1, 0, 0, 1],
            [0, 1, 0, 0, 0, 1, 0, 1, 1],
        ]
    )

    grid_2 = jnp.array(
        [
            [1, 1, 0, 0, 1, 1, 1, 1, 1],
            [1, 1, 0, 1, 1, 1, 1, 0, 0],
            [1, 0, 1, 1, 1, 0, 1, 1, 1],
            [1, 0, 1, 1, 1, 0, 1, 1, 0],
            [0, 1, 1, 1, 0, 1, 0, 1, 0],
            [0, 1, 1, 1, 1, 0, 0, 0, 0],
            [0, 1, 1, 0, 1, 1, 1, 0, 1],
            [0, 1, 1, 1, 1, 0, 1, 1, 1],
            [1, 0, 0, 0, 1, 1, 1, 1, 0],
        ]
    )

    grid_3 = jnp.array(
        [
            [0, 0, 0, 0, 1, 0, 0, 1, 1],
            [0, 0, 1, 0, 1, 0, 0, 0, 1],
            [1, 1, 0, 0, 1, 1, 1, 1, 1],
            [0, 1, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 1, 1, 0, 1, 1, 1],
            [0, 1, 1, 0, 1, 0, 1, 0, 1],
            [1, 0, 0, 0, 1, 1, 0, 1, 0],
            [1, 0, 0, 1, 1, 0, 0, 1, 0],
            [1, 0, 1, 1, 0, 0, 1, 0, 1],
        ]
    )

    grid_4 = jnp.array(
        [
            [1, 1, 0, 1, 0, 0, 1, 1, 1],
            [0, 1, 0, 0, 0, 1, 1, 0, 1],
            [0, 1, 0, 1, 1, 0, 1, 1, 1],
            [1, 0, 0, 0, 0, 0, 0, 1, 0],
            [0, 1, 0, 0, 0, 1, 0, 0, 1],
            [0, 0, 1, 0, 0, 1, 0, 1, 0],
            [1, 0, 0, 1, 1, 1, 1, 0, 0],
            [1, 1, 0, 0, 0, 1, 1, 0, 0],
            [0, 1, 0, 1, 0, 1, 1, 0, 1],
        ]
    )

    grid_5 = jnp.array(
        [
            [0, 1, 1, 0, 1, 0, 0, 1, 1],
            [0, 1, 1, 1, 1, 0, 1, 0, 0],
            [0, 0, 0, 0, 1, 0, 1, 0, 0],
            [0, 1, 0, 0, 0, 1, 0, 0, 1],
            [0, 0, 0, 1, 1, 0, 1, 0, 1],
            [0, 0, 1, 0, 1, 0, 0, 0, 0],
            [1, 0, 0, 0, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
        ]
    )

    grid_6 = jnp.array(
        [
            [0, 1, 1, 0, 0, 0, 0, 1, 0],
            [1, 1, 1, 0, 1, 0, 0, 0, 1],
            [1, 0, 0, 1, 0, 0, 0, 0, 0],
            [0, 1, 1, 1, 1, 0, 1, 1, 1],
            [1, 0, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 0, 0, 0, 0, 1, 1, 0],
            [1, 0, 1, 0, 0, 0, 1, 1, 1],
            [1, 1, 0, 0, 0, 0, 1, 1, 1],
            [1, 1, 0, 0, 0, 0, 0, 1, 1],
        ]
    )

    grid_sample_list = []
    grid_sample_list.append(grid_0)
    grid_sample_list.append(grid_1)
    grid_sample_list.append(grid_2)
    grid_sample_list.append(grid_3)
    grid_sample_list.append(grid_4)
    grid_sample_list.append(grid_5)
    grid_sample_list.append(grid_6)

    combo_solution_list = [1, 1, 0, 0, 0, 2, 1]

    return grid_sample_list, combo_solution_list


def grid_mask_and_combo_solution() -> Tuple[List[Grid], List[int]]:
    grid_0 = jnp.array(
        [
            [1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1],
        ]
    )

    grid_1 = jnp.array(
        [
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
        ]
    )

    grid_2 = jnp.array(
        [
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 1, 1, 0, 0, 0],
            [1, 1, 1, 1, 1, 1, 1, 1, 1],
            [0, 0, 0, 1, 1, 1, 0, 0, 0],
        ]
    )

    grid_3 = jnp.array(
        [
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
        ]
    )

    grid_4 = jnp.array(
        [
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
        ]
    )

    grid_5 = jnp.array(
        [
            [1, 1, 0, 0, 0, 0, 0, 0, 0],
            [1, 1, 0, 0, 0, 0, 0, 0, 0],
            [1, 1, 0, 0, 0, 0, 0, 0, 0],
            [1, 1, 0, 0, 0, 0, 0, 0, 0],
            [1, 1, 0, 0, 0, 0, 0, 0, 0],
            [1, 1, 0, 0, 0, 0, 0, 0, 0],
            [1, 1, 0, 0, 0, 0, 0, 0, 0],
            [1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1],
        ]
    )

    grid_6 = jnp.array(
        [
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
        ]
    )

    grid_list = []
    grid_list.append(grid_0)
    grid_list.append(grid_1)
    grid_list.append(grid_2)
    grid_list.append(grid_3)
    grid_list.append(grid_4)
    grid_list.append(grid_5)
    grid_list.append(grid_6)

    combo_solution_list = [9, 0, 2, 0, 0, 4, 0]

    return grid_list, combo_solution_list


def block_action_mask_solution():
    block_action_mask_solution_0 = jnp.array(
        [
            [False, False, False, False, False, False, False, False, False],
            [False, False, False, False, False, False, False, False, False],
            [False, False, False, False, False, False, False, False, False],
            [False, False, False, False, False, False, False, False, False],
            [False, False, False, False, False, False, False, False, False],
            [False, False, False, False, False, False, False, False, False],
            [False, False, False, False, False, False, False, False, False],
            [False, False, False, False, False, False, False, False, False],
            [False, False, False, False, False, False, False, False, False],
        ]
    )
    block_action_mask_solution_1 = jnp.array(
        [
            [False, False, False, False, False, False, False, False, False],
            [False, False, False, False, False, False, False, False, False],
            [False, False, False, False, False, False, False, False, False],
            [False, False, False, False, False, False, False, False, False],
            [False, False, False, False, False, False, False, False, False],
            [False, False, False, False, False, False, False, False, False],
            [False, False, False, False, False, False, False, False, False],
            [False, False, False, False, False, False, False, False, False],
            [False, False, False, False, False, False, False, False, False],
        ]
    )
    block_action_mask_solution_2 = jnp.array(
        [
            [False, False, False, False, False, False, False, False, False],
            [False, False, False, False, False, False, False, False, False],
            [False, False, False, False, False, False, False, False, False],
            [False, False, False, False, False, False, False, False, False],
            [False, False, False, False, False, False, False, False, False],
            [False, False, False, False, False, False, False, False, False],
            [False, False, False, False, False, False, False, False, False],
            [False, False, False, False, False, False, False, False, False],
            [False, False, False, False, False, False, False, False, False],
        ]
    )
    block_action_mask_solution_3 = jnp.array(
        [
            [False, False, False, False, False, False, False, False, False],
            [False, False, False, False, False, False, False, False, False],
            [False, False, False, False, False, False, False, False, False],
            [False, False, False, False, False, False, False, False, False],
            [False, False, False, False, False, False, False, False, False],
            [False, False, False, False, False, False, False, False, False],
            [False, False, False, False, False, False, False, False, False],
            [False, False, False, False, False, False, False, False, False],
            [False, False, False, False, False, False, False, False, False],
        ]
    )
    block_action_mask_solution_4 = jnp.array(
        [
            [False, False, False, False, False, False, False, False, False],
            [False, False, False, False, False, False, False, False, False],
            [False, False, False, False, False, False, False, False, False],
            [False, False, False, False, False, False, False, False, False],
            [False, False, False, False, False, False, False, False, False],
            [False, False, False, False, False, False, False, False, False],
            [False, False, False, False, False, False, False, False, False],
            [False, False, False, False, False, False, False, False, False],
            [False, False, False, False, False, False, False, False, False],
        ]
    )
    block_action_mask_solution_5 = jnp.array(
        [
            [False, False, True, False, False, True, True, True, False],
            [False, False, True, False, False, False, False, False, False],
            [False, False, False, False, False, False, False, False, False],
            [False, False, False, False, False, False, False, False, False],
            [False, False, False, False, False, True, True, True, False],
            [False, False, False, False, False, False, False, False, False],
            [False, False, False, False, False, False, False, False, False],
            [False, False, False, False, False, False, False, False, False],
            [False, False, False, False, False, False, False, False, False],
        ]
    )
    block_action_mask_solution_6 = jnp.array(
        [
            [True, False, False, False, False, True, True, False, False],
            [False, False, False, False, False, False, False, False, False],
            [False, False, False, False, False, False, True, False, False],
            [False, False, False, False, False, False, False, False, False],
            [False, False, False, False, False, False, False, False, False],
            [False, False, False, False, False, False, False, False, False],
            [False, True, False, False, True, False, False, False, False],
            [False, False, False, False, False, False, False, False, False],
            [False, False, False, False, False, False, False, False, False],
        ]
    )

    grid_mask_and_combo_solution_list = []

    grid_mask_and_combo_solution_list.append(block_action_mask_solution_0)
    grid_mask_and_combo_solution_list.append(block_action_mask_solution_1)
    grid_mask_and_combo_solution_list.append(block_action_mask_solution_2)
    grid_mask_and_combo_solution_list.append(block_action_mask_solution_3)
    grid_mask_and_combo_solution_list.append(block_action_mask_solution_4)
    grid_mask_and_combo_solution_list.append(block_action_mask_solution_5)
    grid_mask_and_combo_solution_list.append(block_action_mask_solution_6)

    return grid_mask_and_combo_solution_list


def grid_mask_and_combo_dataset() -> Iterator[Tuple[Grid, Grid, int]]:
    grid_solution_list, combo_solution_list = grid_mask_and_combo_solution()
    grid_mask_and_combo_dataset = zip(grid_random_samples(), grid_solution_list, combo_solution_list)
    return grid_mask_and_combo_dataset


def block_action_mask_dataset() -> Iterator[Tuple[chex.Array, chex.Array]]:
    block_action_mask_dataset = zip(pad_grid(grid_random_samples()), block_action_mask_solution())
    return block_action_mask_dataset


def apply_valid_action_dataset():
    action_list = [
        jnp.array([0, 6, 0]),
        jnp.array([0, 3, 7]),
        jnp.array([0, 5, 1]),
        jnp.array([0, 0, 7]),
        jnp.array([0, 0, 7]),
        jnp.array([0, 6, 3]),
        jnp.array([0, 4, 5]),
    ]
    grid_solution_list, combo_solution_list = apply_valid_action_solution()
    apply_valid_mask_dataset = zip(
        pad_grid(grid_random_samples_for_apply_valid_action()),
        action_list,
        pad_grid(grid_solution_list),
        combo_solution_list,
    )
    return apply_valid_mask_dataset


def step_terminate_dataset() -> Iterator[Tuple[GridPadded, Blocks, chex.Array]]:
    step_terminate_dataset = zip(
        pad_grid(step_terminate_data_grid()),
        step_terminate_data_blocks(),
        step_terminate_data_action(),
    )
    return step_terminate_dataset


def test_dataset():
    grid_list, combo_solution_list = grid_mask_and_combo_solution()
    assert len(grid_list) == len(combo_solution_list)
    assert len(grid_random_samples()) == len(combo_solution_list)
    assert len(grid_random_samples()) == len(block_action_mask_solution())
    assert len(step_terminate_data_grid()) == len(step_terminate_data_action())
    assert len(step_terminate_data_grid()) == len(step_terminate_data_blocks())
