from typing import Tuple, Optional, Sequence
from functools import cached_property

import jax.numpy as jnp
import jax
import chex
import matplotlib

from blockpuzzlax.environments.blockpuzzle.viewer import BlockPuzzleViewer
from blockpuzzlax.environments.blockpuzzle.reward import WoodokuReward, RewardFn
from blockpuzzlax.environments.blockpuzzle.types import Observation, State
from blockpuzzlax.environments.blockpuzzle.constants import GRID_SIZE, MAX_NUM_BLOCKS, BLOCK_WIDTH, BLOCKS_LIST
from blockpuzzlax.environments.blockpuzzle import utils

from jumanji.env import Environment
from jumanji import specs
from jumanji.types import TimeStep, restart, termination, transition
from jumanji.viewer import Viewer


# TODO : DoneFn
class BlockPuzzle(Environment[State, specs.MultiDiscreteArray, Observation]):
    def __init__(self, reward_fn: Optional[RewardFn] = None, viewer: Optional[Viewer] = None) -> None:
        self.BLOCKS_LIST = jnp.array(BLOCKS_LIST)
        self.reward_fn = reward_fn or WoodokuReward()
        super().__init__()
        self._viewer = viewer or BlockPuzzleViewer()

    def __repr__(self) -> str:
        return f"BlockPuzzle_{GRID_SIZE}x{GRID_SIZE}_{len(self.BLOCKS_LIST)}"

    # TODO: type hint for jitted reset
    def reset(self, key: chex.PRNGKey) -> Tuple[State, TimeStep[Observation]]:
        grid = jnp.zeros((GRID_SIZE, GRID_SIZE), dtype=jnp.int32)
        grid_padded = jnp.pad(
            grid, pad_width=((0, BLOCK_WIDTH - 1), (0, BLOCK_WIDTH - 1)), mode="constant", constant_values=1
        )
        key, key_sample_blocks = jax.random.split(key)
        blocks = utils.sample_block_list(key_sample_blocks, self.BLOCKS_LIST)
        action_mask = jax.vmap(utils.block_action_mask, in_axes=(0, None))(blocks, grid_padded)
        state = State(
            grid_padded=grid_padded,
            action_mask=action_mask,
            key=key,
            blocks=blocks,
            streak=0,
            score=0,
        )

        obs = Observation(grid=grid_padded[:GRID_SIZE, :GRID_SIZE], blocks=blocks, action_mask=action_mask)
        timestep = restart(observation=obs)

        return state, timestep

    def step(self, state: State, action: chex.Array) -> Tuple[State, TimeStep[Observation]]:
        selected_block_idx = action[0]
        selected_block = state.blocks[selected_block_idx]

        valid_actions = state.action_mask
        invalid_action_taken = jnp.logical_not(valid_actions[action[0], action[1], action[2]])

        key, key_random_block = jax.random.split(state.key)

        updated_padded_grid, combo = jax.lax.cond(
            invalid_action_taken,
            lambda padded_grid, block, action: (padded_grid, 0),
            utils.apply_valid_action,
            state.grid_padded,
            selected_block,
            action,
        )

        # Update the state of the block
        updated_blocks = jax.lax.cond(
            invalid_action_taken,
            lambda blocks: blocks,
            lambda blocks: blocks.at[selected_block_idx].set(jnp.zeros((BLOCK_WIDTH, BLOCK_WIDTH), dtype=jnp.int32)),
            state.blocks,
        )

        # If all blocks are used up, new blocks are sampled.
        updated_blocks = jax.lax.cond(
            jnp.sum(updated_blocks) == 0,
            lambda key, _: utils.sample_block_list(key, self.BLOCKS_LIST),
            lambda _, block: block,
            key_random_block,
            updated_blocks,
        )

        updated_streak = jax.lax.cond(
            combo == 0,
            lambda _: 0,
            lambda streak: streak + 1,
            state.streak,
        )

        reward = self.reward_fn(combo, updated_streak, jnp.sum(selected_block))

        new_action_mask = jax.vmap(utils.block_action_mask, in_axes=(0, None))(updated_blocks, updated_padded_grid)

        next_state = State(
            grid_padded=updated_padded_grid,
            action_mask=new_action_mask,
            key=key,
            blocks=updated_blocks,
            streak=updated_streak,
            score=state.score + reward,
        )

        next_obs = Observation(
            grid=updated_padded_grid[:GRID_SIZE, :GRID_SIZE],
            blocks=updated_blocks,
            action_mask=new_action_mask,
        )

        done = jnp.logical_or(invalid_action_taken, ~jnp.any(new_action_mask))

        tiemstep = jax.lax.cond(
            done,
            termination,
            transition,
            reward,
            next_obs,
        )

        return next_state, tiemstep

    @cached_property
    def observation_spec(self) -> specs.Spec[Observation]:
        return specs.Spec(
            Observation,
            "ObservationSpec",
            grid=specs.BoundedArray(
                shape=(GRID_SIZE, GRID_SIZE),
                dtype=jnp.int32,
                minimum=0,
                maximum=1,
                name="grid",
            ),
            blocks=specs.BoundedArray(
                shape=(MAX_NUM_BLOCKS, BLOCK_WIDTH, BLOCK_WIDTH),
                dtype=jnp.int32,
                minimum=0,
                maximum=1,
                name="blocks",
            ),
            action_mask=specs.BoundedArray(
                shape=(MAX_NUM_BLOCKS, GRID_SIZE, GRID_SIZE),
                dtype=bool,
                maximum=True,
                minimum=False,
                name="action_mask",
            ),
        )

    @cached_property
    def action_spec(self) -> specs.MultiDiscreteArray:
        return specs.MultiDiscreteArray(
            num_values=jnp.array([MAX_NUM_BLOCKS, GRID_SIZE, GRID_SIZE]),
            name="action",
            dtype=jnp.int32,
        )

    def render(self, state: State) -> None:
        return self._viewer.render(state)

    def animate(
        self,
        states: Sequence[State],
        interval: int = 200,
        save_path: Optional[str] = None,
    ) -> matplotlib.animation.FuncAnimation:
        return self._viewer.animate(states=states, interval=interval, save_path=save_path)
