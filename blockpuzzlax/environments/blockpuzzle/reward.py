import abc
import chex
import jax


class RewardFn(abc.ABC):
    @abc.abstractmethod
    def __call__(
        self,
        combo: chex.Numeric,
        streak: chex.Numeric,
        n_block_cells: chex.Numeric,
    ) -> chex.Numeric:
        """
        Compute the reward based on the given parameters.
        """


class WoodokuReward(RewardFn):
    def __call__(
        self,
        combo: chex.Numeric,
        streak: chex.Numeric,
        n_block_cells: chex.Numeric,
    ) -> chex.Numeric:
        return jax.lax.select(
            combo > 0,
            28 * combo + 10 * streak + n_block_cells - 20,
            n_block_cells,
        )
