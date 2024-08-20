import chex
import jax
import jax.numpy as jnp
import pytest


@pytest.fixture(scope="module")
def block() -> chex.Array:
    return jnp.array(
        [
            [1, 1, 0, 0, 0],
            [0, 1, 0, 0, 0],
            [1, 1, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
        ]
    )
