from typing import TYPE_CHECKING, NamedTuple
from typing_extensions import TypeAlias

import chex

if TYPE_CHECKING:
    from dataclasses import dataclass
else:
    from chex import dataclass

Grid: TypeAlias = chex.Array
GridPadded: TypeAlias = chex.Array
Blocks: TypeAlias = chex.Array
Block: TypeAlias = chex.Array


@dataclass
class State:
    grid_padded: chex.Array
    action_mask: chex.Array
    key: chex.PRNGKey
    blocks: chex.Array
    streak: chex.Numeric
    score: chex.Numeric


class Observation(NamedTuple):
    grid: chex.Array
    blocks: chex.Array
    action_mask: chex.Array
