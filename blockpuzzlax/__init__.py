from jumanji.registration import register
from blockpuzzlax.version import __version__


register(
    id="Blockpuzzle-v1",
    entry_point="blockpuzzlax.environments:BlockPuzzle",
)
