# Blockpuzzlax

![block_puzzle_random_policy](https://github.com/user-attachments/assets/38f68655-37a9-4045-8864-5c75c3b8e5a6)

This is a reinforcement learning environment for block puzzle-type games implemented in JAX. It is built based on the [**Jumanji**](https://github.com/instadeepai/jumanji).

# Installation

```bash
git clone https://github.com/helpingstar/blockpuzzlax.git
cd blockpuzzlax
pip install -e .
```

# Description

blockpuzzlax is a simple yet strategic puzzle game. The goal of the game is to place various shaped blocks on a 9x9 grid to complete rows, columns, or 3x3 square sections, clearing them and earning the highest possible score.

1. **Placing Blocks**: Players are given three blocks at a time, and they must place these blocks on the grid. Each block comes in different shapes and sizes, and players can only place one block at a time. Once all three blocks are placed, new blocks are provided.

2. **Clearing Lines**: By filling a row, column, or 3x3 section with blocks, the line will clear, and the player earns points. Clearing multiple lines at once yields higher scores.

3. **Bonus Points**: The game rewards bonus points through two mechanisms. When multiple lines are cleared simultaneously, players receive a **Combo** bonus. Additionally, by clearing lines in consecutive turns, players earn a **Streak** bonus, further increasing their score. Strategically placing blocks to maximize these bonuses is key to achieving a high score.

4. **Managing Space**: The game ends when there is no more space left on the grid to place blocks. Therefore, managing space efficiently and leaving room for future blocks is essential to prolong gameplay.

Since there is no time limit in blockpuzzlax, players can carefully plan their moves to survive longer and achieve a higher score.

# Observation

emtpy/filled cells are represented by 0/1.

* board: jax array (int32) of shape (9, 9)
* blocks: jax array (int32) of shape (3, 5, 5)
  * The used blocks are represented by zeros in a 5x5 grid.
* action mask: jax array (bool) of shape (3, 9, 9): indicates which actions are valid.

# Action

The top-left corner of the block is placed at the (row index, column index) on the board.

The action space in blockpuzzlax is represented as a `MultiDiscreteArray` of three integer values.

* 0: block index
* 1: row index
* 2: column index

# Reward

* $n$: The number of cells in a block.
* $c$, combo: The number of lines cleared simultaneously.
* $s$, streak: The number of consecutive lines cleared.

* **WoodokuReward**
  * If there are cleared cells: $28c + 10s + n - 20$
  * If there are no cleared cells: $n$

# Registered Versions

* `Blockpuzzle-v1` : a block puzzle game environment on a 9x9 grid.

# See Also

* [instadeepai/jumanji](https://github.com/instadeepai/jumanji) : JAX-based suite of 22 scalable reinforcement learning environments.
* [corl-team/xland-minigrid](https://github.com/corl-team/xland-minigrid) : XLand-MiniGrid is a scalable JAX toolset for meta-reinforcement learning research.
