from typing import Callable, Optional, Tuple, Union, Sequence

from blockpuzzlax.environments.blockpuzzle.constants import GRID_SIZE, BLOCK_WIDTH, MAX_NUM_BLOCKS
from blockpuzzlax.environments.blockpuzzle.types import State
from jumanji.viewer import Viewer
from numpy.typing import NDArray
import matplotlib.pyplot as plt
import matplotlib.animation
import numpy as np
import jumanji.environments

BLACK = 0.0
WHITE = 1.0
GRAY = 0.5


def put_sticker(sticker, dest, start_r, start_c):
    dest[start_r : start_r + sticker.shape[0], start_c : start_c + sticker.shape[1]] = sticker


def sub_sticker(sticker, dest, start_r, start_c):
    dest[start_r : start_r + sticker.shape[0], start_c : start_c + sticker.shape[1]] -= sticker


class BlockPuzzleViewer(Viewer):
    FIGURE_SIZE = (6, 6)

    def __init__(self, name: str = "BlockPuzzle", render_mode: str = "human") -> None:
        self._name = name

        self._display: Callable[[plt.Figure], Optional[NDArray]]
        if render_mode == "rgb_array":
            self._display = self._display_rgb_array
        elif render_mode == "human":
            self._display = self._display_human
        else:
            raise ValueError(f"Invalid render mode: {render_mode}")

        self._animation: Optional[matplotlib.animation.FuncAnimation] = None

        # size - render
        self.grid_cell_size = 16
        self.grid_line_size = 2
        self.grid_pad_size = 4
        self.block_cell_size = 8
        self.block_line_size = 1
        self.block_outer_pad_size = 8
        self.block_inner_pad_size = 9

        self.grid_room_size = self.grid_cell_size * GRID_SIZE + self.grid_line_size * (GRID_SIZE + 1)
        self.grid_and_pad_size = self.grid_room_size + 2 * self.grid_pad_size
        self.block_room_size = self.block_cell_size * BLOCK_WIDTH + self.block_line_size * (BLOCK_WIDTH + 1)
        self.block_and_pad_height = self.block_room_size + 2 * self.block_outer_pad_size

        self.block_col_step = self.block_room_size + self.block_inner_pad_size

        self.window_height = self.grid_and_pad_size + self.block_and_pad_height
        self.window_width = self.grid_and_pad_size

        # size - text
        self.text_size = 10

        # frame
        self.frame = self._get_window_frame()

        # unit
        grid_unit = np.full((self.grid_cell_size, self.grid_cell_size), GRAY)
        block_unit = np.full((self.block_cell_size, self.block_cell_size), GRAY)
        self._add_light_unit(grid_unit, self.grid_line_size)
        self._add_light_unit(block_unit, self.block_line_size)
        self.inverted_grid_unit = 1 - grid_unit
        self.inverted_block_unit = 1 - block_unit
        self.inverted_grid_unit = np.pad(
            self.inverted_grid_unit, pad_width=(self.grid_line_size, 0), mode="constant", constant_values=0.0
        )
        self.inverted_block_unit = np.pad(
            self.inverted_block_unit, pad_width=(self.block_line_size, 0), mode="constant", constant_values=0.0
        )

    def _add_light_unit(self, unit: NDArray, width: int):
        unit[-width:, :] = BLACK  # bottom
        unit[:, -width:] = BLACK  # right
        unit[:width, :] = WHITE  # top
        unit[:, :width] = WHITE  # left

    def _get_window_frame(self):
        one_grid_frame_pos = np.array([GRAY] * self.grid_line_size + [WHITE] * (self.grid_cell_size)).reshape(-1, 1)
        one_block_frame_pos = np.array([GRAY] * self.block_line_size + [WHITE] * (self.block_cell_size)).reshape(-1, 1)
        grid_frame_line = np.kron(np.ones((1, self.grid_room_size)), one_grid_frame_pos)
        block_frame_line = np.kron(np.ones((1, self.block_room_size)), one_block_frame_pos)
        grid_frame_row = np.tile(grid_frame_line, (GRID_SIZE + 1, 1))[: self.grid_room_size, :]
        block_frame_row = np.tile(block_frame_line, (BLOCK_WIDTH + 1, 1))[: self.block_room_size, :]
        grid_frame = np.minimum(grid_frame_row, grid_frame_row.T)
        block_frame = np.minimum(block_frame_row, block_frame_row.T)

        frame = np.ones((self.window_height, self.window_width))

        put_sticker(grid_frame, frame, self.grid_pad_size, self.grid_pad_size)

        put_sticker(block_frame, frame, self.grid_and_pad_size + self.block_outer_pad_size, self.block_outer_pad_size)
        put_sticker(
            block_frame,
            frame,
            self.grid_and_pad_size + self.block_outer_pad_size,
            self.block_outer_pad_size + self.block_col_step,
        )
        put_sticker(
            block_frame,
            frame,
            self.grid_and_pad_size + self.block_outer_pad_size,
            self.block_outer_pad_size + self.block_col_step * 2,
        )

        return frame

    def render(self, state: State) -> None:
        self._clear_display()
        fig, ax = self._get_fig_ax()
        ax.clear()
        fig.suptitle(f"| score : {state.score:2d} | streak : {state.streak:2d} |", size=self.text_size)
        self._add_grid_image(state, ax)
        return self._display(fig)

    def animate(
        self,
        states: Sequence[State],
        interval: int = 200,
        save_path: Optional[str] = None,
    ) -> matplotlib.animation.FuncAnimation:
        fig, ax = self._get_fig_ax()
        plt.tight_layout()

        # Define a function to animate a single game state.
        def make_frame(state_index: int) -> None:
            state = states[state_index]
            self._add_grid_image(state, ax)
            fig.suptitle(f"| score : {state.score:2d} | streak : {state.streak:2d} |", size=self.text_size)

        # Create the animation object.
        matplotlib.rc("animation", html="jshtml")
        self._animation = matplotlib.animation.FuncAnimation(
            fig,
            make_frame,
            frames=len(states),
            interval=interval,
        )

        # TODO : if save_path: -> pyright Error
        if self._animation is not None and save_path:
            self._animation.save(save_path)

        return self._animation

    def _display_human(self, fig: plt.Figure) -> None:
        if plt.isinteractive():
            fig.canvas.draw()
            if jumanji.environments.is_notebook():
                plt.show(self._name)
        else:
            fig.canvas.draw_idle()
            fig.canvas.flush_events()

    def _add_grid_image(self, state: State, ax: plt.Axes):
        frame = self.frame.copy()
        grid_sticker = np.kron(state.grid_padded[:GRID_SIZE, :GRID_SIZE], self.inverted_grid_unit)
        sub_sticker(grid_sticker, frame, self.grid_pad_size, self.grid_pad_size)
        for i in range(MAX_NUM_BLOCKS):
            block_sticker = np.kron(state.blocks[i], self.inverted_block_unit)
            sub_sticker(
                block_sticker,
                frame,
                self.grid_and_pad_size + self.block_outer_pad_size,
                self.block_outer_pad_size + self.block_col_step * i,
            )
        ax.set_axis_off()
        return ax.imshow(frame, cmap="gray")

    def _display_rgb_array(self, fig: plt.Figure) -> NDArray:
        fig.canvas.draw()
        return np.asarray(fig.canvas.renderer.buffer_rgba())

    def _get_fig_ax(self) -> Tuple[plt.Figure, plt.Axes]:
        recreate = not plt.fignum_exists(self._name)
        fig = plt.figure(self._name, BlockPuzzleViewer.FIGURE_SIZE)
        if recreate:
            if not plt.isinteractive():
                fig.show()
            ax = fig.add_subplot()
        else:
            ax = fig.get_axes()[0]
        return fig, ax

    def _clear_display(self) -> None:
        if jumanji.environments.is_notebook():
            import IPython.display

            IPython.display.clear_output(True)

    def close(self) -> None:
        plt.close(self._name)
