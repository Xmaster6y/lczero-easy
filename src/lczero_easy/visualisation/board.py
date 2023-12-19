"""
Visualisation of the board.
"""

import chess
import chess.svg
import matplotlib
import matplotlib.pyplot as plt
import torch

COLOR_MAP = matplotlib.colormaps["RdYlBu_r"].resampled(1000)
ALPHA = 1.0


def render_heatmap(
    board,
    heatmap,
    square=None,
    vmin=None,
    vmax=None,
    arrows=None,
):
    """
    Render a heatmap on the board.
    """
    if vmin is None:
        vmin = heatmap.min()
    if vmax is None:
        vmax = heatmap.max()
    norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax, clip=False)

    color_dict = {}
    for square_index in range(64):
        color = COLOR_MAP(norm(heatmap[square_index]))
        color = (*color[:3], ALPHA)
        color_dict[square_index] = matplotlib.colors.to_hex(
            color, keep_alpha=True
        )
    fig = plt.figure(figsize=(6, 0.6))
    ax = plt.gca()
    ax.axis("off")
    fig.colorbar(
        matplotlib.cm.ScalarMappable(norm=norm, cmap=COLOR_MAP),
        ax=ax,
        orientation="horizontal",
        fraction=1.0,
    )
    if square is not None:
        try:
            check = chess.parse_square(square)
        except ValueError:
            check = None
    else:
        check = None
    if arrows is None:
        arrows = []
    plt.close()
    return (
        chess.svg.board(
            board,
            check=check,
            fill=color_dict,
            size=350,
            arrows=arrows,
        ),
        fig,
    )


if __name__ == "__main__":
    board = chess.Board()
    svg_board, _ = render_heatmap(board, torch.arange(64))
    with open("gradio/test.svg", "w") as f:
        f.write(svg_board)