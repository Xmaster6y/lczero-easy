"""
Compute attention heatmap for a given model and input.
"""

import chess

from lczero_easy.board import utils as board_utils
from lczero_easy.model import LczerroModelWrapper


def compute_attention_heatmap(
    board: chess.Board,
    wrapper: LczerroModelWrapper,
    attention_layer: int,
    attention_head: int,
):
    """
    Compute attention heatmap for a given model and input.
    """
    board_tensor = board_utils.board_to_tensor112x8x8(board)
    return board_tensor.mean(dim=0).view(64)
