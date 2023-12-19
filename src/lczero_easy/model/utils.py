"""
Utils for the network module.
"""

from typing import List

import chess
import torch

from ..board import utils as board_utils

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def compute_move_prediction(model, board_list: List[chess.Board]):
    """
    Compute the move prediction for a list of boards.
    """
    tensor_list = [
        board_utils.board_to_tensor112x8x8(board).unsqueeze(0)
        for board in board_list
    ]
    batched_tensor = torch.cat(tensor_list, dim=0)
    batched_tensor.to(DEVICE)
    model.to(DEVICE)
    model.eval()
    with torch.no_grad():
        out = model(batched_tensor)
        if len(out) == 2:
            policy, outcome_probs = out
            value = torch.zeros(outcome_probs.shape[0], 1)
        else:
            policy, outcome_probs, value = out

    return policy, outcome_probs, value
