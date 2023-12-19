"""
Visualisation of the policy prediction.
"""

import chess
import torch

from lczero_easy.move import INVERTED_FROM_INDEX, INVERTED_TO_INDEX


def aggregate_policy(policy, aggregate_topk=-1):
    """
    Aggregate the policy for a given board.
    """
    pickup_agg = torch.zeros(64)
    dropoff_agg = torch.zeros(64)
    if aggregate_topk > 0:
        filtered_policy = torch.zeros(1858)
        topk = torch.topk(policy, aggregate_topk)
        filtered_policy[topk.indices] = topk.values
    else:
        filtered_policy = policy
    for square_index in range(64):
        square = chess.SQUARE_NAMES[square_index]
        pickup_agg[square_index] = filtered_policy[
            INVERTED_FROM_INDEX[square]
        ].sum()
        dropoff_agg[square_index] = filtered_policy[
            INVERTED_TO_INDEX[square]
        ].sum()
    return pickup_agg, dropoff_agg
