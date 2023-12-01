"""
Board utilities.
"""

import re
from copy import deepcopy

import chess
import torch


def board_to_tensor13x8x8(board: chess.Board):
    """
    Converts a chess.Board object to a 64 tensor.
    """
    us = board.turn
    them = not us
    plane_orders = {chess.WHITE: "PNBRQK", chess.BLACK: "pnbrqk"}
    plane_order = plane_orders[us] + plane_orders[them]

    fen_rep = board.fen().split(" ")[0]
    fen_rep = re.sub(r"(\d)", lambda m: "0" * int(m.group(1)), fen_rep)
    rows = fen_rep.split("/")
    rev_rows = rows[::-1]
    ordered_fen = "".join(rev_rows)
    tensor13x8x8 = torch.zeros((13, 8, 8), dtype=torch.float)
    for i, piece in enumerate(plane_order):
        tensor13x8x8[i] = torch.tensor(
            [[1 if c == piece else 0 for c in row] for row in ordered_fen]
        ).view(8, 8)
    if board.is_repetition(2):
        tensor13x8x8[12] = torch.ones((8, 8), dtype=torch.float)
    if us == chess.WHITE:
        return tensor13x8x8
    else:
        return tensor13x8x8.flip(2)


def board_to_tensor112x8x8(last_board=chess.Board):
    """
    Create the lc0 112x8x8 tensor from the history of a game.
    """
    board = deepcopy(last_board)
    tensor112x8x8 = torch.zeros((112, 8, 8), dtype=torch.int8)
    us = last_board.turn
    them = not us
    for i in range(8):
        tensor13x8x8 = board_to_tensor13x8x8(board)
        tensor112x8x8[
            i * 13 : (i + 1) * 13
        ] = tensor13x8x8  # TODO: check the history order
        try:
            board.pop()
        except IndexError:
            break
    if last_board.has_queenside_castling_rights(us):
        tensor112x8x8[104] = torch.ones((8, 8), dtype=torch.int8)
    if last_board.has_kingside_castling_rights(us):
        tensor112x8x8[105] = torch.ones((8, 8), dtype=torch.int8)
    if last_board.has_queenside_castling_rights(them):
        tensor112x8x8[106] = torch.ones((8, 8), dtype=torch.int8)
    if last_board.has_kingside_castling_rights(them):
        tensor112x8x8[107] = torch.ones((8, 8), dtype=torch.int8)
    if us == chess.BLACK:
        tensor112x8x8[108] = torch.ones((8, 8), dtype=torch.int8)
    if last_board.is_fifty_moves():
        tensor112x8x8[109] = torch.ones((8, 8), dtype=torch.int8)
    tensor112x8x8[111] = torch.ones((8, 8), dtype=torch.int8)
    return tensor112x8x8


if __name__ == "__main__":
    board = chess.Board()
    tensor = board_to_tensor13x8x8(board)
    print(tensor.sum(axis=0))
    print(tensor.sum(axis=(1, 2)))
    full_tensor = board_to_tensor112x8x8(board)
    print(full_tensor)
