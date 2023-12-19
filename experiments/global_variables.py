"""
Global variables used by Gradio.
"""

import chess
import torch

from lczero_easy.model import LczerroModelWrapper

MODEL_DIRECTORY = "experiments/onnx_models"
MODEL_NAMES = [
    "768x15x24h-t82-2-swa-5230000.onnx",
    "128x10-2020_0324_1742_00_266.onnx",
    "t1-smolgen-512x15x8h-distilled-swa-3395000.onnx",
]
models = {
    name: LczerroModelWrapper(f"{MODEL_DIRECTORY}/{name}")
    for name in MODEL_NAMES
}
FIGURE_DIRECTORY = "experiments/figures"

playing_board = chess.Board()
ai_color = torch.randint(0, 2, (1,)).item() == 0
ai_victories = 0
human_victories = 0
legal_move_index = 0
