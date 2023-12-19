"""
Gradio interface for playing chess.
"""

import chess
import chess.svg
import gradio as gr
import torch

from experiments import global_variables
from lczero_easy.move import utils as move_utils


def ai_play(model_name, depth):
    if global_variables.playing_board.turn != global_variables.ai_color:
        gr.Warning("AI is not playing this turn.")
        gr.Warning("Only evaluating the position.")
    model = global_variables.models[model_name]
    policy, outcome, value = model.prediction(global_variables.playing_board)
    policy = torch.softmax(policy, dim=-1)
    value = value.item()
    us_win = outcome[0].item()
    draw = outcome[1].item()
    them_win = outcome[2].item()

    us = global_variables.playing_board.turn
    us_them = (us, not us)
    legal_moves = [
        move_utils.encode_move(move, us_them)
        for move in global_variables.playing_board.legal_moves
    ]
    filtered_policy = torch.zeros(1858)
    filtered_policy[legal_moves] = policy[legal_moves]
    topk_moves = torch.topk(filtered_policy, 1)
    move = move_utils.decode_move(
        topk_moves.indices[0], us_them, global_variables.playing_board
    )

    if global_variables.playing_board.turn == global_variables.ai_color:
        global_variables.playing_board.push(move)
    white_win = us_win if us == chess.WHITE else them_win
    black_win = us_win if us == chess.BLACK else them_win
    return (
        f"Value: {value:.2f}  -  "
        f"White win: {white_win:.2f}  -  "
        f"Draw: {draw:.2f}  -  Black win: {black_win:.2f}"
    )


def render_board(arrows=None):
    svg_board = chess.svg.board(
        global_variables.playing_board,
        size=350,
        arrows=arrows if arrows is not None else [],
    )
    with open("experiments/figures/engine.svg", "w") as f:
        f.write(svg_board)
    return "experiments/figures/engine.svg"


def make_game_info():
    return (
        f"AI color: {'WHITE' if global_variables.ai_color else 'BLACK'}  -  "
        f"AI victories: {global_variables.ai_victories}  -  "
        f"Human victories: {global_variables.human_victories}"
    )


def load_game():
    return render_board(), make_game_info()


def ai_button_click(
    model_name,
    depth,
):
    info = ai_play(model_name, depth)
    return render_board(), info


def reset_game_click():
    global_variables.playing_board = chess.Board()
    global_variables.ai_color = torch.randint(0, 2, (1,)).item() == 0
    return render_board(), make_game_info(), ""


def prev_move_click():
    legal_moves = list(global_variables.playing_board.legal_moves)
    n_move = len(legal_moves)
    global_variables.legal_move_index = (
        global_variables.legal_move_index - 1
    ) % n_move
    move = legal_moves[global_variables.legal_move_index]
    arrows = [(move.from_square, move.to_square)]
    return render_board(arrows)


def next_move_click():
    legal_moves = list(global_variables.playing_board.legal_moves)
    n_move = len(legal_moves)
    global_variables.legal_move_index = (
        global_variables.legal_move_index + 1
    ) % n_move
    move = legal_moves[global_variables.legal_move_index]
    arrows = [(move.from_square, move.to_square)]
    return render_board(arrows)


def play_move_click():
    move = list(global_variables.playing_board.legal_moves)[
        global_variables.legal_move_index
    ]
    global_variables.playing_board.push(move)
    global_variables.legal_move_index = 0
    if global_variables.playing_board.is_game_over():
        if global_variables.playing_board.result() == "1-0":
            global_variables.ai_victories += 1
        else:
            global_variables.human_victories += 1
        return (
            render_board(),
            f"Game over: {global_variables.playing_board.result()}",
        )
    return render_board(), ""


with gr.Blocks() as interface:
    with gr.Row():
        with gr.Column():
            with gr.Group():
                with gr.Row():
                    model_name = gr.Dropdown(
                        label="Model name",
                        choices=global_variables.MODEL_NAMES,
                        value=global_variables.MODEL_NAMES[0],
                    )
                    depth = gr.Radio(label="Depth", choices=[0], value=0)
            with gr.Group():
                ai_button = gr.Button("AI play")
                play_info = gr.Textbox(
                    label="Play info", lines=1, max_lines=1, value=""
                )

            with gr.Group():
                with gr.Row():
                    prev_move = gr.Button("Previous legal move")
                    play_move = gr.Button("Play move")
                    next_move = gr.Button("Next legal move")
                play_info = gr.Textbox(
                    label="Play info", lines=1, max_lines=1, value=""
                )

            game_info = gr.Textbox(
                label="Game info", lines=1, max_lines=1, value=""
            )
            reset_game = gr.Button("Reset game")

        with gr.Column():
            image = gr.Image(label="Board")

    prev_move.click(prev_move_click, inputs=[], outputs=[image])
    next_move.click(next_move_click, inputs=[], outputs=[image])

    play_move.click(play_move_click, inputs=[], outputs=[image, play_info])
    ai_button.click(
        ai_button_click, inputs=[model_name, depth], outputs=[image, play_info]
    )

    reset_game.click(
        reset_game_click, inputs=[], outputs=[image, game_info, play_info]
    )
    interface.load(load_game, inputs=[], outputs=[image, game_info])
