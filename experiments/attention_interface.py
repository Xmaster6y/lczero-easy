"""
Gradio interface for plotting policy.
"""

import chess
import gradio as gr

from experiments import global_variables
from lczero_easy.visualisation.board import render_heatmap
from lczero_easy.xai import attention


def make_plot(
    board_fen,
    action_seq,
    model_name,
    attention_layer,
    attention_head,
):
    try:
        board = chess.Board(board_fen)
    except ValueError:
        board = chess.Board()
        gr.Warning("Invalid FEN, using starting position.")
    if action_seq:
        try:
            for action in action_seq.split():
                board.push_uci(action)
        except ValueError:
            gr.Warning("Invalid action sequence, using starting position.")
            board = chess.Board()
    wrapper = global_variables.wrappers[model_name]
    heatmap = attention.compute_attention_heatmap(
        board,
        wrapper,
        attention_layer,
        attention_head,
    )
    svg_board, fig = render_heatmap(board, heatmap, vmin=0.0, vmax=1.0)
    with open(f"{global_variables.FIGURE_DIRECTORY}/attention.svg", "w") as f:
        f.write(svg_board)
    return f"{global_variables.FIGURE_DIRECTORY}/attention.svg", fig


with gr.Blocks() as interface:
    with gr.Row():
        with gr.Column():
            board_fen = gr.Textbox(
                label="Board starting FEN",
                lines=1,
                max_lines=1,
                value=chess.STARTING_FEN,
            )
            action_seq = gr.Textbox(
                label="Action sequence",
                lines=1,
                max_lines=1,
                value=(
                    "e2e3 b8c6 d2d4 e7e5 g1f3 d8e7 "
                    "d4d5 e5e4 f3d4 c6e5 f2f4 e5g6"
                ),
            )
            with gr.Group():
                with gr.Row():
                    model_name = gr.Dropdown(
                        label="Model name",
                        choices=global_variables.MODEL_NAMES,
                        value=global_variables.MODEL_NAMES[0],
                    )
                with gr.Row():
                    attention_layer = gr.Slider(
                        label="Attention layer",
                        minimum=0,
                        maximum=11,
                        step=1,
                        value=0,
                    )
                    attention_head = gr.Slider(
                        label="Attention head",
                        minimum=0,
                        maximum=7,
                        step=1,
                        value=0,
                    )
            button = gr.Button("Plot attention")
            colorbar = gr.Plot(label="Colorbar")
        with gr.Column():
            image = gr.Image(label="Board")

    inputs = [
        board_fen,
        action_seq,
        model_name,
        attention_layer,
        attention_head,
    ]
    outputs = [image, colorbar]
    board_fen.submit(make_plot, inputs=inputs, outputs=outputs)
    action_seq.submit(make_plot, inputs=inputs, outputs=outputs)
    model_name.change(make_plot, inputs=inputs, outputs=outputs)
    attention_layer.change(make_plot, inputs=inputs, outputs=outputs)
    attention_head.change(make_plot, inputs=inputs, outputs=outputs)
    button.click(make_plot, inputs=inputs, outputs=outputs)
