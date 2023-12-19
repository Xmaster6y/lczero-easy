"""
Gradio interface for plotting policy.
"""

import gradio as gr

from experiments import global_variables
from lczero_easy.visualisation.model import render_architecture

model_rendered = []


def render_model(
    model_name,
    tensor_type,
):
    global model_rendered
    if model_name not in model_rendered:
        model = global_variables.models[model_name]
        model.ensure_loaded()
        render_architecture(
            model.model,
            name=model_name,
            directory=global_variables.FIGURE_DIRECTORY,
        )
        model_rendered.append(model_name)
    return (
        f"{global_variables.FIGURE_DIRECTORY}/{model_name}_{tensor_type}.svg"
    )


with gr.Blocks() as interface:
    with gr.Row():
        model_name = gr.Dropdown(
            label="Model name",
            choices=global_variables.MODEL_NAMES,
            value=global_variables.MODEL_NAMES[0],
        )
        tensor_type = gr.Dropdown(
            label="Tensor type",
            choices=["policy", "outcome_probs", "value"],
            value="policy",
        )

    button = gr.Button(label="Render")

    image = gr.Image(label="Board")

    button.click(
        render_model, inputs=[model_name, tensor_type], outputs=[image]
    )
    model_name.change(
        render_model, inputs=[model_name, tensor_type], outputs=[image]
    )
    tensor_type.change(
        render_model, inputs=[model_name, tensor_type], outputs=[image]
    )
