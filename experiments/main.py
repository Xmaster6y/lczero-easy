"""
Gradio demo for lczero-easy.
"""

import gradio as gr

from experiments import (
    chess_engine,
    encoding_interface,
    model_interface,
    policy_interface,
)

demo = gr.TabbedInterface(
    [
        policy_interface.interface,
        chess_engine.interface,
        model_interface.interface,
        encoding_interface.interface,
    ],
    ["Policy", "Chess Engine", "Model Architecture", "Board Encodings"],
    title="Lc0 Easy",
    analytics_enabled=False,
)

if __name__ == "__main__":
    demo.launch()