"""
Visualisation of the model.
"""

import torch
import torchviz


def render_architecture(model, name: str = "model", directory: str = ""):
    """
    Render the architecture of the model.
    """
    out = model(torch.zeros(1, 112, 8, 8))
    if len(out) == 2:
        policy, outcome_probs = out
        value = torch.zeros(outcome_probs.shape[0], 1)
    else:
        policy, outcome_probs, value = out
    torchviz.make_dot(
        policy, params=dict(list(model.named_parameters()))
    ).render(f"{directory}/{name}_policy", format="svg")
    torchviz.make_dot(
        outcome_probs, params=dict(list(model.named_parameters()))
    ).render(f"{directory}/{name}_outcome_probs", format="svg")
    torchviz.make_dot(
        value, params=dict(list(model.named_parameters()))
    ).render(f"{directory}/{name}_value", format="svg")
