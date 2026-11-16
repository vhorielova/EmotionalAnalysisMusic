import torch
from config import LEARNING_RATE_AST


def build_optimizer(model):
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LEARNING_RATE_AST,
        weight_decay=0.01
    )

    return optimizer