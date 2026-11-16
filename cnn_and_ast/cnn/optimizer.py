import torch
from config import LEARNING_RATE_CNN

def build_optimizer(model):
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE_CNN, weight_decay=1e-4)

    return optimizer