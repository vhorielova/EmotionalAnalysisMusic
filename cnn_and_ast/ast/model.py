import torch
from transformers import ASTForAudioClassification
from config import EMOTION_NAMES

class ASTWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        return self.model(input_values=x).logits

def build_model():
    name = "MIT/ast-finetuned-audioset-10-10-0.4593"

    model = ASTForAudioClassification.from_pretrained(
        name,
        num_labels=4,
        ignore_mismatched_sizes=True
    )

    model.config.id2label = {
        i: label for i, label in enumerate(EMOTION_NAMES)
    }

    model.config.label2id = {
        label: i for i, label in enumerate(EMOTION_NAMES)
    }

    return model

def freeze_ast_layers(model):
    for name, param in model.named_parameters():

        if "audio_spectrogram_transformer.embeddings" in name:
            param.requires_grad = False

        for i in range(8):
            if f"audio_spectrogram_transformer.layers.{i}." in name:
                param.requires_grad = False

    trainable = sum(
        p.numel() for p in model.parameters()
        if p.requires_grad
    )

    frozen = sum(
        p.numel() for p in model.parameters()
        if not p.requires_grad
    )

    print(f"Trainable parameters: {trainable:,}")
    print(f"Frozen parameters: {frozen:,}")