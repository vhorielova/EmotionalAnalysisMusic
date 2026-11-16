import torch
from cnn_and_ast.ast.data_prep import prepare_data
from cnn_and_ast.ast.model import build_model, ASTWrapper, freeze_ast_layers
from transformers import ASTFeatureExtractor
from cnn_and_ast.training import train_model, get_device, build_criterion
from config import MAX_EPOCHS_AST, LEARNING_RATE_AST
from cnn_and_ast.ast.optimizer import build_optimizer
from cnn_and_ast.test import test_model


def run_ast(df, train_ids, test_ids):
    name = "MIT/ast-finetuned-audioset-10-10-0.4593"

    extractor = ASTFeatureExtractor.from_pretrained(name)

    train_loader, val_loader, test_loader, train_dataset, labels = prepare_data(
        df, train_ids, test_ids, extractor
    )

    device = get_device()
    print("Device:", device)

    model = build_model().to(device)
    model = ASTWrapper(model)

    # for n, _ in model.named_parameters(): print(n)

    freeze_ast_layers(model)
    for name, param in model.named_parameters():
        if "encoder.layer" in name:
            print(name, param.requires_grad)

    criterion = build_criterion(train_dataset, labels, device)

    optimizer = build_optimizer(model)

    # train_model(model, train_loader, val_loader, criterion, optimizer, device, "best_ast.pt", MAX_EPOCHS_AST)

    test_model(model, test_loader, criterion, device, "best_ast.pt", "ast")
