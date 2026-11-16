from cnn_and_ast.cnn.model import EmotionCNN
from cnn_and_ast.cnn.data_prep import prepare_data
from cnn_and_ast.cnn.diagnostics import print_batch_info, print_model_info
from cnn_and_ast.training import get_device, build_criterion, overfit_one_batch, train_model
from cnn_and_ast.test import test_model
import torch
from config import RANDOM_STATE, MAX_EPOCHS_CNN
from cnn_and_ast.cnn.optimizer import build_optimizer

def run_cnn(df, train_ids, test_ids):
    # data
    train_loader, val_loader, test_loader, train_dataset, labels = prepare_data(df, train_ids, test_ids)
    print_batch_info(train_loader, val_loader, test_loader)

    # model
    device = get_device()
    model = EmotionCNN().to(device)
    print_model_info(model, train_loader, device)

    criterion = build_criterion(train_dataset, labels, device)
    overfit_one_batch(model, train_loader, criterion, device)

    torch.manual_seed(RANDOM_STATE)
    model = EmotionCNN().to(device)
    optimizer = build_optimizer(model)

    # train_model(model, train_loader, val_loader, criterion, optimizer, device, "best_cnn.pt", MAX_EPOCHS_CNN)


    model = EmotionCNN().to(device)
    test_model(model, test_loader, criterion, device, "best_cnn.pt", "cnn")