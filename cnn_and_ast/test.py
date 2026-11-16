import torch
from sklearn.metrics import classification_report, confusion_matrix
from cnn_and_ast.evaluation import evaluate
from config import EMOTION_NAMES
from reporting.predict import predict_songs
from reporting.reporting import save_predictions

def test_model(model, loader, criterion, device, file_path, model_name):
    state = torch.load(file_path, map_location=device)
    model.load_state_dict(state)

    # test_loss, predictions, labels = evaluate(model, loader, criterion, device)
    # print(classification_report(labels, predictions, target_names=EMOTION_NAMES))
    # print(confusion_matrix(labels, predictions))

    song_preds = predict_songs(model, loader, device)
    save_predictions(model_name, song_preds)