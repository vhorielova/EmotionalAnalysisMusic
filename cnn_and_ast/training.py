import torch
from collections import Counter
from sklearn.metrics import f1_score, classification_report, confusion_matrix
from cnn_and_ast.evaluation import evaluate

def get_device():
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")

def build_criterion(train_dataset, labels, device):
    counts = Counter(labels[song_id] for song_id, _ in train_dataset.segments)
    n_total = len(train_dataset)

    weights = [n_total / (4 * counts[c]) for c in range(4)]
    print(weights)

    class_weights = torch.tensor(weights, dtype=torch.float32).to(device)
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
    return criterion

def overfit_one_batch(model, train_loader, criterion, device):
    x, y = next(iter(train_loader))
    x, y = x.to(device), y.to(device)

    model.eval()
    with torch.no_grad():
        loss = criterion(model(x), y)
    print(loss.item())

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    model.train()
    for step in range(100):
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()

        if step % 10 == 0:
            print(step, loss.item())

def train_model(model, train_loader, val_loader, criterion, optimizer, device, save_path, MAX_EPOCHS):
    best_f1 = 0
    early_stopping_counter = 0

    for i in range(MAX_EPOCHS):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, predictions, labels = evaluate(model, val_loader, criterion, device)
        f1_s = f1_score(labels, predictions, average="macro")
        print(f"epoch {i+1}: train loss {train_loss:.3f} | val loss {val_loss:.3f} | val macro-F1 {f1_s:.3f}")
        print(predictions[:10])
        print(labels[:10])

        if best_f1 < f1_s:
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_f1 = f1_s
            early_stopping_counter = 0
        else:
            early_stopping_counter += 1

        if early_stopping_counter >= 7: break
    model.load_state_dict(best_state)
    val_loss, predictions, labels = evaluate(model, val_loader, criterion, device)
    print(classification_report(labels, predictions))
    print(confusion_matrix(labels, predictions))
    torch.save(best_state, save_path)

def train_one_epoch(model, loader, criterion, optimizer, device):
    total_loss = 0
    model.train()

    for batch in loader:
        x, y = batch[0], batch[1]
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()


    return total_loss/len(loader)