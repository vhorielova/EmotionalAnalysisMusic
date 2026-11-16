from collections import defaultdict
import torch

def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0

    song_logits = defaultdict(list)
    song_targets = {}

    with torch.no_grad():
        offset = 0

        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss = criterion(out, y)
            total_loss += loss.item()

            batch_song_ids = [song_id for song_id, _ in loader.dataset.segments[offset: offset + len(y)]]
            offset += len(y)

            for i, sid in enumerate(batch_song_ids):
                song_logits[sid].append(out[i].cpu())
                song_targets[sid] = y[i].cpu().item()

    all_preds = []
    all_labels = []

    for sid, logits_list in song_logits.items():
        avg_logits = torch.stack(logits_list).mean(dim=0)
        pred_class = avg_logits.argmax(dim=0).item()
        all_preds.append(pred_class)
        all_labels.append(song_targets[sid])

    return total_loss / len(loader), all_preds, all_labels