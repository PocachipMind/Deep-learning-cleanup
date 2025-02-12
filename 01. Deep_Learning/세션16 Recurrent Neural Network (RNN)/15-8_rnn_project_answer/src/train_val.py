from tqdm import tqdm
import torch


def train_loop(model, train_dataloader, loss_fn, optimizer):
    device = next(model.parameters()).device
    train_loss_history = []
    tbar = tqdm(train_dataloader)
    model.train()
    max_norm = 5
    correct = 0

    size = len(train_dataloader.dataset)
    for batch in tbar:
        optimizer.zero_grad()

        input_ids = batch["input_ids"].to(device)
        label = batch["label"].to(device)
        pred = model(input_ids)
        loss = loss_fn(pred, label)
        loss.backward()

        # Gradient Clipping (안정적인 LSTM 학습을 위해)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

        optimizer.step()
        tbar.set_description(f"loss - {loss.item():.3f}")
        correct += (pred.argmax(1) == label).type(torch.float).sum().item()

        train_loss_history.append(loss.item())

    correct /= size

    return train_loss_history, correct


def val_loop(model, test_dataloader, loss_fn):
    device = next(model.parameters()).device
    tbar = tqdm(test_dataloader)
    model.eval()

    correct = 0
    test_loss = 0
    size = len(test_dataloader.dataset)
    num_batches = len(test_dataloader)

    for batch in tbar:
        with torch.no_grad():
            input_ids = batch["input_ids"].to(device)
            label = batch["label"].to(device)
            pred = model(input_ids)
            loss = loss_fn(pred, label)
            correct += (pred.argmax(1) == label).type(torch.float).sum().item()
            tbar.set_description(f"loss - {loss.item():.3f}")
            test_loss += loss.item()

    test_loss /= num_batches
    correct /= size

    return test_loss, correct
