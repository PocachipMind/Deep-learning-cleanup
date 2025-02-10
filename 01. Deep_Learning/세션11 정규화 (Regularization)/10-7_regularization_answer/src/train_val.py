from tqdm import tqdm
import torch


def train_loop(model, train_dataloader, loss_fn, optimizer):
    device = next(model.parameters()).device

    train_loss_history = []
    tbar = tqdm(train_dataloader)
    model.train()

    for batch, label in tbar:
        batch = batch.to(device)
        label = label.to(device)

        pred = model(batch)
        loss = loss_fn(pred, label)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        tbar.set_description(f"Train Loss: {loss.item():.4f}")
        train_loss_history.append(loss.item())

    return train_loss_history


def val_loop(model, test_dataloader, loss_fn):
    device = next(model.parameters()).device

    tbar = tqdm(test_dataloader)
    num_batches = len(test_dataloader)
    size = len(test_dataloader.dataset)
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for batch, label in tbar:
            batch = batch.to(device)
            label = label.to(device)

            pred = model(batch)
            loss = loss_fn(pred, label)
            test_loss += loss.item()
            correct += (pred.argmax(1) == label).type(torch.float).sum().item()

            tbar.set_description(f"Test Loss: {loss.item():.4f}")
    test_loss /= num_batches
    correct /= size

    return test_loss, correct
