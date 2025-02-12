from tqdm import tqdm
import torch


def train_loop(
    model,
    train_dataloader,
    optimizer,
):
    device = next(model.parameters()).device
    train_loss_history = []

    tbar = tqdm(train_dataloader)
    model.train()

    correct = 0
    model.train()
    size = len(train_dataloader.dataset)

    for batch in tbar:
        optimizer.zero_grad()

        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        token_type_ids = batch["token_type_ids"].to(device)
        label = batch["label"].to(device)
        output = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            labels=label,
            return_dict=True,
        )

        loss = output.loss
        loss.backward()
        optimizer.step()

        tbar.set_description(f"loss - {loss.item():.3f}")

        pred = output.logits

        correct += (pred.argmax(1) == label).type(torch.float).sum().item()
        train_loss_history.append(loss.item())

    correct /= size

    return train_loss_history, correct


def val_loop(
    model,
    test_dataloader,
):
    device = next(model.parameters()).device
    correct = 0
    test_loss = 0
    size = len(test_dataloader.dataset)
    num_batches = len(test_dataloader)

    tbar = tqdm(test_dataloader)
    model.eval()

    for batch in tbar:
        with torch.no_grad():
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            token_type_ids = batch["token_type_ids"].to(device)
            label = batch["label"].to(device)
            output = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                labels=label,
                return_dict=True,
            )
            loss = output.loss
            pred = output.logits

            correct += (pred.argmax(1) == label).type(torch.float).sum().item()
            tbar.set_description(f"loss - {loss.item():.3f}")
            test_loss += loss.item()

    test_loss /= num_batches
    correct /= size

    return test_loss, correct
