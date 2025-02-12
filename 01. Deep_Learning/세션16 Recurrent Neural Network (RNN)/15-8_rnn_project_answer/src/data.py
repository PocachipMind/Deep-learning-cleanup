from datasets import load_dataset
from transformers import AutoTokenizer
from torch.utils.data import DataLoader


def get_imdb_dataset():
    dataset = load_dataset("imdb")
    return dataset


def get_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
    return tokenizer


def tokenize_dataset(dataset, tokenizer):
    dataset = dataset.map(
        lambda examples: tokenizer(
            examples["text"], truncation=True, padding="max_length"
        ),
        batched=True,
    )
    dataset.set_format(
        type="torch", columns=["input_ids", "token_type_ids", "attention_mask", "label"]
    )

    return dataset


def get_dataloader(tokenized_dataset):
    train_dataloader = DataLoader(
        tokenized_dataset["train"],
        batch_size=32,
        shuffle=True,
        num_workers=8,
        pin_memory=True,
    )
    test_dataloader = DataLoader(
        tokenized_dataset["test"],
        batch_size=32,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
    )
    return train_dataloader, test_dataloader
