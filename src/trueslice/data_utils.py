# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
#
# Modified for TrueSlice.

import logging

import datasets
import torch
from torch.utils.data import DataLoader, Dataset, Subset, SubsetRandomSampler
from transformers import PreTrainedTokenizerBase


def get_dataset(name: str,
    *,
    seed: int = 42,
) -> datasets.DatasetDict:
    """
    Get the dataset from the HuggingFace datasets library.

    Args:
        name: The name of the HuggingFace dataset to load. Must be one of "wikitext2", "ptb", "c4" or "alpaca".

    Returns:
        The dataset.
    """
    logging.info(f"Loading dataset: {name}")

    ds_properties = {
        "wikitext2": {"path": "wikitext", "config_name": "wikitext-2-raw-v1"},
        "ptb": {"path": "ptb_text_only", "config_name": "penn_treebank"},
        "c4": {
            "path": "allenai/c4",
            "config_name": "en",
            "data_files": {
                "train": "en/c4-train.00000-of-01024.json.gz",
                "validation": "en/c4-validation.00000-of-00008.json.gz",
            },
            "verification_mode": "no_checks",
            "cols_to_remove": ['url', 'timestamp'],
        },
        "alpaca": {"path": "tatsu-lab/alpaca", "cols_to_remove": ['input', 'output', 'instruction']},
    }

    if name not in ds_properties:
        raise NotImplementedError("The provided dataset is not supported")

    properties = ds_properties[name]
    ds = datasets.load_dataset(
        properties["path"], name=properties.get("config_name"), data_files=properties.get("data_files"), verification_mode=properties.get("verification_mode")
    )

    if "cols_to_remove" in properties:
        ds = ds.remove_columns(properties["cols_to_remove"])

    # if alpaca, create a test and validation set from the training set
    if name == "alpaca":
        ds = ds["train"].train_test_split(test_size=0.2, seed=seed)
        temp_ds = ds.pop("test")
        temp_ds = temp_ds.train_test_split(test_size=0.5, seed=seed)
        ds["test"] = temp_ds["train"]
        ds["validation"] = temp_ds["test"]

    logging.info("Loading dataset done")
    return ds


def prepare_train_dataloader(
    dataset: datasets.Dataset,
    tokenizer: PreTrainedTokenizerBase,
    max_seqlen: int = 2048,
    batch_size: int = 1,
    nsamples: int = 128,
    varied_seqlen: bool = False,
    seed=42,
    exact_seqlen: bool = False,
) -> DataLoader[dict[str, torch.Tensor]]:
    """
    Get a DataLoader from a dataset.

    Args:
        dataset: The dataset to create a dataloader from.
        tokenizer: The tokenizer to use.
        max_seqlen: The maximum sequence length, used for truncation of sequences in the dataset.
        batch_size: The batch size.
        nsamples: The number of samples to produce.
        varied_seqlen: If False, concatenate multiple examples from the dataset into one example until max_seqlen is reached.
        seed: The seed for sampling the dataset.
        exact_seqlen: Preserve sampled token IDs without a text round trip (fixed-length benchmarks).

    Returns:
        A DataLoader.
    """
    if exact_seqlen and (varied_seqlen or not nsamples):
        raise ValueError("exact_seqlen requires fixed-length sampling and a positive nsamples")

    logging.info(f"Preparing dataloader")

    if not varied_seqlen and not nsamples:
        logging.warning(
            "varied_seqlen=False, but nsamples is not specified. This will lead to tokenization of the entire dataset, which will be slow."
        )

    data_name = dataset.column_names[0]
    ds = dataset.filter(lambda x: len(x[data_name]) > 0)

    if not varied_seqlen:
        # create a new dataset where each example is a concatenation of multiple examples of total length = max_seqlen.
        data_list = ds[data_name]
        new_data_list = []

        generator = torch.Generator().manual_seed(seed)
        indices = list(range(len(data_list)))

        while len(new_data_list) < nsamples and len(indices) > 0:
            start_idx = torch.randint(0, len(indices), (1,), generator=generator).item()
            idx = start_idx
            tokens = []
            while len(tokens) < max_seqlen and idx < len(indices):
                item = data_list[indices[idx]]
                sep = "" if not tokens else "\n\n"
                tokens += tokenizer.tokenize(sep + item)
                idx += 1

            indices = indices[:start_idx] + indices[idx:]  # remove the used indices

            if len(tokens) >= max_seqlen:
                tokens = tokens[:max_seqlen]  # truncate to max_seqlen
                if exact_seqlen:
                    new_data_list.append(tokenizer.convert_tokens_to_ids(tokens))
                else:
                    new_data_list.append(tokenizer.convert_tokens_to_string(tokens))

        ds = datasets.Dataset.from_dict({data_name: new_data_list})

    def tokenize(data_batch):
        if exact_seqlen:
            input_ids = torch.tensor(data_batch[data_name], dtype=torch.long)
            return {"input_ids": input_ids, "attention_mask": torch.ones_like(input_ids), "labels": input_ids.clone()}

        # tokenize then pad each batch according to the longest sequence in the batch
        batch = tokenizer(
            data_batch[data_name],
            padding="longest",
            max_length=max_seqlen,
            truncation=True,
            return_tensors="pt",
        )
        batch["labels"] = batch["input_ids"].clone()
        return batch

    # tokenize lazily
    ds.set_transform(tokenize)

    # Keep sampling and batch order independent of model initialization, optional
    # evaluations, and other consumers of the global random number generator.
    generator = torch.Generator().manual_seed(seed)
    sampler = SubsetRandomSampler(torch.randperm(len(ds), generator=generator)[:nsamples], generator=generator)

    loader = DataLoader(ds, batch_size=batch_size, sampler=sampler, generator=generator)
    logging.info(f"Preparing dataloader done")
    return loader


def prepare_test_dataloader(
    dataset: datasets.Dataset,
    tokenizer: PreTrainedTokenizerBase,
    seqlen: int = 2048,
    batch_size: int = 1,
    *,
    nsamples: int | None = None,
    seed: int = 42,
) -> DataLoader[dict[str, torch.Tensor]]:
    """
    Pack the test corpus into full sequences and optionally select a seeded subset.

    With nsamples=None, evaluate the full packed corpus for standard perplexity
    comparisons. A positive nsamples caps the number of sequences, not batches.

    Args:
        dataset: The dataset to create a dataloader from.
        tokenizer: The tokenizer to use.
        seqlen: The sequence length of sequences in the dataset.
        batch_size: The batch size.
        nsamples: Maximum number of full sequences, or None for the whole corpus.
        seed: Seed for selecting evaluation sequences.

    Returns:
        A DataLoader.
    """

    logging.info(f"Preparing test dataloader")

    class TestDataset(Dataset):
        def __init__(self, ds, tokenizer, seqlen=2048):
            """Tokenize the entire dataset and reshape it into sequences of length seqlen."""

            text_column = 'text' if 'text' in ds.column_names else 'sentence'
            tokenized_ds = tokenizer("\n\n".join(ds[text_column]), return_tensors='pt')
            nsamples = tokenized_ds.input_ids.numel() // seqlen

            input_ids = tokenized_ds.input_ids[0, : nsamples * seqlen]
            input_ids = input_ids.reshape(nsamples, seqlen)
            attn_mask = tokenized_ds.attention_mask[0, : nsamples * seqlen]
            attn_mask = attn_mask.reshape(nsamples, seqlen)

            self.input_ids = input_ids
            self.attn_mask = attn_mask

        def __getitem__(self, idx):
            return {"input_ids": self.input_ids[idx], "attention_mask": self.attn_mask[idx]}

        def __len__(self):
            return len(self.input_ids)

    test_ds = TestDataset(dataset, tokenizer, seqlen)
    if not len(test_ds):
        raise ValueError("Evaluation dataset is too short to produce one full sequence")

    generator = torch.Generator().manual_seed(seed)
    if nsamples is not None:
        if nsamples < len(test_ds):
            indices = torch.randperm(len(test_ds), generator=generator)[:nsamples].tolist()
            test_ds = Subset(test_ds, indices)
        elif nsamples > len(test_ds):
            logging.warning("Requested %d evaluation sequences; only %d are available", nsamples, len(test_ds))

    loader = DataLoader(test_ds, batch_size=batch_size, generator=generator)
    logging.info("Evaluation uses %d sequences of %d tokens", len(test_ds), seqlen)
    logging.info(f"Preparing test dataloader done")
    return loader
