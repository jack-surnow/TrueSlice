# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
#
# Modified for TrueSlice.

import logging
import random
from itertools import islice

import datasets
import torch
from torch.utils.data import DataLoader, Dataset
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
        },
        "alpaca": {"path": "tatsu-lab/alpaca"},
    }

    if name not in ds_properties:
        raise NotImplementedError("The provided dataset is not supported")

    properties = ds_properties[name]
    ds = datasets.load_dataset(
        properties["path"], name=properties.get("config_name"), data_files=properties.get("data_files"), verification_mode=properties.get("verification_mode")
    )

    if name == "wikitext2":
        ds = _prepare_wikitext2(ds)
    elif name == "c4":
        ds = _prepare_c4(ds)
    elif name == "alpaca":
        ds = _prepare_alpaca(ds, seed=seed)

    logging.info("Loading dataset done")
    return ds


def _prepare_wikitext2(dataset: datasets.DatasetDict) -> datasets.DatasetDict:
    """Group each split's paragraphs and headings into complete article rows."""
    prepared = datasets.DatasetDict()
    for split, data in dataset.items():
        documents, rows = [], []
        seen_title = False
        for text in data['text']:
            title = text.strip()
            # '= Article =' starts a document; '= = Section = =' does not.
            if title.startswith('= ') and title.endswith(' =') and not title.startswith('= ='):
                if seen_title:
                    documents.append('\n\n'.join(rows))
                    rows = []
                seen_title = True
            rows.append(text)
        if rows:
            documents.append('\n\n'.join(rows))
        prepared[split] = datasets.Dataset.from_dict({'text': documents})
        logging.info('WikiText %s: assembled %d articles from %d rows', split, len(documents), len(data))
    return prepared


def _prepare_c4(dataset: datasets.DatasetDict) -> datasets.DatasetDict:
    """Keep corpus text without web metadata."""
    return dataset.remove_columns(['url', 'timestamp'])


def _prepare_alpaca(dataset: datasets.DatasetDict, *, seed: int) -> datasets.DatasetDict:
    """Keep the combined text and create seeded 80/10/10 splits."""
    dataset = dataset.remove_columns(['input', 'output', 'instruction'])
    prepared = dataset['train'].train_test_split(test_size=0.2, seed=seed)
    held_out = prepared.pop('test').train_test_split(test_size=0.5, seed=seed)
    prepared['test'] = held_out['train']
    prepared['validation'] = held_out['test']
    return prepared


class _TokenDataset(Dataset):
    """Token sequences padded to seqlen, with masks marking only real tokens."""

    def __init__(self, sequences, tokenizer, seqlen):
        self.sequences = sequences
        self.seqlen = seqlen
        self.padding_side = tokenizer.padding_side
        self.pad_token_id = tokenizer.pad_token_id
        if self.pad_token_id is None:
            self.pad_token_id = tokenizer.eos_token_id
        if self.pad_token_id is None and any(len(ids) < seqlen for ids in sequences):
            raise ValueError('Padding requires a tokenizer with a pad_token_id or eos_token_id')

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, index):
        ids = self.sequences[index]
        mask = torch.ones_like(ids)
        padding = self.seqlen - len(ids)
        if padding:
            sides = (padding, 0) if self.padding_side == 'left' else (0, padding)
            ids = torch.nn.functional.pad(ids, sides, value=self.pad_token_id)
            mask = torch.nn.functional.pad(mask, sides, value=0)
        return {'input_ids': ids, 'attention_mask': mask}


def _text_column(dataset):
    """Prefer the corpus text over any metadata columns."""
    return next((name for name in ('text', 'sentence') if name in dataset.column_names), dataset.column_names[0])


def _token_sequences(dataset, tokenizer, seqlen, varied_seqlen, seed, document_limit):
    """Shuffle complete documents and yield chunks, carrying overflow forward."""
    column = _text_column(dataset)
    separator = tokenizer.encode('\n\n', add_special_tokens=False) if not varied_seqlen else []
    window_rng = random.Random(seed)  # Window draws must not affect document order.
    pending = []
    first_document = True
    for row in dataset.shuffle(seed=seed):
        text = row[column]
        if not text or not text.strip():
            continue
        # Long documents are split below, so the tokenizer's context-length
        # warning does not apply. Only documents needed for nsamples are encoded.
        tokens = tokenizer.encode(text, add_special_tokens=False, verbose=False)
        if not tokens:
            continue
        if document_limit is not None and len(tokens) > document_limit:
            start = window_rng.randrange(len(tokens) - document_limit + 1)
            tokens = tokens[start:start + document_limit]
        tokens = pending + ([] if first_document else separator) + tokens
        first_document = False
        stop = len(tokens) // seqlen * seqlen
        for start in range(0, stop, seqlen):
            yield tokens[start:start + seqlen]
        pending = tokens[stop:]
        if varied_seqlen and pending:
            yield pending
            pending = []
    # A packed tail is dropped only when the corpus has been exhausted.


def _prepare_dataloader(dataset, tokenizer, seqlen, batch_size, nsamples, varied_seqlen, seed, *, calibration):
    if seqlen <= 0 or batch_size <= 0 or (nsamples is not None and nsamples <= 0):
        raise ValueError('Sequence length, batch size, and sample count must be positive')
    purpose = 'Calibration' if calibration else 'Evaluation'
    prefix = []
    if calibration and tokenizer.bos_token_id is not None:
        # Having a BOS token does not mean a tokenizer inserts it by default.
        # Follow the native policy (e.g. Llama adds BOS; Qwen does not).
        if tokenizer.encode('', add_special_tokens=True)[:1] == [tokenizer.bos_token_id]:
            prefix = [tokenizer.bos_token_id]
    content_length = seqlen - len(prefix)
    if content_length <= 0:
        raise ValueError('Sequence length must leave room for content after BOS')
    chunks = _token_sequences(dataset, tokenizer, content_length, varied_seqlen, seed,
                              document_limit=4 * seqlen if calibration else None)
    sequences = [torch.tensor(prefix + ids, dtype=torch.long)
                 for ids in islice(chunks, nsamples)]
    if not sequences:
        raise ValueError(f'{purpose} dataset is too short to produce one sequence')
    if nsamples is not None and len(sequences) < nsamples:
        logging.warning('Requested %d %s sequences; only %d are available',
                        nsamples, purpose.lower(), len(sequences))
    token_dataset = _TokenDataset(sequences, tokenizer, seqlen)
    logging.info('%s uses %d sequences of %d tokens (%d nonpadding tokens)',
                 purpose, len(sequences), seqlen, sum(len(ids) for ids in sequences))
    # Selection is fixed at preparation time; iteration never reshuffles samples
    # or consumes the global PyTorch RNG.
    return DataLoader(token_dataset, batch_size=batch_size,
                      generator=torch.Generator().manual_seed(seed))


def prepare_cal_dataloader(
    dataset: datasets.Dataset,
    tokenizer: PreTrainedTokenizerBase,
    seqlen: int = 2048,
    batch_size: int = 1,
    *,
    nsamples: int | None = 128,
    varied_seqlen: bool = False,
    seed: int = 42,
) -> DataLoader[dict[str, torch.Tensor]]:
    """Shuffle documents and retain up to 4 * seqlen tokens from each.

    Longer documents contribute a seeded random window. By default, pack with
    '\n\n' separators into full sequences and discard only the final corpus tail.
    varied_seqlen keeps documents separate and pads their final chunks to seqlen.
    When the tokenizer normally adds BOS, reserve one slot for it in every chunk;
    content overflow is carried forward rather than truncated to make room.
    nsamples counts sequences, including padded chunks; None consumes the corpus.
    Dataset rows must be complete documents; get_dataset assembles WikiText articles.
    """
    return _prepare_dataloader(dataset, tokenizer, seqlen, batch_size, nsamples,
                               varied_seqlen, seed, calibration=True)


def prepare_eval_dataloader(
    dataset: datasets.Dataset,
    tokenizer: PreTrainedTokenizerBase,
    seqlen: int = 2048,
    batch_size: int = 1,
    *,
    nsamples: int | None = None,
    varied_seqlen: bool = False,
    seed: int = 42,
) -> DataLoader[dict[str, torch.Tensor]]:
    """Shuffle complete documents without caps or random windows.

    Pack with '\n\n' separators, preserving overflow and discarding only the
    final incomplete corpus tail. varied_seqlen instead keeps documents separate
    and pads each final chunk. Stop at nsamples; None consumes all documents.
    Every iteration reuses the same sequences in the same seeded order.
    """
    return _prepare_dataloader(dataset, tokenizer, seqlen, batch_size, nsamples,
                               varied_seqlen, seed, calibration=False)
