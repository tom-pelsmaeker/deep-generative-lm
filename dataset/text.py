"""
Text datatset iterators, as an extension of the PyTorch Dataset class.

class SimpleTextData(): reads a text file line by line up to a specified sequence length.
class SimpleTextDataSplit(): extends SimpleTextData() by splitting the data in train and val sets.
class TextDataPadded(): extends SimpleTextData() by padding the text up to the specified sequence length.
"""
import os.path as osp
import sys

import numpy as np
import math
import torch
from torch.utils.data import Dataset

# We include the path of the toplevel package in the system path so we can always use absolute imports within the package.
toplevel_path = osp.abspath(osp.join(osp.dirname(__file__), '..'))
if toplevel_path not in sys.path:
    sys.path.insert(1, toplevel_path)

from util.error import InvalidLengthError  # noqa: E402

__author__ = "Tom Pelsmaeker"
__copyright__ = "Copyright 2018"


class SimpleTextData(Dataset):
    """Dataset of text that reads the first N tokens from each line in the given textfile as data.

    Args:
        file(str): name of the file containing the text data already converted to indices.
        seq_len(int): maximum length of sequences. Longer sequences will be cut at this length.
    """

    def __init__(self, file, seq_len):
        if seq_len == 0:
            self._seq_len = len(max(open(file, 'r'), key=len).split())
        else:
            self._seq_len = seq_len

        self._data = [line.split()[:self._seq_len] for line in open(file, 'r') if line != "\n"]
        self._data_len = len(self._data)

    def __len__(self):
        return self._data_len

    def __getitem__(self, idx):
        return torch.LongTensor(self._data[idx])


class TextDataSplit(SimpleTextData):
    """Dataset of text that allows a train/validation split from a single file. Extends SimpleTextData().

    Args:
        file(str): name of the file containing the text data already converted to indices.
        seq_len(int): maximum length of sequences. Longer sequences will be cut at this length.
        train(bool): True when training, False when testing.
    """

    def __init__(self, file, seq_len, train):
        super().__init__(file, seq_len)
        if train:
            self._data = self._data[:int(self.data.shape[0] * 0.9), :]
        else:
            self._data = self._data[int(self.data.shape[0] * 0.9):, :]
        self._data_len = self.data.shape[0]


class TextDataUnPadded(SimpleTextData):
    """
    Dataset of text that prepares sequences for padding, but does not pad them yet. Extends SimpleTextData().

    Args:
        file(str): name of the file containing the text data already converted to indices.
        seq_len(int): maximum length of sequences. shorter sequences will be padded to this length.
        pad_token(int): token that is appended to sentences shorter than seq_len.
    """

    def __init__(self, file, seq_len, pad_token):
        super().__init__(file, seq_len)

        # This class also provides reversed sequences that are needed in certain generative model training
        self._reverse_data = [line.split()[:self._seq_len][::-1] for line in open(file, 'r') if line != '\n']
        self._pad_token = pad_token

    def __getitem__(self, idx):
        return self._data[idx], self._reverse_data[idx], self._pad_token


class TextDataPadded(TextDataUnPadded):
    """
    Dataset of text that pads sequences up to the specified sequence length. Extends TextDataUnPadded().

    Args:
        file(str): name of the file containing the text data already converted to indices.
        seq_len(int): maximum length of sequences. shorter sequences will be padded to this length.
        pad_token(int): token that is appended to sentences shorter than seq_len.
    """

    def __init__(self, file, seq_len, pad_token):
        super().__init__(file, seq_len, pad_token)

        self._seq_lens = []
        for line in self._data:
            self._seq_lens.append(len(line))
            if len(line) < self._seq_len:
                line.extend([pad_token] * (self._seq_len - len(line)))
        for reverse_line in self._reverse_data:
            if len(reverse_line) < self._seq_len:
                reverse_line.extend([pad_token] * (self._seq_len - len(reverse_line)))

        self._seq_lens = torch.LongTensor(self._seq_lens)
        self._data = torch.from_numpy(np.array(self._data, dtype=np.int64))
        self._reverse_data = torch.from_numpy(np.array(self._reverse_data, dtype=np.int64))
        self._mask = 1. - (self._data == pad_token).float()

    def __getitem__(self, idx):
        return self._data[idx], self._seq_lens[idx], self._mask[idx], self._reverse_data[idx]


def sort_collate(batch):
    """Custom collate_fn for DataLoaders, sorts data based on sequence lengths.

    Note that it is assumed that the variable on which to sort will be in the second position of the input tuples.

    Args:
        batch(list of tuples): a batch of data provided by a DataLoader given a Dataset, i.e a list of length batch_size
            of tuples, where each tuple contains the variables of the DataSet at a single index.

    Returns:
        list of tensors: the batch of data, with a tensor of length batch_size per variable in the DataSet,
            sorted according to the second variable which is assumed to be length information. The list contains
            [data, lengths, ...].

    Raises:
        InvalidLengthError: if the input has less than two variables per index.
    """
    if len(batch[0]) < 2:
        raise InvalidLengthError("Batch needs to contain at least data (batch[0]) and lengths (batch[1]).")

    # Unpack batch from list of tuples [(x_i, y_i, ...), ...] to list of tensors [x, y, ...]
    batch = [torch.stack([b[i] for b in batch]) for i in range(len(batch[0]))]

    # Get lengths from second tensor in batch and sort all batch data based on those lengths
    _, indices = torch.sort(batch[1], descending=True)
    batch = [data[indices] for data in batch]

    return batch


def sort_pad_collate(batch):
    """Custom collate_fn for DataLoaders, pads data and sorts based on sequence lengths.

    This collate function works together with the TextDataUnPadded Dataset, that provides a batch of data in the correct
    format for this function to pad and sort.

    Args:
        batch(list of tuples): a batch of data provided by a DataLoader given a Dataset, i.e a list of length batch_size
            of tuples, where each tuple contains the variables of the DataSet at a single index. Each tuple must contain
            (data_i, reversed_data_i, pad_token).

    Returns:
        list of tensors: the batch of data, with a tensor of length batch_size per variable in the DataSet,
            sorted according to the second variable which is assumed to be length information. The list contains:
            [data, lengths, mask, reversed data].

    Raises:
        InvalidLengthError: if the input does not have three variables per index.
    """
    if len(batch[0]) != 3:
        raise InvalidLengthError(
            "Batch needs to contain data (batch[0]), reverse_data (batch[1]) and pad_token (batch[2]).")

    # Unpack batch from list of tuples [(x_i, y_i, ...), ...] to list of lists [x, y, ...]
    batch = [[b[i] for b in batch] for i in range(len(batch[0]))]

    # Pad tensors
    x_len = torch.tensor([len(line) for line in batch[0]])
    max_len = x_len.max().item()
    pad_token = batch[2][0]

    for line in batch[0]:
        if len(line) < max_len:
            line.extend([pad_token] * (max_len - len(line)))
    for line in batch[1]:
        if len(line) < max_len:
            line.extend([pad_token] * (max_len - len(line)))

    # Store data tensors in correct format and order
    batch[0] = torch.from_numpy(np.array(batch[0], dtype=np.int64))
    batch.append(torch.from_numpy(np.array(batch[1], dtype=np.int64)))
    # Store length and mask in correct format and order
    batch[1] = x_len
    batch[2] = 1. - (batch[0] == pad_token).float()

    # Get lengths from second tensor in batch and sort all batch data based on those lengths
    _, indices = torch.sort(batch[1], descending=True)
    batch = [data[indices] for data in batch]

    return batch
