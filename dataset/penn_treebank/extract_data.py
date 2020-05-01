#!/usr/bin/python3
"""
Script that extracts the data from the Penn Treebank and converts it to indices.
"""
from collections import Counter, defaultdict
import pickle
import sys
import os.path as osp

import numpy as np

# We include the path of the toplevel package in the system path so we can always use absolute imports within the package.
toplevel_path = osp.abspath(osp.join(osp.dirname(__file__), '../..'))
if toplevel_path not in sys.path:
    sys.path.insert(1, toplevel_path)

from util.settings import parse_arguments  # noqa: E402

__author__ = "Tom Pelsmaeker"
__copyright__ = "Copyright 2020"


def convert_to_indices(files, max_vocab, replace, sos, eos, pad):
    counter = Counter()
    counter.update([replace])
    sentences = []
    for i, file in enumerate(files):
        sentences.append([])
        with open(file, 'r', encoding='utf8') as f:
            sentences[i] = ["{} {} {}".format(sos, line, eos).split() for line in f]
            counter.update([token for sentence in sentences[i] for token in sentence])

    word_to_idx = defaultdict(lambda: 0)
    idx_to_word = {0: replace}
    num = 1
    for i, (token, _) in enumerate(counter.most_common(max_vocab)):
        if token is not replace:
            word_to_idx[token] = i + num
            idx_to_word[i + num] = token
        else:
            num = 0
    word_to_idx[pad] = max_vocab
    idx_to_word[max_vocab] = pad

    for file, dataset in zip(files, sentences):
        with open(file.split('.')[0] + ".indices", 'w', encoding='utf8') as f:
            f.write("\n".join([" ".join([str(word_to_idx[token]) for token in sentence]) for sentence in dataset]))

    pickle.dump(idx_to_word, open("idx_to_word.pickle", 'wb'))
    pickle.dump(dict(word_to_idx), open("word_to_idx.pickle", 'wb'))

    print("Number of unique tokens: {}".format(len(counter)))
    print("Vocab size: {}".format(len(idx_to_word)))


def get_stats(file, ret=False):
    with open(file, 'r', encoding='utf8') as f:
        lengths = np.array([len(line.split()) for line in f])

    mean = lengths.mean()
    std = lengths.std()
    max = lengths.max()
    min = lengths.min()
    num_sent = len(lengths)

    cutoff = int(mean + 3 * std)
    num_outliers = sum([1 for l in lengths if l > cutoff])
    pct_outliers = (float(num_outliers) / num_sent) * 100

    if ret:
        return mean, std, max, cutoff, pct_outliers
    else:
        print("{}| mean: {}, std: {}, max: {}, min: {}, pct outliers: {}".format(
            file, mean, std, max, min, pct_outliers))


if __name__ == '__main__':
    opt = parse_arguments()[0]
    in_files = ['train.txt', 'val.txt', 'test.txt']
    max_vocab = 10001

    convert_to_indices(in_files, max_vocab, opt.unk_token, opt.sos_token, opt.eos_token, opt.pad_token)
