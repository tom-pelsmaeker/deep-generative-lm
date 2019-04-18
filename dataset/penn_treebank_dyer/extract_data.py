#!/usr/bin/env python3
import sys
import os
osp = os.path
import re
from collections import Counter, defaultdict
import pickle

import numpy as np
import pandas as pd
from nltk.tree import Tree
from nltk.corpus import BracketParseCorpusReader

# We include the path of the toplevel package in the system path so we can always use absolute imports within the package.
toplevel_path = osp.abspath(osp.join(osp.dirname(__file__), '../..'))
if toplevel_path not in sys.path:
    sys.path.insert(1, toplevel_path)

from util.settings import parse_arguments  # noqa: E402


def parse_orcale_to_sent(file):
    with open(file, 'r', encoding='utf8') as f:
        text = f.read().split("\n\n")

    sentences = [s.split("\n")[4].split(" ") for s in text if s]
    return sentences


def extract_text_data(old_files, pre_files, new_files, replace, sos, eos, pad):
    counter = Counter()
    sentences = []
    counter.update([replace, sos, eos, pad])
    for i, file in enumerate(old_files):
        sentences.append(parse_orcale_to_sent(file))
        counter.update([token for sentence in sentences[i] for token in sentence])

    max_vocab = len(counter)
    word_list = [word[0] for word in counter.most_common(max_vocab)]

    for i, (file, new_file) in enumerate(zip(pre_files, new_files)):
        with open(file, 'r', encoding='utf8') as f:
            new_sentences = "\n".join(
                [" ".join(unkify("{} {} {}".format(sos, line, eos).split(), word_list)) for line in f])
        with open(new_file, 'w') as f:
            f.write(new_sentences)

    return counter


def extract_original_dyer(oracles, new_files, replace, sos, eos, pad):
    counter = Counter()
    sentences = []
    counter.update([replace, sos, eos, pad])
    for i, file in enumerate(oracles):
        sentences.append(parse_orcale_to_sent(file))
        counter.update([token for sentence in sentences[i] for token in sentence])

    max_vocab = len(counter)
    for i, file in enumerate(new_files):
        sentences[i] = "\n".join([" ".join([sos] + line + [eos]) for line in sentences[i]])
        with open(file, 'w') as f:
            f.write(sentences[i])

    return counter


def unkify(tokens, words_dict):
    """Taken from get_oracle.py"""
    final = []
    for token in tokens:
        # only process the train singletons and unknown words
        if len(token.rstrip()) == 0:
            final.append('UNK')
        if not(token.rstrip() in words_dict):
            numCaps = 0
            hasDigit = False
            hasDash = False
            hasLower = False
            for char in token.rstrip():
                if char.isdigit():
                    hasDigit = True
                elif char == '-':
                    hasDash = True
                elif char.isalpha():
                    if char.islower():
                        hasLower = True
                    elif char.isupper():
                        numCaps += 1
            result = 'UNK'
            lower = token.rstrip().lower()
            ch0 = token.rstrip()[0]
            if ch0.isupper():
                if numCaps == 1:
                    result = result + '-INITC'
                    if lower in words_dict:
                        result = result + '-KNOWNLC'
                else:
                    result = result + '-CAPS'
            elif not(ch0.isalpha()) and numCaps > 0:
                result = result + '-CAPS'
            elif hasLower:
                result = result + '-LC'
            if hasDigit:
                result = result + '-NUM'
            if hasDash:
                result = result + '-DASH'
            if lower[-1] == 's' and len(lower) >= 3:
                ch2 = lower[-2]
                if not(ch2 == 's') and not(ch2 == 'i') and not(ch2 == 'u'):
                    result = result + '-s'
            elif len(lower) >= 5 and not(hasDash) and not(hasDigit and numCaps > 0):
                if lower[-2:] == 'ed':
                    result = result + '-ed'
                elif lower[-3:] == 'ing':
                    result = result + '-ing'
                elif lower[-3:] == 'ion':
                    result = result + '-ion'
                elif lower[-2:] == 'er':
                    result = result + '-er'
                elif lower[-3:] == 'est':
                    result = result + '-est'
                elif lower[-2:] == 'ly':
                    result = result + '-ly'
                elif lower[-3:] == 'ity':
                    result = result + '-ity'
                elif lower[-1] == 'y':
                    result = result + '-y'
                elif lower[-2:] == 'al':
                    result = result + '-al'
            result = 'UNK' if result not in words_dict else result  # Added this line -Daan
            final.append(result)
        else:
            final.append(token.rstrip())

    return final


def convert_to_indices(files, counter):
    """Converts sentences to indices and adds special tokens."""
    max_vocab = len(counter)

    word_to_idx = dict()
    idx_to_word = dict()
    for i, (token, _) in enumerate(counter.most_common(max_vocab)):
        word_to_idx[token] = i
        idx_to_word[i] = token

    for file in files:
        with open(file, 'r', encoding='utf8') as f:
            dataset = [sen.split(" ") for sen in f.read().split("\n")]
        with open(file.split('.')[0] + ".indices", 'w', encoding='utf8') as f:
            f.write("\n".join([" ".join([str(word_to_idx[token]) for token in sentence]) for sentence in dataset]))

    pickle.dump(idx_to_word, open("idx_to_word.pickle", 'wb'))
    pickle.dump(dict(word_to_idx), open("word_to_idx.pickle", 'wb'))

    print("Number of unique tokens: {}".format(len(counter)))
    print("Vocab size: {}".format(len(idx_to_word)))


if __name__ == '__main__':
    opt = parse_arguments()[0]
    oracles = ['ptb.train.oracle', 'ptb.dev.oracle', 'ptb.test.oracle']
    files = ['train.txt', 'val.txt', 'test.txt']

    counter = extract_original_dyer(oracles, files, opt.unk_token, opt.sos_token, opt.eos_token, opt.pad_token)
    convert_to_indices(files, counter)
