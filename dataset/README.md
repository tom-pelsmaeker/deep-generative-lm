# Dataset

## Overview
We currently supply pre-processing tools for two versions of the Penn Treebank. The folder [penn_treebank_dyer](https://github.com/tom-pelsmaeker/deep-generative-lm/tree/master/dataset/penn_treebank_dyer/) contains tools to extract data from oracle files with complex masking of unknown types, as used in the [Recurrent Neural Network Grammar](https://arxiv.org/abs/1602.07776) paper. This data was used in all experiments of [our paper](https://arxiv.org/abs/1904.08194). The folder [penn_treebank](https://github.com/tom-pelsmaeker/deep-generative-lm/tree/master/dataset/penn_treebank/) contains tools to extract data from text files with simple masking of unknown types and a smaller vocabulary (V=10000), as used by T. Mikolov in his [PhD thesis](http://www.fit.vutbr.cz/~imikolov/rnnlm/thesis.pdf).

__Results will differ from our paper when using the data with simple masking.__

## Pre-processing
Make sure the correct train, val and test files are present in the correct folder. Then navigate to the desired folder (penn_treebank or penn_treebank_dyer) and run:
```
./extract_data
```
This will preprocess the data to the correct format and produce files containing the sentences in index form. Now, models can be trained on the data using the [main script](https://github.com/tom-pelsmaeker/deep-generative-lm/blob/master/main.py). Using the --ptb_type argument when running the main script will help the script to automatically find the data; `--ptb_type mik` will look for data with simple masking, and `--ptb_type dyer` for the data with complex masking.

## Obtaining the data
The Penn Treebank dataset is only available under a paid license, so we cannot directly supply the oracle files. However, if you or your institute are licensed to use the PTB, feel free to send me an email with proof and I will provide the data we used in our paper.

Mikolovs data has been released [here](https://github.com/townie/PTB-dataset-from-Tomas-Mikolov-s-webpage/tree/master/data). __Make sure you have the rights to use this data.__

## Using other data
In principle, the models can be trained on any textual data that is clearly separated into sequences. Thus, if you wish to use this repository with other data, write your own pre-processing script that outputs a train, val and test file in index format, as well as pickled word-to-index and index-to-word dictionaries. If you then provide the path to the folder containing this pre-processed data to the main script, it *should* work.
