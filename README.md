# Effective Estimation of Deep Generative Language Models

## Overview
This repository contains the code needed to run the experiments presented in the paper [Effective Estimation of Deep Generative Language Models](https://arxiv.org/abs/1904.08194)[[1 ]](https://github.com/tom-pelsmaeker/deep-generative-lm#citation).

## Setup
To start experimenting, clone the repository to your local device and install the following dependencies:
- __python__ >= 3.6
- pip install -r __requirements.txt__
- __hyperspherical_vae__: the code was tested with [this fork](https://github.com/tom-pelsmaeker/s-vae-pytorch), get the latest version from [here](https://github.com/nicola-decao/s-vae-pytorch).
- [__torch_two_sample__](https://github.com/josipd/torch-two-sample)

## Quick Start
1. Download and pre-process the Penn Treebank data, see the [data folder](https://github.com/tom-pelsmaeker/deep-generative-lm/dataset/).
2. Train a RNNLM:
```
./main.py --model deterministic --mode train --pre_def 1 --ptb_type mik
```
3. Train a default SenVAE:
```
./main.py --model bowman --mode train --pre_def 1 --ptb_type mik
```
4. Train a SenVAE with a target rate of 5, using MDR:
```
./main.py --model bowman --mode train --pre_def 1 --ptb_type mik --lagrangian 1 --min_rate 5 --save_suffix mdr
```
5. Train a SenVAE with IAF flow and a target rate of 5, using MDR:
```
./main.py --model flowbowman --flow iaf --mode train --pre_def 1 --ptb_type mik --lagrangian 1 --min_rate 5 --save_suffix iaf
```
6. Evaluate the models:
```
./main.py --model deterministic --mode test --pre_def 1 --ptb_type mik
./main.py --model bowman --mode test --pre_def 1 --ptb_type mik
./main.py --model bowman --mode test --pre_def 1 --ptb_type mik --lagrangian 1 --min_rate 5 --save_suffix mdr
./main.py --model flowbowman --flow iaf --mode test --pre_def 1 --ptb_type mik --lagrangian 1 --min_rate 5 --save_suffix iaf
```
6. Print some samples:
```
./main.py --model deterministic --mode qualitative --pre_def 1 --ptb_type mik
./main.py --model bowman --mode qualitative --pre_def 1 --ptb_type mik
./main.py --model bowman --mode qualitative --pre_def 1 --ptb_type mik --lagrangian 1 --min_rate 5 --save_suffix mdr
./main.py --model flowbowman --flow iaf --mode qualitative --pre_def 1 --ptb_type mik --lagrangian 1 --min_rate 5 --save_suffix iaf
```

## Structure
- [main.py](https://github.com/tom-pelsmaeker/deep-generative-lm/main.py): the main script that handles all command line arguments.
- [dataset](https://github.com/tom-pelsmaeker/deep-generative-lm/dataset/): expected location of data. Contains code for preprocessing and batching PTB.
- [model](https://github.com/tom-pelsmaeker/deep-generative-lm/model/): all components for the various models tested in the paper.
- [scripts](https://github.com/tom-pelsmaeker/deep-generative-lm/scripts/): various scripts for training/testing/BayesOpt, etcetera.
- [util](https://github.com/tom-pelsmaeker/deep-generative-lm/util/): utility functions for storage, evaluation and more.

## Settings
There are many command line settings available to tweak the experimental setup. Please see the [settings file](https://github.com/tom-pelsmaeker/deep-generative-lm/util/settings.py) for a complete overview. Here, we will highlight the most important settings:
```
--script: [generative|bayesopt|grid] chooses which script to run. generative is used for training/testing a single model, bayesopt and grid run Bayesian Optimization and Grid search respectively. Please see the scripts for more information about their usage.
--mode: [train|test|novelty|qualitative] select in which mode to run the generative script.
--save_suffix: to give your model a name.
--seed: set a random seed.
--model: [deterministic|bowman|flowbowman] the model to use. Deterministic refers to the RNNLM, bowman to the SenVAE and flowbowman to the SenVAE with expressive latent model.
--lagrangian: set to 1 to use the MDR objective.
--min_rate: specify a minimum rate, in nats.
--flow: [diag|iaf|vpiaf|planar] the type of flow to use with the flowbowman model.
--prior: [weak|mog|vamp] the type of prior to use with the flowbowman model.
--data_folder: path to your pre-processed data.
--out_folder: path to store experiments.  
--ptb_type: [|mik|dyer] choose between simple (mik) and expressive (dyer) unked PTB. paths to out and data are set automatically.
--pre_def: set to 1 to use encoder-decoder hyperparameters that match the ones in the paper.
--local_rank: which GPU to use. Set to -1 to run on CPU.
```

## Further Usage Examples
__TODO__

## Citation
If you use this code in your project, please cite:
```
[1] Pelsmaeker, T., and Aziz, W. (2019). Effective Estimation of Deep Generative Language Models. arXiv preprint arXiv:1904.08194.
```

BibTeX format:
```
@article{effective2019pelsmaeker,
  title={Effective Estimation of Deep Generative Language Models},
  author={Pelsmaeker, Tom and
          Aziz, Wilker},
  journal={arXiv preprint arXiv:1904.08194},
  year={2019}
}
```

## License
MIT
