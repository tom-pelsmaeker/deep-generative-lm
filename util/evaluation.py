"""Functions that aid the evaluation of various trained models."""

from functools import reduce
import sys
import os.path as osp
from random import random

import numpy as np
import torch
from nltk.translate.bleu_score import corpus_bleu
from scipy.stats import gaussian_kde
from pyter import ter

# We include the path of the toplevel package in the system path so we can always use absolute imports within the package.
toplevel_path = osp.abspath(osp.join(osp.dirname(__file__), '..'))
if toplevel_path not in sys.path:
    sys.path.insert(1, toplevel_path)

from util.error import UnknownArgumentError  # noqa: E402


"""The Functions below are from: https://github.com/napsternxg/pytorch-practice/blob/master/Pytorch%20-%20MMD%20VAE.ipynb"""


def compute_kernel(x, y):
    """Computes a positive definite kernel function between x and y.

    Args:
        x(torch.FloatTensor): [-1, x_dim] tensor.
        y(torch.FloatTensor): [-1, y_dim] tensor.

    Returns:
        torch.FloatTensor: kernel between x and y of [x_dim, y_dim].
    """
    x_size = x.size(0)
    y_size = y.size(0)
    dim = x.size(1)
    x = x.unsqueeze(1)  # (x_size, 1, dim)
    y = y.unsqueeze(0)  # (1, y_size, dim)
    tiled_x = x.expand(x_size, y_size, dim)
    tiled_y = y.expand(x_size, y_size, dim)
    kernel_input = (tiled_x - tiled_y).pow(2).mean(2)/float(dim)
    return torch.exp(-kernel_input)  # (x_size, y_size)


def compute_mmd(x, y):
    """(Biased) Estimator of the Maximum Mean Discrepancy two sets of samples."""
    x_kernel = compute_kernel(x, x)
    y_kernel = compute_kernel(y, y)
    xy_kernel = compute_kernel(x, y)
    mmd = x_kernel.mean() + y_kernel.mean() - 2*xy_kernel.mean()
    return mmd


"""------------------------------------------------------------------------------------------------------------------"""


def compute_mutual_information(samples, log_p_z, avg_H, avg_KL, method, kde_method, log_q_z=None):
    """Computes the mutual information given a batch of samples and their LL under the sampling distribution.

    If log_q_z is not provided, this method uses kernel density estimation on the provided samples to compute this
    quantity. Otherwise, we will use the provided log_q_z to estimate the MI, either through Hoffman's or Zhao's method.

    Args:
        samples(torch.FloatTensor): [N, z_dim] dimensional tensor of samples from q(z|x).
        log_p_z(torch.FloatTensor): [N] dimensional tensor of log-probabilities of the samples under p(z).
        avg_H(torch.FloatTensor): [] dimensional tensor containing the average entropy of q(z|x).
        avg_KL(torch.FloatTensor): [] dimensional tensor containing the average KL[q(z|x)||p(z)].
        method: method for estimating the mutual information.
        kde_method: method for obtaining the kde likelihood estimates of the samples under q(z).
        log_q_z: [N] dimensional tensor of log-probabilities of the samples under q(z), or None.
    Returns:
        mi_estimate(float): estimated mutual information between X and Z given N samples from q(z|x).
        marg_KL(float): estimated KL[q(z)||p(z)] given N samples from q(z|x).
    """
    if log_q_z is None:
        log_q_z = kde_gauss(samples, kde_method)

    marg_KL = (log_q_z - log_p_z).mean()

    if method == 'zhao':  # https://arxiv.org/pdf/1806.06514.pdf (Lagrangian VAE)
        mi_estimate = (avg_H - log_q_z.mean()).item()
    elif method == 'hoffman':  # http://approximateinference.org/accepted/HoffmanJohnson2016.pdf (ELBO surgery)
        mi_estimate = (avg_KL - marg_KL).item()
    else:
        raise UnknownArgumentError('MI method {} is unknown. Please choose [zhao, hoffman]'.format(method))

    return mi_estimate, marg_KL.item()


def kde_gauss(samples, method):
    """KDE estimation with a Gaussian kernel for a batch of samples. Returns the log probability of each sample."""
    if method == 'scipy':
        # [num, z_dim] -> [z_dim, num]
        samples_cpu = samples.cpu().numpy().transpose()
        kde = gaussian_kde(samples_cpu)
        # [num] with p(samples) under kernels
        return samples.new_tensor(kde.logpdf(samples_cpu))
    elif method == 'pytorch':
        # [num]
        return torch.log(samples.new_tensor(compute_kernel(samples.cpu(), samples.cpu()).mean(dim=1) + 1e-80))
    else:
        raise UnknownArgumentError('KDE method {} is unknown. Please choose [scipy, pytorch]'.format(method))


def compute_bleu(pred, data, pad_idx):
    """Computes the weighted corpus BLEU of predicted sentences.

    Args:
        pred(list): [num_sentences, max_len]. Predictions in index form.
        data(list): [num_sentences, max_len]. Gold standard indices.

    Return:
        float: corpus weighted BLEU for 1- to 4-grams between 0 and 100.
    """
    pred = [remove_padding(p, pad_idx) for p in pred]
    data = [[remove_padding(d, pad_idx)] for d in data]
    return corpus_bleu(data, pred) * 100


def compute_ter(pred, data, pad_idx):
    """Computes the translation error rate of predicted sentences.

    Args:
        pred(list): [num_sentences, max_len]. Predictions in index form.
        data(list): [num_sentences, max_len]. Gold standard indices.

    Return:
        float: corpus TER between 0 and 1.
    """
    pred = [remove_padding(p, pad_idx) for p in pred]
    data = [remove_padding(d, pad_idx) for d in data]
    return sum([ter(p, d) for p, d in zip(pred, data)]) / float(len(pred))


def compute_novelty(sentences, corpus_file, opt, idx_to_word):
    """Computes the novelty of a batch of sentences given a corpus."""
    # Prepare sampled sentences and corpus to compare to
    ref = sentences[0].split("\n")
    sentences = [s.split(" ") for s in sentences[1].split("\n")]
    with open(corpus_file, 'r') as f:
        corpus = [s.rstrip().split(" ") for s in f.readlines()]

    # Remove sentences much longer than the sampled sentences length
    corpus = [s for s in corpus if len(s) < opt.sample_len + 5]

    # Compute the novelty for each sentence
    novelty = []
    closest = []
    for i, sen in enumerate(sentences):
        print("Computing novelty for sentence {}/{}.\n".format(i, len(sentences)))
        mindex = np.argmin(np.array([ter(sen, s) for s in corpus]))
        novelty.append(ter(sen, corpus[mindex]))
        closest.append(" ".join([idx_to_word[int(idx)] for idx in corpus[mindex]]))
        print("Novelty: {}, Sentence: {}, Closest: {}\n".format(novelty[i], ref[i], closest[i]))
    return sum(novelty) / float(len(novelty)), sorted(zip(novelty, ref, closest))


def remove_padding(sentence, pad_idx):
    """Removes the paddings from a sentence"""
    try:
        return sentence[:sentence.index(pad_idx)]
    except ValueError:
        return sentence


def compute_active_units(mu, delta):
    """Computes an estimate of the number of active units in the latent space.

    Args:
        mu(torch.FloatTensor): [n_samples, z_dim]. Batch of posterior means.
        delta(float): variance threshold. Latent dimensions with a variance above this threshold are active.

    Returns:
        int: the number of active dimensions.
    """
    outer_expectation = torch.mean(mu, 0)**2
    inner_expectation = torch.mean(mu**2, 0)
    return torch.sum(inner_expectation - outer_expectation > delta).item()


def compute_accuracy(pred, data):
    """Computes the accuracy of predicted sequences.

    Args:
        pred(torch.Tensor): predictions produces by a model in index form, can be both a vector of last words
            and a matrix containing a batch of predicted sequences.
        data(torch.Tensor): the gold standard to compare the predictions against.

    Returns:
        float: the fraction of correct predictions.
    """
    # Handle different cases, 1 vs all outputs
    denom = reduce(lambda x, y: x * y, pred.shape)

    if len(data) == 1:
        target = data[0][:, 1:]
    else:
        # Here we have to ignore padding from the computation
        target = data[0][:, 1:]
        pred[data[2][:, 1:] == 0] = -1
        denom = denom - torch.sum(1. - data[2])

    return float(torch.eq(target, pred).sum()) / denom


def compute_perplexity(log_likelihoods, seq_lens):
    """Computes a MC estimate of perplexity per word based on given likelihoods/ELBO.

    Args:
        log_likelihoods(list of float): likelihood or ELBO from N runs over the same data.
        seq_lens(list of int): the length of sequences in the data, for computing an average.

    Returns:
        perplexity(float): perplexity per word of the data.
        variance(float): variance of the log_likelihoods/ELBO that were used to compute the estimate.
    """
    # Compute perplexity per word and variance of perplexities in the samples
    perplexity = np.exp(np.array(log_likelihoods).mean() / np.array(seq_lens).mean())
    if len(log_likelihoods) > 1:
        variance = np.array(log_likelihoods).mean(axis=1).std(ddof=1)
    else:
        variance = 0.0

    return perplexity, variance


def get_samples(opt, model, idx_to_word, word_to_idx):
    """Get a number of text samples from the provided model."""
    pad_idx = word_to_idx[opt.pad_token]
    try:
        samples = model.sample_sequences(torch.full(
            [opt.num_samples, 1], word_to_idx[opt.sos_token], dtype=torch.int64, device=opt.device), opt.sample_len, word_to_idx[opt.eos_token], pad_idx, opt.sample_softmax)
    except RuntimeError as e:
        raise RuntimeError("Not enough memory to sample.") from e

    sample_indices = "\n".join([" ".join([str(s) for s in sample if s != pad_idx]) for sample in samples])
    samples = "\n".join([" ".join([idx_to_word[s] for s in sample if s != pad_idx]) for sample in samples])

    return samples, sample_indices


class AdaptiveRate():
    """Adapt the minimum rate for a hinge loss in a model."""

    def __init__(self, opt):
        self.opt = opt
        self.prev_elbo = []
        self.prev_kl = []
        self.ticker = 0

        if self.opt.kl_step < 1.:
            self.opt.warm_up_rate = int(1. / self.opt.kl_step)

    def __call__(self, losses, model):
        """Adapt rate of model to the best observed KL + an increment."""
        self.prev_elbo.append(losses["NLL"].item() + losses["KL"].item())
        self.prev_kl.append(losses["KL"].item() / model.scale.item())
        if self.opt.num_rate_check >= 1.:
            if self.ticker % self.opt.num_rate_check == 0 and self.ticker > self.opt.warm_up_rate:
                model.min_rate = self.prev_kl[np.argmin(self.prev_elbo)] + self.opt.rate_increment
        self.ticker += 1
