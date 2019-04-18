"""
A deterministic decoder.
"""

import numpy as np
import sys
import os.path as osp
from time import time
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.distributions import Normal, Categorical, Bernoulli
from torch.nn.utils.rnn import pack_padded_sequence, PackedSequence, pad_packed_sequence

# We include the path of the toplevel package in the system path so we can always use absolute imports within the package.
toplevel_path = osp.abspath(osp.join(osp.dirname(__file__), '..'))
if toplevel_path not in sys.path:
    sys.path.insert(1, toplevel_path)

from util.error import UnknownArgumentError  # noqa: E402
from model.dropout import FlexibleDropout  # noqa: E402
from model.base_decoder import BaseDecoder  # noqa: E402


__author__ = "Tom Pelsmaeker"
__copyright__ = "Copyright 2018"


class DeterministicDecoder(BaseDecoder):
    """A deterministic decoder, i.e. a RNN with next-word prediction objective.

    Args:
        device(torch.device): the device (cpu/gpu) on which the model resides.
        seq_len(int): maximum length of sequences passed to the model.
        kl_step(int): step size of linear kl weight increment during training of the model.
        word_p(float): probability of dropping a word, i.e. mapping it to <unk>, before decoding.
        parameter_p(float): probability of dropping a row in the weight layers, using Gal's dropout on non-rec layers.
        var_mask(boolean): whether to use a different parameter dropout mask at every timestep.
        unk_index(int): index of the <unk> token in a one-hot representation.
        css(boolean): whether to use CSS softmax approximation.
        N(int): number of sequences in the dataset, for the regularization weight.
        rnn_type(str): which RNN to use. [GRU, LSTM] are supported.
        v_dim(int): size of the vocabulary.
        x_dim(int): size of input embeddings.
        h_dim(int): size of hidden layers of the RNN.
        l_dim(int): number of layers of the RNN.
    """

    def __init__(self, device, seq_len, word_p, parameter_p, drop_type, unk_index, css, sparse, N, rnn_type, tie_in_out, v_dim, x_dim, h_dim, s_dim, l_dim):
        super(DeterministicDecoder, self).__init__(device, seq_len, word_p, parameter_p,
                                                   drop_type, unk_index, css, N, rnn_type, v_dim, x_dim, h_dim, s_dim, l_dim)

        self.tie_in_out = tie_in_out

        # The model embeds words and passes them through the RNN to get a probability of next words.
        self.emb = nn.Embedding(v_dim, x_dim, sparse=bool(sparse))
        # We currently support GRU and LSTM type RNNs
        if rnn_type == "GRU":
            if self.drop_type in ["varied", "shared"]:
                # Varied and shared dropout modes only drop input and output layer. Shared shares between timesteps.
                self.grnn = nn.GRU(x_dim, h_dim, l_dim, batch_first=True)
            else:
                self.grnn = nn.ModuleList([nn.GRUCell(x_dim, h_dim, 1)])
                self.grnn.extend([nn.GRUCell(h_dim, h_dim, 1)
                                  for _ in range(l_dim - 1)])
        elif rnn_type == "LSTM":
            if self.drop_type in ["varied", "shared"]:
                self.grnn = nn.LSTM(x_dim, h_dim, l_dim, batch_first=True)
            else:
                self.grnn = nn.ModuleList([nn.LSTMCell(x_dim, h_dim, 1)])
                self.grnn.extend([nn.LSTMCell(h_dim, h_dim, 1)
                                  for _ in range(l_dim - 1)])
        self.linear = nn.Linear(h_dim, v_dim)

    @property
    def linear(self):
        return self._linear

    @linear.setter
    def linear(self, val):
        self._linear = val

        if self.tie_in_out:
            if self.h_dim != self.x_dim:
                raise InvalidArgumentError("h_dim should match x_dim when tying weights.")
            self._linear.weight = self.emb.weight

    def forward(self, data, log_likelihood=False, extensive=False):
        """Forward pass through the decoder which returns a loss and prediction.

        Args:
            data(list of torch.Tensor): a batch of datapoints, containing at least a tensor of sequences and optionally
                tensors with length information and a mask as well, given variable length sequences.

        Returns:
            loss(torch.FloatTensor): computed loss, averaged over the batch, summed over the sequence.
            kl(torch.FloatTensor): For compatibility with latent variable models; always zero.
            pred(torch.LongTensor): most probable sequences given the data, as predicted by the model.
        """
        x_in, x_len, x_mask = self._unpack_data(data, 3)
        losses = defaultdict(lambda: torch.tensor(0., device=self.device))

        # Before decoding, we map a fraction of words to <UNK>, weakening the Decoder
        self.word_dropout.sample_mask(self.word_p, x_in.shape)
        x_dropped = x_in.clone()
        x_dropped[self.word_dropout._mask == 0] = self.unk_index
        x = self.emb(x_dropped[:, :-1])

        scores = self._rnn_forward(x, x_len)

        # Compute loss, averaged over the batch, but summed over the sequence
        if self.css and self.training:
            loss = self._css(scores, x_in[:, 1:])
        else:
            loss = self.reconstruction_loss(scores.contiguous().view(
                [-1, scores.shape[2]]), x_in[:, 1:].contiguous().view([-1])).view(scores.shape[0], scores.shape[1])

        if x_len is not None:
            # If we had padded sequences as input, we need to mask the padding from the loss
            losses["NLL"] = torch.sum(torch.mean(loss * x_mask[:, 1:], 0))
        else:
            losses["NLL"] = torch.sum(torch.mean(loss, 0))

        # We also return the predictions, i.e. the most probable token per position in the sequences
        pred = torch.max(scores.detach(), dim=2)[1]

        # We use L2-regularization scaled by dropout on the network layers (Gal, 2015)
        losses["L2"] = self._l2_regularization()

        if log_likelihood:
            losses["NLL"] = losses["NLL"].unsqueeze(0)

        if extensive:
            return losses, pred, x.new_tensor([[1, 1]]), x.new_tensor([[1, 1]]), x.new_tensor([[1, 1]]), x.new_tensor([[1, 1]]), x.new_tensor([[1]]),  x.new_tensor([[1]])
        else:
            return losses, pred

    def _rnn_forward(self, x, x_len):
        """Recurrent part of the forward pass. Decides between fast or slow based on the dropout type."""
        # Drop rows of the input
        shape = torch.Size(x.shape) if self.var_mask else torch.Size([x.shape[0], 1, self.x_dim])
        h = self.parameter_dropout_in(x, self.parameter_p, shape=shape)

        # We have to run a (slow) for loop to use recurrent dropout
        if self.drop_type == "recurrent":
            # Sample fixed dropout masks for every timestep
            shape = torch.Size([x.shape[0], int(self.h_dim/self.l_dim)])
            for i in range(self.l_dim):
                self.parameter_dropout_hidden[i].sample_mask(self.parameter_p, shape)
                self.parameter_dropout_out[i].sample_mask(self.parameter_p, shape)
                if self.rnn_type == "LSTM":
                    self.parameter_dropout_context[i].sample_mask(self.parameter_p, shape)

            # Forward passing with application of dropout
            scores = []
            if self.rnn_type == "GRU":
                h_p = list(torch.unbind(self._init_hidden(x.shape[0])))
            else:
                h_p = list(torch.unbind(self._init_hidden(x.shape[0])))
                c_p = list(torch.unbind(self._init_hidden(x.shape[0])))
            for j in range(x.shape[1]):
                h_j = h[:, j, :]
                for i, grnn in enumerate(self.grnn):
                    if self.rnn_type == "GRU":
                        h_j = grnn(h_j, h_p[i])
                        h_p[i] = self.parameter_dropout_hidden[i].apply_mask(h_j)
                    else:
                        h_j, c_j = grnn(h_j, (h_p[i], c_p[i]))
                        h_p[i] = self.parameter_dropout_hidden[i].apply_mask(h_j)
                        c_p[i] = self.parameter_dropout_context[i].apply_mask(c_j)
                    h_j = self.parameter_dropout_out[i].apply_mask(h_j)
                scores.append(self.linear(h_j))
            scores = torch.stack(scores, 1)
        # For the input/output dropout we can use fast CUDA RNNs
        else:
            # To h: [batch_size, seq_len, h_dim] we apply the same mask: [batch_size, 1, h_dim] at every timestep
            shape = torch.Size(h.shape) if self.var_mask else torch.Size([x.shape[0], 1, self.h_dim])
            if x_len is not None:
                h = pack_padded_sequence(h, x_len - 1, batch_first=True)
            h, _ = self.grnn(h)
            if x_len is not None:
                h = pad_packed_sequence(h, batch_first=True, total_length=x.shape[1])[0]
            # We also apply the same dropout mask to every timestep in the output hidden states
            h = self.parameter_dropout_out(h, self.parameter_p, shape=shape)
            scores = self.linear(h)

        return scores

    def sample_sequences(self, x_i, seq_len, eos_token, pad_token, sample_softmax=False):
        """'Sample' sequences from the (learned) decoder given a prefix of tokens.

        Args:
            x_i(torch.Tensor): initial tokens or sequence of tokens to start generating from.
            seq_len(int): length of the sampled sequences after the prefix. Defaults to preset seq_len.
            eos_token(int): the end of sentence indicator.
            pad_token(int): the token used for padding sentences shorter than seq_len.

        Returns:
            list: a list of sampled sequences of pre-defined length.
        """
        if seq_len is not None:
            self.seq_len = seq_len
        else:
            warn("No sequence length provided, preset seq_len will be used.")

        with torch.no_grad():
            if sample_softmax:
                h_i = None
                c_i = None
            else:
                h_i = self._sample_hidden(x_i.shape[0])
                c_i = self._sample_hidden(x_i.shape[0])

            samples = []

            # Sampling pass through the sequential decoder
            # The prefix is automatically consumed by the first step through the RNN
            for i in range(x_i.shape[1]):
                samples.append(x_i[:, i].squeeze().tolist())
            for i in range(self.seq_len):
                x_i = self.emb(x_i)
                if self.rnn_type == "GRU":
                    h, h_i = self.grnn(x_i, h_i)
                else:
                    h, h_i, c_i = self.grnn(x_i, (h_i, c_i))
                # scores: [batch_size, h_dim]
                scores = self.linear(h[:, -1])

                # x_i: [batch_size, 1]
                if sample_softmax:
                    # Sample the output Bernoulli
                    x_i = torch.multinomial(F.softmax(scores, 1), 1)
                else:
                    # Argmax based on stochasticity from hidden
                    x_i = torch.max(scores, dim=1, keepdim=True)[1]
                samples.append(x_i.squeeze().tolist())

        # Pad samples after the first <eos> token
        samples = np.array(samples).T
        eos_spot = np.argwhere(samples == eos_token)
        prev_row = -1
        for spot in eos_spot:
            if spot[0] != prev_row:
                try:
                    samples[spot[0], spot[1]+1:] = pad_token
                except IndexError:
                    pass
            else:
                pass
            prev_row = spot[0]

        return list(samples)

    def _sample_hidden(self, batch_size):
        """Sample the hidden state of a GRU RNN from a standard normal."""
        return torch.normal(mean=torch.zeros((self.l_dim, batch_size, self.h_dim), device=self.device))
