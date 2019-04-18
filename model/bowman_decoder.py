"""
A variational sentence decoder based on 'Generating Sentences from a Continuous Space' by Bowman et al. (2016).

class BowmanDecoder(): pytorch Module that handles the initialization and forward pass through the Bowman model.
class FlowBowmanDecoder(): extends BowmanDecoder(). Handles more expressive latent models (flows).
"""

import numpy as np
import os.path as osp
import sys
from time import time
from warnings import warn
from collections import defaultdict

import torch
import torch.nn as nn
from torch.nn import Parameter
from torch.nn.init import xavier_normal_
import torch.nn.functional as F
from torch.distributions import Normal, Categorical, Bernoulli
from torch.nn.utils.rnn import pack_padded_sequence, PackedSequence, pad_packed_sequence

# Requires S-VAE pytorch extension, at fork: https://github.com/tom-pelsmaeker/s-vae-pytorch
from hyperspherical_vae.distributions import VonMisesFisher

# We include the path of the toplevel package in the system path so we can always use absolute imports within the package.
toplevel_path = osp.abspath(osp.join(osp.dirname(__file__), '..'))
if toplevel_path not in sys.path:
    sys.path.insert(1, toplevel_path)

from util.error import InvalidArgumentError, UnknownArgumentError  # noqa: E402
from model.autoregressive import IAF, Diag, Planar  # noqa: E402
from model.base_decoder import GenerativeDecoder  # noqa: E402
from model.parameter_networks import BowmanEncoder  # noqa: E402


__author__ = "Tom Pelsmaeker"
__copyright__ = "Copyright 2018"


class BowmanDecoder(GenerativeDecoder):
    """A variational auto-encoder for sentences with a single Gaussian latent variable on the sentence level.

    Args:
        device(torch.device): the device (cpu/gpu) on which the model resides.
        seq_len(int): maximum length of sequences passed to the model.
        kl_step(int): step size of linear kl weight increment during training of the model.
        word_p(float): probability of dropping a word, i.e. mapping it to <unk>, before decoding.
        min_rate(float): minimum rate of information from the posterior to the prior.
        unk_index(int): index of the <unk> token in a one-hot representation.
        v_dim(int): size of the vocabulary.
        x_dim(int): size of input embeddings.
        h_dim(int): size of hidden layers of the RNN.
        z_dim(int): size of the latent variable.
    """

    def __init__(self, device, seq_len, kl_step, word_p, word_p_enc, parameter_p, encoder_p, drop_type, min_rate, unk_index, css, sparse, N,
                 rnn_type, tie_in_out, beta, lamb, mmd, ann_mode, rate_mode, posterior, hinge_weight, k, ann_word, word_step, v_dim, x_dim, h_dim, z_dim, s_dim, l_dim, h_dim_enc, l_dim_enc, lagrangian, constraint, max_mmd):
        super(BowmanDecoder, self).__init__(device, seq_len, word_p, word_p_enc, parameter_p, encoder_p, drop_type, min_rate,
                                            unk_index, css, N, rnn_type, kl_step, beta, lamb, mmd, ann_mode, rate_mode, posterior, hinge_weight, ann_word, word_step,
                                            v_dim, x_dim, h_dim, s_dim, z_dim, l_dim, h_dim_enc, l_dim_enc, lagrangian, constraint, max_mmd)

        # The inference model, that is, the model that encodes data into an approximation of latent distribution
        self.encoder = BowmanEncoder(x_dim, h_dim_enc, z_dim, l_dim_enc,
                                     self.encoder_p, self.var_mask, self.posterior, k)

        # The generative model, that is, the model that decodes from the latent space to data, or generates novel data.
        self.emb = nn.Embedding(v_dim, x_dim, sparse=bool(sparse))
        self.ztohidden = nn.Linear(z_dim, h_dim * l_dim)
        self.decoder = nn.GRU(x_dim, h_dim, l_dim, batch_first=True)
        self.linear = nn.Linear(h_dim, v_dim)

        # Tying weights in the input and output layer might increase performance (Inan et al., 2016)
        if tie_in_out:
            if self.h_dim != self.x_dim:
                raise InvalidArgumentError("h_dim should match x_dim when tying weights.")
            self.linear.weight = self.emb.weight

    def forward(self, data, log_likelihood=False, extensive=False):
        """The forward pass encodes sequences into a latent space and decodes it back into a predicted sequence.

        Args:
            data(list of torch.Tensor): a batch of training data, containing at least the sequences, and optionally
                the sequence lengths and a mask for padding different length sequences within a batch.
            log_likelihood(boolean): Whether to use a faster-forward pass for log-likelihood estimation.
            extensive(boolean): Whether to have expanded returns for evaluation outside the class.

        Returns:
            loss(torch.FloatTensor): computed loss, averaged over the batch, summed over the sequence.
            kl(torch.FloatTensor): computed kl, averaged over the batch, summed over dimensions.
            pred(torch.LongTensor): most probable sequences given the data, as predicted by the model.
        """
        x_in, x_len, x_mask = self._unpack_data(data, 3)
        z, z_prior, mu, var = self._encode(x_in, x_len, log_likelihood)
        scores, pred = self._decode(x_in, x_len, z, mu, log_likelihood)
        losses = self._compute_losses(x_in, x_mask, scores, z, z_prior, mu, var, log_likelihood)
        self._update_scale(torch.mean(losses["KL"]) / self.scale.item())
        self._update_word_p()

        if extensive:
            log_q_z_x = self._sample_log_likelihood(z, torch.tensor(
                [1.], device=self.device), 1, mu, var).unsqueeze(0)
            log_p_z = self._prior_log_likelihood(z)

            return losses, pred, var, mu, z, z_prior, log_q_z_x, log_p_z
        else:
            return losses, pred

    def _encode(self, x_in, x_len, log_likelihood):
        """Encodes the input data into Gaussian parameters and samples a latent representation."""
        if self.mmd and not log_likelihood:
            z_prior, _, _ = self._sample_from_prior(x_in.shape[0])
        else:
            z_prior = x_in.new_tensor([[1., 1.]])

        if self.use_prior:
            z, mu, var = self._sample_from_prior(x_in.shape[0])
            if self.posterior == "vmf":
                mu[:, 0] = 1.
        else:
            self.word_dropout.sample_mask(self.word_p_enc, x_in.shape)
            x_dropped = x_in.clone()
            x_dropped[self.word_dropout._mask == 0] = self.unk_index
            z, mu, var = self._sample_from_posterior(x_dropped, x_len, log_likelihood)

        return z, z_prior, mu, var

    def _sample_from_prior(self, num_samples):
        """Obtain a number of sample from the prior."""
        z_prior = self._sample_z(shape=torch.Size([num_samples, self.z_dim]))
        mu = torch.zeros_like(z_prior)
        var = torch.ones_like(z_prior)
        return z_prior, mu, var

    def _sample_from_posterior(self, x_in, x_len, log_likelihood):
        """Obtain samples from the posterior given data tensor."""
        if log_likelihood:
            # The LL estimate is made per datapoint, given as a repeated batch. We save some memory here.
            x = self.emb(x_in[0].unsqueeze(0))
            mu, var = self.encoder(x.detach(), x_len[0].unsqueeze(0))
            mu = mu.expand(x_in.shape[0], *mu.size()[1:])
            var = var.expand(x_in.shape[0], *var.size()[1:])
        else:
            x = self.emb(x_in)
            mu, var = self.encoder(x.detach(), x_len)
        z = self._sample_z(mu, var)
        return z, mu, var

    def _decode(self, x_in, x_len, z, mu, log_likelihood):
        """Decodes the a sampled latent representation of the input back to the input."""
        # Before decoding, we map a fraction of words to <UNK>, weakening the Decoder
        self.word_dropout.sample_mask(self.word_p, x_in.shape)
        x_dropped = x_in.clone()
        x_dropped[self.word_dropout._mask == 0] = self.unk_index
        x = self.emb(x_dropped[:, :-1])

        if self.training or log_likelihood:
            scores = self._compute_scores(x, x_len, x_in, z)
            pred = torch.max(scores.detach(), dim=2)[1] if not log_likelihood else None
        else:
            scores = self._compute_scores(x, x_len, x_in, z)
            pred = torch.max(self._compute_scores(x, x_len, x_in, mu).detach(), dim=2)[1]

        return scores, pred

    def _compute_scores(self, x, x_len, x_in, z):
        """Computes the scores over the vocabulary distribution given data x and latent z."""
        h = self.ztohidden(z)
        h = torch.stack(torch.chunk(h, self.l_dim, 1))

        shape = torch.Size(x.shape) if self.var_mask else torch.Size([x_in.shape[0], 1, self.x_dim])
        x = self.parameter_dropout_in(x, self.parameter_p, shape=shape)
        if x_len is not None:
            x = pack_padded_sequence(x, x_len - 1, batch_first=True)
        h, _ = self.decoder(x, h)
        if x_len is not None:
            h = pad_packed_sequence(h, batch_first=True, total_length=x_in.shape[1] - 1)[0]
        shape = torch.Size(h.shape) if self.var_mask else torch.Size([x_in.shape[0], 1, self.h_dim])
        h = self.parameter_dropout_out(h, self.parameter_p, shape=shape)
        scores = self.linear(h)

        return scores

    def _compute_losses(self, x_in, x_mask, scores, z, z_prior, mu, var, log_likelihood):
        """Computes the various VAE losses given scores and latent variables."""
        losses = defaultdict(lambda: torch.tensor(0., device=self.device))
        self._compute_rec_loss(scores, x_in, x_mask, log_likelihood, losses)
        if not self.use_prior:
            self._compute_reg_loss(z, mu, var, log_likelihood, losses)
            self._compute_aux_loss(z, z_prior, log_likelihood, losses)
        return losses

    def _compute_rec_loss(self, scores, x_in, x_mask, log_likelihood, losses):
        """Compute the reconstruction term in the loss"""
        if self.css and self.training:
            losses["NLL"] = torch.mean(self._css(scores, x_in[:, 1:]) * x_mask[:, 1:], 0).sum()
        else:
            if log_likelihood:
                # We return the non-mean NLL when we want to estimate the likelihood
                losses["NLL"] = torch.sum(self.reconstruction_loss(scores.contiguous().view(
                    [-1, scores.shape[2]]), x_in[:, 1:].contiguous().view([-1])).view(scores.shape[0], scores.shape[1]) * x_mask[:, 1:], 1)
            else:
                losses["NLL"] = torch.mean(self.reconstruction_loss(scores.contiguous().view(
                    [-1, scores.shape[2]]), x_in[:, 1:].contiguous().view([-1])).view(scores.shape[0], scores.shape[1]) * x_mask[:, 1:], 0).sum()

    def _compute_reg_loss(self, z, mu, var, log_likelihood, losses):
        """Compute the regularization term in the loss."""
        if log_likelihood:
            # When we use a multisample estimate of the ELBO we lose access to the analytical KL
            losses["KL"] = self._sample_log_likelihood(z, torch.tensor(
                [1.], device=self.device), 1, mu, var) - self._prior_log_likelihood(z)
        else:
            losses["KL"] = torch.mean(self.scale.item() * self._kl_divergence(mu, var,
                                                                              None, None, torch.tensor([1.], device=self.device), 1))

    def _prior_log_likelihood(self, z):
        """Evaluate the log-likelihood of a sample given the prior."""
        return self._sample_log_likelihood(z, torch.tensor([1.], device=self.device), 1)

    def _compute_aux_loss(self, z, z_prior, log_likelihood, losses):
        """Compute auxiliary parts of the loss."""
        if not log_likelihood:
            # The lagrangian flag determines whether we apply Lagrangian opt to constraints
            if self.lagrangian:
                # We compute the mmd if necessary for the constraint
                if 'mmd' in self.constraint:
                    mmd = self._mmd(z, z_prior)
                else:
                    mmd = None
                losses["Constraint"] = self._compute_constraints(
                    losses['KL'] / self.scale.item(), mmd)
                losses["Lag_Weight"] = self.lag_weight.sum()
            # These are constraints with pre-set weights, invoked when self.lagrangian is False
            else:
                losses["Hinge"] = self._hinge_loss(losses["KL"] / self.scale.item(), self.min_rate)
                if self.rate_mode == "fb" and self.training:
                    losses["KL"] = losses["Hinge"]
                    losses["Hinge"] = torch.tensor(0., device=self.device)
                if self.mmd:
                    losses["MMD"] = (self.alpha + self.lamb - 1) * self._mmd(z, z_prior)

            losses["L2"] = self._l2_regularization()

    def sample_sequences(self, x_i, seq_len, eos_token, pad_token, sample_softmax=False):
        """Samples sequences from the (learned) decoder given a prefix of tokens.

        Args:
            x_i(torch.Tensor): initial tokens or sequence of tokens to start generating from.
            seq_len(int): length of the sampled sequences.
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
            # When generating, we sample z from the prior N(0, I)
            z, _, _ = self._sample_from_prior(x_i.shape[0])
            samples = self._sample_from_z(x_i, z, sample_softmax)

        return self._pad_samples(samples, eos_token, pad_token)

    def sample_posterior(self, seq, num, eos_token, pad_token, x_i=None, mode="teacher", seq_len=None, sample_softmax=False):
        """Guided posterior sampling."""
        if seq_len is None:
            self.seq_len = seq.shape[1] + 5
        else:
            self.seq_len = seq_len

        with torch.no_grad():
            sequences = seq.expand(num, -1)
            x_len = sequences.new_tensor([sequences.shape[1]]).expand(num)
            z = self._sample_from_posterior(sequences, x_len, False)[0]
            if mode == "teacher":
                _, samples = self._decode(sequences, x_len, z, z, False)
                samples = samples.squeeze().transpose(0, 1).tolist()
            elif mode == "free":
                samples = self._sample_from_z(x_i, z, sample_softmax)

        return self._pad_samples(samples, eos_token, pad_token)

    def homotopies(self, seq_1, seq_2, num, x_i, eos_token, pad_token, seq_len=None, sample_softmax=False):
        """Computes num homotopies between two sequences."""
        if seq_len is None:
            self.seq_len = max(seq_1.shape[1], seq_2.shape[1]) + 5
        else:
            self.seq_len = seq_len

        with torch.no_grad():
            # Embed sequences
            z_1 = self._sample_from_posterior(seq_1, seq_1.new_tensor([seq_1.shape[1]]), False)[0]
            z_2 = self._sample_from_posterior(seq_2, seq_2.new_tensor([seq_2.shape[1]]), False)[0]

            # Get in-between z using linear homeotopy
            z = torch.cat([(1. - i / num) * z_1 + i / num * z_2 for i in range(num + 1)], 0)
            samples = self._sample_from_z(x_i, z, sample_softmax)

        return self._pad_samples(samples, eos_token, pad_token)

    def _pad_samples(self, samples, eos_token, pad_token):
        "Pads a batch of samples after the eos token."
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

    def _sample_from_z(self, x_i, z, sample_softmax):
        """Sample sequences given a sampled z and prefix x_i."""
        h_i = self.ztohidden(z)
        h_i = torch.stack(torch.chunk(h_i, self.l_dim, 1))
        samples = []

        # Sampling pass through the sequential decoder
        # The prefix is automatically consumed by the first step through the RNN
        for i in range(x_i.shape[1]):
            samples.append(x_i[:, i].squeeze().tolist())
        for i in range(self.seq_len):
            x_i = self.emb(x_i)
            # h: [batch_size, prefix_len, h_dim]
            h, h_i = self.decoder(x_i, h_i)
            # scores: [batch_size, v_dim]
            scores = self.linear(h[:, -1])

            if sample_softmax:
                x_i = torch.multinomial(F.softmax(scores, 1), 1)
            else:
                # Here we argmax, so any stochasticity stems from the latent variable
                # x_i: [batch_size, 1]
                x_i = torch.max(scores, dim=1, keepdim=True)[1]
            samples.append(x_i.squeeze().tolist())
        return samples

    def _l2_regularization(self):
        """Computes the l2 regularization term according to the Gal dropout paper with t=1."""
        l2_loss = torch.tensor(0., device=self.device)
        if self.training:
            for parameter in self.named_parameters():
                if "emb" in parameter[0]:
                    continue

                # The decoder and encoder have seperate dropout
                if "encoder" in parameter[0]:
                    if "bias" in parameter[0]:
                        # We assume a Gaussian prior with unit variance around the bias values (Gal, 2016)
                        l2_loss += 1. / (2 * self.N) * (parameter[1] ** 2).sum()
                    elif "hh" in parameter[0]:
                        # For non-dropped weights we also assume a Gaussian prior with unit variance
                        if self.drop_type == "recurrent":
                            l2_loss += (1. - self.encoder_p) / (2 * self.N) * (parameter[1] ** 2).sum()
                        else:
                            l2_loss += 1. / (2 * self.N) * (parameter[1] ** 2).sum()
                    else:
                        l2_loss += (1. - self.encoder_p) / (2 * self.N) * (parameter[1] ** 2).sum()
                else:
                    if "bias" in parameter[0]:
                        # We assume a Gaussian prior with unit variance around the bias values (Gal, 2016)
                        l2_loss += 1. / (2 * self.N) * (parameter[1] ** 2).sum()
                    elif "hh" in parameter[0]:
                        # For non-dropped weights we also assume a Gaussian prior with unit variance
                        if self.drop_type == "recurrent":
                            l2_loss += (1. - self.parameter_p) / (2 * self.N) * (parameter[1] ** 2).sum()
                        else:
                            l2_loss += 1. / (2 * self.N) * (parameter[1] ** 2).sum()
                    else:
                        # For dropped weights we have to scale the regularization because of Bernoulli prior (Gal, 2016)
                        l2_loss += (1. - self.parameter_p) / (2 * self.N) * (parameter[1] ** 2).sum()
        return l2_loss


class FlowBowmanDecoder(BowmanDecoder):

    def __init__(self, device, seq_len, kl_step, word_p, word_p_enc, parameter_p, encoder_p, drop_type, min_rate, unk_index, css, sparse, N,
                 rnn_type, tie_in_out, beta, lamb, mmd, ann_mode, rate_mode, posterior, hinge_weight, k, ann_word, word_step,
                 flow, flow_depth, hidden_depth, prior, num_weights, mean_len, std_len,
                 v_dim, x_dim, h_dim, z_dim, s_dim, l_dim, h_dim_enc, l_dim_enc, c_dim, lagrangian, constraint, max_mmd):
        super(FlowBowmanDecoder, self).__init__(device, seq_len, kl_step, word_p, word_p_enc, parameter_p, encoder_p, drop_type,
                                                min_rate, unk_index, css, sparse, N, rnn_type, tie_in_out, beta, lamb,
                                                mmd, ann_mode, rate_mode, posterior, hinge_weight, k, ann_word, word_step, v_dim, x_dim, h_dim, z_dim, s_dim, l_dim,
                                                h_dim_enc, l_dim_enc, lagrangian, constraint, max_mmd)

        if flow == "diag":
            self.flow = Diag()
        elif flow == "iaf":
            self.flow = IAF(c_dim, z_dim, flow_depth, hidden_depth)
        elif flow == "vpiaf":
            self.flow = IAF(c_dim, z_dim, flow_depth, hidden_depth, scale=False)
        elif flow == 'planar':
            self.flow = Planar(h_dim_enc * 2, z_dim, hidden_depth)
        else:
            raise UnknownArgumentError(
                "Flow type not recognized: {}. Please choose [diag, iaf, vpiaf, planar]".format(flow))

        if prior in ["mog", "vamp", "weak"]:
            self.prior = prior
        else:
            raise notImplementedError("No implementation for prior: {}".format(prior))

        if self.prior == "mog":
            # Create learnable mixture parameters with fixed weights and initialize
            mixture_weights = torch.Tensor(num_weights).fill_(1./num_weights)
            self.register_buffer("mixture_weights", mixture_weights)
            self.mixture_mu = Parameter(torch.Tensor(num_weights, self.z_dim))
            if self.posterior == "vmf":
                self.mixture_var = Parameter(torch.Tensor(num_weights, 1))
            else:
                self.mixture_var = Parameter(torch.Tensor(num_weights, self.z_dim))
            xavier_normal_(self.mixture_var)
            xavier_normal_(self.mixture_mu)
        elif self.prior == "vamp":
            # Create learnable pseudoinputs with fixed weights and initialize
            pseudo_weights = torch.Tensor(num_weights).fill_(1./num_weights)
            self.register_buffer("pseudo_weights", pseudo_weights)
            self.pseudo_inputs = Parameter(torch.Tensor(num_weights, seq_len, x_dim))
            xavier_normal_(self.pseudo_inputs)

            # Sample lengths for the pseudo inputs based on database statistics
            pre_lengths = Normal(mean_len, std_len).sample(torch.Size([num_weights]))
            lengths = torch.clamp(torch.round(pre_lengths), 1, seq_len).long()
            lengths = torch.sort(lengths, descending=True)[0]
            self.register_buffer("pseudo_lengths", lengths)

        # layer to context
        if 'iaf' in flow:
            self.h_to_context = nn.Linear(h_dim_enc * l_dim_enc * 2, c_dim)
        else:
            self.h_to_context = nn.Sequential()

    def forward(self, data, log_likelihood=False, extensive=False):
        """The forward pass encodes sequences into a latent space and decodes it back into a predicted sequence.

        Args:
            data(list of torch.Tensor): a batch of training data, containing at least the sequences, and optionally
                the sequence lengths and a mask for padding different length sequences within a batch.
            log_likelihood(boolean): Whether to use a faster-forward pass for log-likelihood estimation.
            extensive(boolean): Whether to have expanded returns for evaluation outside the class.

        Returns:
            loss(torch.FloatTensor): computed loss, averaged over the batch, summed over the sequence.
            kl(torch.FloatTensor): computed kl, averaged over the batch, summed over dimensions.
            pred(torch.LongTensor): most probable sequences given the data, as predicted by the model.
        """
        x_in, x_len, x_mask = self._unpack_data(data, 3)
        z_0, z, z_prior, z_mu, mu, var, logdet = self._encode(x_in, x_len, log_likelihood)
        scores, pred = self._decode(x_in, x_len, z, z_mu, log_likelihood)
        losses = self._compute_losses(x_in, x_mask, scores, z_0, z, z_prior, mu, var, logdet, log_likelihood)
        self._update_scale(torch.mean(losses["KL"]) / self.scale.item())
        self._update_word_p()

        if extensive:
            log_q_z_x = self._sample_log_likelihood(z, torch.tensor(
                [1.], device=self.device), 1, mu, var).unsqueeze(0) - logdet
            log_p_z = self._prior_log_likelihood(z)

            return losses, pred, var, mu, z, z_prior, log_q_z_x, log_p_z
        else:
            return losses, pred

    def _encode(self, x_in, x_len, log_likelihood):
        """Encode into parameters of a diag Gaussian and sample z: [batch_size x z_dim]. Then apply flow."""
        if self.mmd:
            z_prior, _, _ = self._sample_from_prior(x_in.shape[0])
        else:
            z_prior = x_in.new_tensor([[1.]])

        if self.use_prior:
            z, mu, var = self._sample_from_prior(x_in.shape[0])
            z_mu = mu
            z_0 = None
            logdet = None
        else:
            self.word_dropout.sample_mask(self.word_p_enc, x_in.shape)
            x_dropped = x_in.clone()
            x_dropped[self.word_dropout._mask == 0] = self.unk_index
            z, mu, var, logdet, z_0, z_mu = self._sample_from_posterior(x_dropped, x_len, log_likelihood)

        return z_0, z, z_prior, z_mu, mu, var, logdet

    def _sample_from_posterior(self, x_in, x_len, log_likelihood):
        """Obtain samples from the posterior given a data tensor."""
        if log_likelihood:
            # The LL estimate is made per datapoint, given as a repeated batch. We save some memory here.
            x = self.emb(x_in[0].unsqueeze(0))
            mu, var, h = self.encoder(x.detach(), x_len[0].unsqueeze(0), True)
            h = self.h_to_context(h)
            h = h.expand(x_in.shape[0], *h.size()[1:])
            mu = mu.expand(x_in.shape[0], *mu.size()[1:])
            var = var.expand(x_in.shape[0], *var.size()[1:])
        else:
            x = self.emb(x_in)
            mu, var, h = self.encoder(x.detach(), x_len, True)
            h = self.h_to_context(h)
        z_0 = self._sample_z(mu, var)
        z, logdet = self.flow(z_0, h)
        if not self.training:
            z_mu, _ = self.flow(mu, h)
        else:
            z_mu = None
        return z, mu, var, logdet, z_0, z_mu

    def _sample_from_prior(self, num_samples):
        """Obtain a number of samples from the prior."""
        if self.prior == "weak":
            # We sample from a uniform prior
            z_prior = self._sample_z(shape=torch.Size([num_samples, self.z_dim]))
            mu = torch.zeros_like(z_prior)
            var = torch.ones_like(z_prior)
            if self.posterior == 'vmf':
                mu[:, 0] = 1.
        elif self.prior == "mog":
            # We first sample num_samples modes of the mixture prior, and then sample each mode
            # [num, ]
            k = Categorical(logits=F.log_softmax(self.mixture_weights)).sample(
                torch.Size([num_samples])).long()
            mu, var = self._get_mixture_parameters()

            # [num, z_dim]
            mu = torch.index_select(mu, 0, k)
            var = torch.index_select(var, 0, k)
            z_prior = self._sample_z(mu, var)
        elif self.prior == "vamp":
            # We first sample num_samples pseudoinputs, and then return their posterior samples
            # [num, ]
            k = Categorical(logits=F.log_softmax(self.pseudo_weights)).sample(
                torch.Size([num_samples])).long()
            k = torch.sort(k, descending=False)[0]
            # [num, seq_len, x_dim]
            x = torch.index_select(self.pseudo_inputs, 0, k)
            # [num, 1]
            x_len = torch.index_select(self.pseudo_lengths, 0, k)

            mu, var = self.encoder(x, x_len)
            z_prior = self._sample_z(mu, var)

        return z_prior, mu, var

    def _compute_losses(self, x_in, x_mask, scores, z_0, z, z_prior, mu, var, logdet, log_likelihood):
        """Computes the various VAE losses given scores and latent variables."""
        losses = defaultdict(lambda: torch.tensor(0., device=self.device))
        self._compute_rec_loss(scores, x_in, x_mask, log_likelihood, losses)
        if not self.use_prior:
            self._compute_reg_loss(z_0, z, mu, var, logdet, log_likelihood, losses)
            self._compute_aux_loss(z, z_prior, log_likelihood, losses)
        return losses

    def _compute_reg_loss(self, z_0, z, mu, var, logdet, log_likelihood, losses):
        entropy = self._sample_log_likelihood(z_0, torch.tensor(
            [1.], device=self.device), dim=1, mu=mu, var=var)
        nll_prior = -self._prior_log_likelihood(z)
        losses["KL"] = entropy + nll_prior - logdet
        if not log_likelihood:
            losses["KL"] = self.scale.item() * torch.mean(losses["KL"])

    def _prior_log_likelihood(self, z):
        if self.prior == "weak":
            return self._sample_log_likelihood(z, torch.tensor([1.], device=self.device), 1)
        elif self.prior == "mog":
            mu, var = self._get_mixture_parameters()
            log_k = F.log_softmax(self.mixture_weights)
            # [batch_size, num_mixture]
            mixture_log = self._sample_log_likelihood(z.unsqueeze(
                1), torch.tensor([[1.]], device=self.device), dim=2, mu=mu, var=var)
            # [batch_size]
            return torch.logsumexp(mixture_log + log_k, dim=1)
        elif self.prior == "vamp":
            # [num_pseudo, z_dim]
            mu, var = self.encoder(self.pseudo_inputs, self.pseudo_lengths)
            # [num_pseudo, ]
            log_k = F.log_softmax(self.pseudo_weights)

            # [batch_size, num_pseudo]
            pseudo_log = self._sample_log_likelihood(z.unsqueeze(1), torch.tensor(
                [[1.]], device=self.device), dim=2, mu=mu, var=var)
            # [batch_size, ]
            return torch.logsumexp(pseudo_log + log_k, dim=1)

    def _get_mixture_parameters(self):
        if self.posterior == "vmf":
            mu = self.mixture_mu / self.mixture_mu.norm(2, dim=-1, keepdim=True)
            var = torch.tanh(self.mixture_var) * ((self.z_dim * 3.5 - self.z_dim / 3.5) / 2.) + \
                ((self.z_dim * 3.5 + self.z_dim / 3.5) / 2)
        else:
            mu = self.mixture_mu
            var = F.softplus(self.mixture_var)
        return mu, var
