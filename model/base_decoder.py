"""
Base recurrent decoder classes.

class BaseDecoder(): A template for RNN-type models for language generation (decoding).
class GenerativeDecoder(): Extends BaseDecoder(). A template for stochastic RNN-type models for language generation.
"""

import numpy as np
import sys
import os.path as osp
from warnings import warn

import torch
from torch import nn
from torch.nn import Parameter
from torch.distributions import Normal, kl_divergence
from torch_two_sample import MMDStatistic
# Requires S-VAE pytorch extension https://github.com/tom-pelsmaeker/s-vae-pytorch
from hyperspherical_vae.distributions import VonMisesFisher
from hyperspherical_vae.distributions import HypersphericalUniform

# We include the path of the toplevel package in the system path so we can always use absolute imports within the package.
toplevel_path = osp.abspath(osp.join(osp.dirname(__file__), '..'))
if toplevel_path not in sys.path:
    sys.path.insert(1, toplevel_path)

from model.dropout import FlexibleDropout  # noqa: E402
from util.error import InvalidArgumentError, UnknownArgumentError  # noqa: E402

__author__ = "Tom Pelsmaeker"
__copyright__ = "Copyright 2020"


class BaseDecoder(nn.Module):
    """
    A basic template that defines some methods and properties shared by all RNN-type language models.
    """

    def __init__(self, device, seq_len, word_p, parameter_p,
                 drop_type, unk_index, css, N, rnn_type, v_dim, x_dim, h_dim, s_dim, l_dim):
        super(BaseDecoder, self).__init__()

        self.device = device
        self.seq_len = seq_len
        self.min_word_p = torch.tensor(word_p, device=device, dtype=torch.float)
        self.word_p = torch.tensor(word_p, device=device, dtype=torch.float)
        self.parameter_p = torch.tensor(parameter_p, device=device, dtype=torch.float)
        self.drop_type = drop_type
        self.unk_index = unk_index
        self.css = css
        self.N = N
        self.rnn_type = rnn_type
        self.v_dim = v_dim
        self.x_dim = x_dim
        self.h_dim = h_dim
        self.s_dim = s_dim
        self.l_dim = l_dim
        self._scale = torch.tensor(1.0, device=self.device)

        # When the drop type is varied, we vary the mask
        if self.drop_type == "varied":
            self.var_mask = True
        else:
            self.var_mask = False

        # Base classifier loss for RNN-type language models
        self.reconstruction_loss = nn.CrossEntropyLoss(reduction='none')

        # Dropout for words, as in Bowman (2015)
        self.word_dropout = FlexibleDropout()

        # Parameter dropout drops rows of weight matrices
        self.parameter_dropout_in = FlexibleDropout()
        if self.drop_type == "recurrent":
            # Given recurrent dropout we use dropout in between layers
            self.parameter_dropout_out = [FlexibleDropout() for _ in range(l_dim)]
            self.parameter_dropout_hidden = [FlexibleDropout() for _ in range(l_dim)]
            if self.rnn_type == "LSTM":
                # LSTMs also need dropout for the context vectors
                self.parameter_dropout_context = [FlexibleDropout() for _ in range(l_dim)]
        else:
            self.parameter_dropout_out = FlexibleDropout()

    @property
    def word_p(self):
        return self._word_p

    @word_p.setter
    def word_p(self, val):
        if isinstance(val, float):
            val = self.beta.new_tensor(val)

        if not isinstance(val, torch.FloatTensor) and not isinstance(val, torch.cuda.FloatTensor):
            raise InvalidArgumentError("word_p should be a float or FloatTensor, not {}.".format(type(val)))

        if val > self.min_word_p and val <= 1.:
            self._word_p = val
        elif val > 1.:
            self._word_p = self.min_word_p.new_tensor(1.)
        elif val <= self.min_word_p:
            self._word_p = self.min_word_p.new_tensor(self.min_word_p.item())

    @property
    def seq_len(self):
        return self._seq_len

    @seq_len.setter
    def seq_len(self, val):
        if not isinstance(val, int):
            raise InvalidArgumentError("seq_len should be an integer.")
        self._seq_len = val

    @property
    def scale(self):
        return self._scale

    @scale.setter
    def scale(self, val):
        pass

    @property
    def drop_type(self):
        return self._drop_type

    @drop_type.setter
    def drop_type(self, value):
        if value not in ["varied", "shared", "recurrent"]:
            raise UnknownArgumentError(
                "Unknown drop_type: {}. Please choose [varied, shared, recurrent]".format(value))
        self._drop_type = value

    @property
    def rnn_type(self):
        return self._rnn_type

    @rnn_type.setter
    def rnn_type(self, value):
        if value not in ["LSTM", "GRU"]:
            raise UnknownArgumentError("Unknown rnn_type {}. Please choose [GRU, LSTM]".format(value))
        self._rnn_type = value

    def se_pass(self, *argv):
        """Implement the capacity to return embeddings for SentEval into each new model."""
        raise NotImplementedError()

    def sample_sequences(self, *argv):
        raise NotImplementedError()

    def _unpack_data(self, data, N):
        """Unpacks the input data to the forward pass and supplies missing data.

        Args:
            data(list of torch.Tensor): data provided to forward pass. We assume the following ordering
                [input, length(optional), mask(optional), reversed(optional), reversed_length(optional)]
            N(int): the number of data tensors to return. Can be 1-4.

        Returns:
            x_in(torch.Tensor): batch of input sequences.
            x_len(torch.Tensor): lengths of input sequences or None.
            x_mask(torch.Tensor): mask over the padding of input sequences that are not of max length or None.
            x_reverse(torch.Tensor): batch of reversed input sequences or None.
        """
        # Checks and padding of data, so we have N tensors or None to process
        if not isinstance(data[0], torch.Tensor):
            raise InvalidArgumentError("Data should contain a torch Tensor with data at the first position.")
        if N < 1 or N > 4:
            raise InvalidArgumentError("N should be between 1 and 4.")
        data = (data + [None, ] * N)[:N]
        for d in data:
            if not isinstance(d, torch.Tensor) and d is not None:
                raise InvalidArgumentError("Data should contain only torch Tensors or None.")

        # If no mask is given, we create an empty mask as placeholder.
        if N > 2 and data[2] is None:
            data[2] = torch.ones(data[0].shape).to(self.device)
            if data[1] is not None:
                warn("Data length is given without mask. Assuming all sentences are of the same length. Sentences shorter than {} words will not be masked.".format(self.seq_len))

        # When the reversed data is not given, we assume no padding and reverse the sequence ourselves
        if N > 3 and data[3] is None:
            warn("Reversed data not provided. We assume no padding and reverse the data cheaply.")
            indices = torch.arange(data[0].shape[1] - 1, -1, -1)
            data[3] = data[0].index_select(1, indices)

        return data

    def _css(self, scores, targets):
        """
        Computes the negative log-likelihood with a CSS approximate of the Softmax.

        Args:
            scores(torch.FloatTensor): [B x S-1 x V]
            targets(torch.LongTensor): [B x S-1]

        Returns:
            torch.FloatTensor: negative log-likelihood. [B x S-1]
        """
        # Obtain the positive and negative set, both parts of the support
        positive_set = targets.unique()
        neg_dim = self.v_dim - positive_set.shape[0]
        weights = np.ones(self.v_dim) / neg_dim
        weights[positive_set] = 0
        negative_set = torch.tensor(np.random.choice(self.v_dim, self.s_dim,
                                                     replace=False, p=weights)).to(self.device)

        # Extract the scores of the support, normalizing the negative set in the process
        log_kappa = torch.log(torch.tensor(neg_dim / self.s_dim, device=self.device))
        scores[:, :, negative_set] += log_kappa
        support = scores[:, :, torch.cat((positive_set, negative_set))]

        # The softmax stabilizer
        u = torch.max(support, dim=2)[0]

        # Compute the log of stable exponentials. We also need to shift the scores.
        log_support = torch.log(torch.exp(support - u.unsqueeze(2)).sum(dim=2))
        log_scores = torch.log(torch.exp(torch.gather(scores, 2, targets.unsqueeze(2)) - u.unsqueeze(2))).squeeze(2)

        # We return the negative log likelihood
        return -(log_scores - log_support)

    def _init_hidden(self, batch_size):
        """Initialize the hidden state of a GRU RNN."""
        h = torch.zeros((self.l_dim, batch_size, int(self.h_dim/self.l_dim))).to(self.device)
        return h

    def _l2_regularization(self):
        """Computes the l2 regularization term according to the Gal dropout paper with t=1."""
        l2_loss = torch.tensor(0., device=self.device)
        if self.training:
            for parameter in self.named_parameters():
                if "emb" in parameter[0]:
                    continue

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


class GenerativeDecoder(BaseDecoder):
    """
    A basic template that defines some methods and properties shared by all generative RNN-type language models.
    """

    def __init__(self, device, seq_len, word_p, word_p_enc, parameter_p, encoder_p, drop_type, min_rate, unk_index,
                 css, N, rnn_type, kl_step, beta, lamb, mmd, ann_mode, rate_mode, posterior, hinge_weight, ann_word,
                 word_step, v_dim, x_dim, h_dim, s_dim, z_dim, l_dim, h_dim_enc, l_dim_enc, lagrangian, constraint,
                 max_mmd, max_elbo, alpha):
        super(GenerativeDecoder, self).__init__(device, seq_len, word_p, parameter_p, drop_type, unk_index, css, N,
                                                rnn_type, v_dim, x_dim, h_dim, s_dim, l_dim)
        # Recurrent dropout was never implemented for the VAE's because it doesn't work well
        if self.drop_type == "recurrent":
            raise InvalidArgumentError(
                "Recurrent dropout not implemented for this model. Please choose ['varied', 'shared']")
        # LSTM's are not supported because GRU's work equally well (with less parameters)
        if self.rnn_type == "LSTM":
            raise InvalidArgumentError("LSTM not implemented for this model. Please choose ['GRU']")

        # Choose between the vMF-autoencoder and Gauss-autoencoder
        self.posterior = posterior

        # Encoder architecture settings
        self.encoder_p = torch.tensor(encoder_p, device=self.device, dtype=torch.float)
        self.h_dim_enc = h_dim_enc
        self.l_dim_enc = l_dim_enc
        self.word_p_enc = word_p_enc

        # Optimization hyperparameters
        self.min_rate = torch.tensor(min_rate, device=self.device, dtype=torch.float)  # minimum Rate of hinge/FB
        self.beta = torch.tensor(beta, device=self.device, dtype=torch.float)  # beta value of beta-VAE
        self.alpha = torch.tensor(alpha, device=self.device, dtype=torch.float)  # alpha value of InfoVAE
        self.lamb = torch.tensor(lamb, device=self.device, dtype=torch.float)  # lambda value of InfoVAE
        self.kl_step = torch.tensor(kl_step, device=self.device, dtype=torch.float)  # Step size of KL annealing
        self.hinge_weight = torch.tensor(hinge_weight, device=self.device, dtype=torch.float)  # Weight of hinge loss
        # Step size of word dropout annealing
        self.word_step = torch.tensor(word_step, device=self.device, dtype=torch.float)
        self.max_mmd = torch.tensor(max_mmd, device=self.device, dtype=torch.float)  # Maximum MMD
        self.max_elbo = torch.tensor(max_elbo, device=self.device, dtype=torch.float)  # Maximum ELBO

        #  Optimization modes
        self.mmd = mmd  # When true, we add the maximum mean discrepancy to the loss, and optimize the InfoVAE
        self.ann_mode = ann_mode  # The mode of annealing
        self.rate_mode = rate_mode  # How to force the VAE to encode a minimum rate
        self.ann_word = ann_word  # Whether to anneal word dropout

        # The weight of the constraint in the Lagrangian dual function
        # Hardcoded start at 1.01
        self.lagrangian = lagrangian
        self.constraint = constraint
        self.lag_weight = Parameter(torch.tensor([1.01] * len(self.constraint)))

        if self.ann_word:
            self.word_p = 1.

        self.z_dim = z_dim

        # We start the scale factor at zero, to be incremented linearly with kl_step every forward pass
        if self.ann_mode == "linear":
            self.scale = torch.tensor(self.kl_step.item() * self.beta.item(), device=self.device)
        # Or we start the scale at 10%, to be increased or decreased in 10% increments based on a desired rate
        elif self.ann_mode == "sfb":
            self.scale = torch.tensor(0.1 * self.beta.item(), device=self.device)

        # This switch should be manually managed from training/testing scripts to select generating from prior/posterior
        self.use_prior = False

        # N(0, I) error distribution to sample from latent spaces with reparameterized gradient
        self.error = Normal(torch.tensor(0., device=device),
                            torch.tensor(1., device=device))

    @property
    def constraint(self):
        return self._constraint

    @constraint.setter
    def constraint(self, vals):
        if type(vals) != list:
            if isinstance(vals, str):
                vals = [vals]
            else:
                raise InvalidArgumentError('constraint should be a list or str')
        for val in vals:
            if val not in ['mdr', 'mmd', 'elbo']:
                raise UnknownArgumentError(
                    'constraint {} unknown. Please choose [mdr, mmd].'.format(val))

        self._constraint = vals

    @property
    def min_rate(self):
        return self._min_rate

    @min_rate.setter
    def min_rate(self, val):
        if isinstance(val, float):
            val = torch.tensor(val, device=self.device)

        if not isinstance(val, torch.FloatTensor) and not isinstance(val, torch.cuda.FloatTensor):
            raise InvalidArgumentError("min_rate should be a float or FloatTensor.")

        if val > 0:
            self._min_rate = val
        else:
            self._min_rate = torch.tensor(0., device=self.device)

    @property
    def ann_mode(self):
        return self._ann_mode

    @ann_mode.setter
    def ann_mode(self, value):
        if value not in ["linear", "sfb"]:
            raise UnknownArgumentError("Unknown ann_mode {}. Please choose [linear, sfb]".format(value))
        self._ann_mode = value

    @property
    def scale(self):
        """The current scale factor of the KL-divergence."""
        if self.training:
            return self._scale
        else:
            # We always want the full KL when testing
            return torch.tensor(1.0, device=self.device)

    @scale.setter
    def scale(self, val):
        if not isinstance(val, float) and not isinstance(val, torch.FloatTensor) and not \
                isinstance(val, torch.cuda.FloatTensor):
            raise InvalidArgumentError("scale should be a float.")

        if isinstance(val, float):
            val = self.beta.new_tensor(val)

        if self.training:
            if val <= self.beta and val >= 0.:
                self._scale = val
            elif val > self.beta and self._scale < self.beta:
                self._scale = torch.tensor(self.beta.item(), device=self.device)
            elif val < 0.:
                raise InvalidArgumentError("scale should be positive.")

            if val < 0.0001:
                self._scale = torch.tensor(0.0001, device=self.device)

    @property
    def use_prior(self):
        """This boolean switch determines whether to sample from the prior or posterior when computing results."""
        return self._use_prior

    @use_prior.setter
    def use_prior(self, val):
        if not isinstance(val, bool):
            if not isinstance(val, int):
                raise InvalidArgumentError("use_prior should be a boolean switch.")
            elif val != 0 and val != 1:
                raise InvalidArgumentError("Only 0 or 1 can be interpreted as a boolean.")
            else:
                self._use_prior = bool(val)
        else:
            self._use_prior = val

    @property
    def posterior(self):
        return self._posterior

    @posterior.setter
    def posterior(self, val):
        if val not in ["gaussian", "vmf"]:
            return UnknownArgumentError("Unknown posterior: {}. Please choose [gaussian, vmf].".format(val))
        self._posterior = val

    def _sample_z(self, mu=None, var=None, shape=None, det=False):
        if self.posterior == "gaussian":
            return self._gaussian_sample_z(mu, var, shape, det)
        elif self.posterior == "vmf":
            return self._vmf_sample_z(mu, var, shape, det)

    def _kl_divergence(self, mu, var, mu_2, var_2, mask, dim):
        if self.posterior == "gaussian":
            return self._gaussian_kl_divergence(mu, var, mu_2, var_2, mask, dim)
        elif self.posterior == "vmf":
            return self._vmf_kl_divergence(mu, var)

    def _sample_log_likelihood(self, sample, mask, dim, mu=None, var=None):
        if self.posterior == "gaussian":
            return self._gaussian_log_likelihood(sample, mask, dim, mu, var)
        elif self.posterior == "vmf":
            return self._vmf_log_likelihood(sample, mu, var)

    def _vmf_sample_z(self, location, kappa, shape, det):
        """Reparameterized sample from a vMF distribution with location and concentration kappa."""
        if location is None and kappa is None and shape is not None:
            if det:
                raise InvalidArgumentError("Cannot deterministically sample from the Uniform on a Hypersphere.")
            else:
                return HypersphericalUniform(self.z_dim - 1, device=self.device).sample(shape[:-1])
        elif location is not None and kappa is not None:
            if det:
                return location
            if self.training:
                return VonMisesFisher(location, kappa).rsample()
            else:
                return VonMisesFisher(location, kappa).sample()
        else:
            raise InvalidArgumentError("Either provide location and kappa or neither with a shape.")

    def _vmf_kl_divergence(self, location, kappa):
        """Get the estimated KL between the VMF function with a uniform hyperspherical prior."""
        return kl_divergence(VonMisesFisher(location, kappa), HypersphericalUniform(self.z_dim - 1, device=self.device))

    def _vmf_log_likelihood(self, sample, location=None, kappa=None):
        """Get the log likelihood of a sample under the vMF distribution with location and kappa."""
        if location is None and kappa is None:
            return HypersphericalUniform(self.z_dim - 1, device=self.device).log_prob(sample)
        elif location is not None and kappa is not None:
            return VonMisesFisher(location, kappa).log_prob(sample)
        else:
            raise InvalidArgumentError("Provide either location and kappa or neither.")

    def _gaussian_sample_z(self, mu, var, shape, det):
        """Sample from a Gaussian distribution with mean mu and variance var."""
        if mu is None and var is None and shape is not None:
            if det:
                return torch.zeros(shape, device=self.device)
            else:
                return self.error.sample(shape)
        elif mu is not None and var is not None:
            if det:
                return mu
            else:
                return mu + torch.sqrt(var) * self.error.sample(var.shape)
        else:
            raise InvalidArgumentError("Provide either mu and var or neither with a shape.")

    def _gaussian_kl_divergence(self, mu_1, var_1, mu_2, var_2, mask, dim):
        """Computes the batch KL-divergence between two Gaussian distributions with diagonal covariance."""
        if mu_2 is None and var_2 is None:
            return 0.5 * torch.sum((-torch.log(var_1) + var_1 + mu_1 ** 2 - 1) * mask.unsqueeze(dim), dim=dim)
        elif mu_2 is not None and var_2 is not None:
            return 0.5 * torch.sum((torch.log(var_2) - torch.log(var_1) + var_1 / var_2
                                    + (mu_2 - mu_1) ** 2 / var_2 - 1) * mask.unsqueeze(dim), dim=dim)
        else:
            raise InvalidArgumentError("Either provide mu_2 and var_2 or neither.")

    def _gaussian_entropy(self, var, mask, dim):
        """Computes the entropy of a Multivariate Gaussian with diagonal Covariance."""
        return 0.5 * torch.sum((torch.log(2 * np.pi * var) + 1.) * mask.unsqueeze(dim), dim=dim)

    def _gaussian_log_likelihood(self, sample, mask, dim, mu=None, var=None):
        """Computes the log likelihood of a given sample under a gaussian with given parameters.

        If mu or var is not given they are assumed to be standard.
        """
        if mu is None and var is None:
            return -0.5 * torch.sum((torch.log(sample.new_tensor(2*np.pi)) + sample**2) * mask.unsqueeze(dim), dim=dim)
        elif mu is None:
            return -0.5 * torch.sum((torch.log(2*np.pi*var) + sample**2 / var) * mask.unsqueeze(dim), dim=dim)
        elif var is None:
            return -0.5 * torch.sum((torch.log(sample.new_tensor(2*np.pi)) + (sample - mu)**2)
                                    * mask.unsqueeze(dim), dim=dim)
        else:
            return -0.5 * torch.sum((torch.log(2*np.pi*var) + (sample - mu)**2 / var) * mask.unsqueeze(dim), dim=dim)

    def _hinge_loss(self, kl, rate):
        """Computes the hinge loss between the mean KL-divergence and a specified Rate"""
        if self.rate_mode == "hinge":
            return self.hinge_weight * torch.max(torch.tensor(0., device=self.device), rate - kl)
        elif self.rate_mode == "fb":
            return torch.max(rate, kl)

    def _hinge_loss_mean(self, kl, rate):
        """Computes the mean hinge loss between a batch of KL-divergences and a specified Rate."""
        return torch.mean(torch.max(torch.tensor(0., device=self.device), rate - kl))

    def _mmd(self, sample_1, sample_2):
        """Computes an unbiased estimate of the MMD between two distributions given a set of samples from both."""
        mmd = MMDStatistic(max(2, sample_1.shape[0]), max(2, sample_2.shape[0]))
        if sample_1.shape[0] == 1:
            return 10000 * mmd(sample_1.expand(2, -1), sample_2.expand(2, -1), [1. / sample_1.shape[1]])
        else:
            return 10000 * mmd(sample_1, sample_2, [1. / sample_1.shape[1]])

    def _compute_gamma(self, kl):
        """Computes a scale factor for the KL divergence given a desired rate."""
        if kl < self.min_rate:
            self.scale = self.scale / (1. + self.kl_step.item() * 10)
        elif kl > self.min_rate * 1.05:
            self.scale = self.scale * (1. + self.kl_step.item() * 10)

    def _update_scale(self, kl):
        """Updates the scale factor for the KL divergence."""
        if self.ann_mode == "linear":
            self.scale = self.scale + self.kl_step.item()
        elif self.ann_mode == "sfb":
            self._compute_gamma(kl)

    def _update_word_p(self):
        """Updates the word dropout probability."""
        if self.ann_word:
            self.word_p = self.word_p - self.word_step.item()

    def q_z_estimate(self, z, mu, var):
        """Computes an estimate of q(z), the marginal posterior."""
        # z = [S, z_dim], mu = [N, z_dim], var = [N, z_dim], log_q_z = [S, N]
        log_q_z_x = self._sample_log_likelihood(z.unsqueeze(1), torch.tensor(
            [[1.]], device=self.device), dim=2, mu=mu, var=var)
        # [S,]
        log_q_z = torch.logsumexp(log_q_z_x, dim=1) - \
            torch.log(torch.tensor(log_q_z_x.shape[1], device=self.device, dtype=torch.float))
        return log_q_z

    def _compute_constraint(self, i, constraint, kl, mmd, nll):
        """Computes a constraint with weight updated with lagrangian relaxation."""
        if constraint == 'mdr':
            # Specifies a minimum desired rate, i.e. KL >= min_rate
            return self.lag_weight[i].abs() * (self.min_rate - kl)
        elif constraint == 'mmd':
            # Specifies a maximum desired MMD, i.e. MMD <= max_mmd
            return self.lag_weight[i].abs() * (mmd - self.max_mmd)
        elif constraint == 'elbo':
            # Specifies a maximum desired ELBO, i.e. ELBO <= max_elbo
            return self.lag_weight[i].abs() * (nll + kl - self.max_elbo) - (self.alpha + 1) * nll - kl

    def _compute_constraints(self, losses, mmd):
        """Computes all constraints and adds them together."""
        losses['Constraint'] = 0.
        for i, constraint in enumerate(self.constraint):
            losses["Constraint_{}".format(i)] = self._compute_constraint(
                i, constraint, losses['KL'], mmd, losses['NLL'])
            losses["Constraint"] += losses["Constraint_{}".format(i)]
