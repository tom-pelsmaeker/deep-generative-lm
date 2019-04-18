"""
Simple FeedForward networks to estimate parameters of latent Gaussian distributions.
"""

import sys
import os.path as osp

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, PackedSequence, pad_packed_sequence
from torch.autograd import Function

# We include the path of the toplevel package in the system path so we can always use absolute imports within the package.
toplevel_path = osp.abspath(osp.join(osp.dirname(__file__), '..'))
if toplevel_path not in sys.path:
    sys.path.insert(1, toplevel_path)

from model.dropout import FlexibleDropout

__author__ = "Tom Pelsmaeker"
__copyright__ = "Copyright 2018"


class MuForward(nn.Module):
    """Small feedforward module that outputs the mean of a Gaussian."""

    def __init__(self, in_dim, out_dim):
        super(MuForward, self).__init__()

        # Model structure
        self.linear_in = nn.Linear(in_dim, out_dim * 2)
        self.linear_out = nn.Linear(out_dim * 2, out_dim)

    def forward(self, inp):
        """Input is assumed to be a sequence of torch tensors with size [batch_size x dim]."""
        # Stack inputs along the second dimension
        x = torch.cat(inp, dim=1)

        # Forward pass
        h = self.linear_in(x)
        h = torch.tanh(h)
        mu = self.linear_out(h)

        return mu


class varForward(nn.Module):
    """Small Feedforward module that outputs the diagonal covariance of a Gaussian."""

    def __init__(self, in_dim, out_dim):
        super(varForward, self).__init__()

        # Model structure
        self.linear_in = nn.Linear(in_dim, out_dim * 2)
        self.linear_out = nn.Linear(out_dim * 2, out_dim)

    def forward(self, inp):
        """Input is assumed to be a sequence of torch tensors with size [batch_size x dim]"""
        # Stack inputs along the second dimension
        x = torch.cat(inp, dim=1)

        # Forward pass
        h = self.linear_in(x)
        h = torch.tanh(h)
        var = self.linear_out(h)

        return F.softplus(var)


class BowmanEncoder(nn.Module):
    "Bowman style LSTM-encoder of a sentence into a mean and var."

    def __init__(self, in_dim, h_dim, out_dim, l_dim, p, var_mask, posterior, k=0, drop_inp=True):
        super().__init__()

        self.p = p
        self.var_mask = var_mask
        self.in_dim = in_dim
        self.h_dim = h_dim
        self.out_dim = out_dim
        self.drop_inp = drop_inp
        self.posterior = posterior
        self.k = k
        self.grad_clamp = SoftClampGradients.apply

        # Model structure
        self.gru = nn.GRU(in_dim, h_dim, l_dim, batch_first=True, bidirectional=True)
        self.mu = nn.Linear(2 * l_dim * h_dim, out_dim)
        if self.posterior == "gaussian":
            self.var = nn.Linear(2 * l_dim * h_dim, out_dim)
        elif self.posterior == "vmf":
            self.var = nn.Linear(2 * l_dim * h_dim, 1)

        self.drop = FlexibleDropout()

    def forward(self, x, x_len, return_hidden=False):
        """Forward takes a sequence and possibly a length masking, to return a mu and var of diag Gaussian."""
        # x = [B x S x in_dim], x_len = [B]
        # Apply dropout to the input
        if self.drop_inp:
            shape = torch.Size(x.shape) if self.var_mask else torch.Size([x.shape[0], 1, self.in_dim])
            x = self.drop(x, self.p, shape=shape)

        # Pack padded if sequences have variable lengths
        if x_len is not None:
            x = pack_padded_sequence(x, x_len, batch_first=True)

        # Pass through GRU, obtain final output, and reshape to [B x 2 * l_dim * h_dim]
        _, h_n = self.gru(x)
        h_n = h_n.transpose(0, 1).contiguous().view([x_len.shape[0], -1])

        # Apply dropout to the output
        h_n = self.drop(h_n, self.p)

        # Estimate Gaussian Parameters
        mu = self.mu(h_n)
        var = self.var(h_n)

        if self.posterior == "vmf":
            # The location of the vMF distribution on the unit sphere
            mu = mu / mu.norm(2, dim=-1, keepdim=True)
            if self.k <= 0.:
                # Use tanh to clamp the var between z_dim / 3 and z_dim * 3
                # var = F.softplus(var)
                var = torch.tanh(var) * ((self.out_dim * 3.5 - self.out_dim / 3.5) / 2.) + \
                    ((self.out_dim * 3.5 + self.out_dim / 3.5) / 2)

                # Clamp the gradients to make sure tanh does not oversaturate
                var = self.grad_clamp(var, self.out_dim / 3.45, self.out_dim * 3.45)
            else:
                # We can also fix k in advance
                var = torch.ones_like(var) * self.k * self.out_dim
        else:
            var = F.softplus(var)

        if return_hidden:
            return mu, var, h_n
        else:
            return mu, var


class SoftClampGradients(Function):
    """Heuristic soft clamping of the gradients.

    Whenever the input is below a certain threshold, we set all posiive gradients to zero.
    Similarly, whenever the input is above a certain threshold, we set all negative gradients to zero.
    """

    @staticmethod
    def forward(ctx, var, lowerbound, upperbound):
        ctx.save_for_backward(var, var.new_tensor(lowerbound), var.new_tensor(upperbound))
        return var

    @staticmethod
    def backward(ctx, grad_output):
        var, lowerbound, upperbound = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input[(var > upperbound) * (grad_input < 0.)] = 0.
        grad_input[(var < lowerbound) * (grad_input > 0.)] = 0.
        return grad_input, None, None
