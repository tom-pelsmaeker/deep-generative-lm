"""
Flows for VAEs.

Parts of the code below are adapted from: https://github.com/riannevdberg/sylvester-flows/blob/master/models/flows.py

# ORIGINAL LICENSE ###############################################################
#                                                                                #
# MIT License                                                                    #
#                                                                                #
# Copyright (c) 2018 Rianne van den Berg                                         #
#                                                                                #
# Permission is hereby granted, free of charge, to any person obtaining a copy   #
# of this software and associated documentation files (the "Software"), to deal  #
# in the Software without restriction, including without limitation the rights   #
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell      #
# copies of the Software, and to permit persons to whom the Software is          #
# furnished to do so, subject to the following conditions:                       #
#                                                                                #
# The above copyright notice and this permission notice shall be included in all #
# copies or substantial portions of the Software.                                #
#                                                                                #
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR     #
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,       #
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE    #
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER         #
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,  #
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE  #
# SOFTWARE.                                                                      #
##################################################################################
"""

import math
import sys
import os.path as osp
from warnings import warn

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import xavier_normal_
from torch.nn.parameter import Parameter

# We include the path of the toplevel package in the system path so we can always use absolute imports within the package.
toplevel_path = osp.abspath(osp.join(osp.dirname(__file__), '..'))
if toplevel_path not in sys.path:
    sys.path.insert(1, toplevel_path)

from util.error import UnknownArgumentError, InvalidArgumentError  # noqa: E402


class AutoregressiveLinear(nn.Module):
    """Simple autoregressive Linear layer.

    This network is a drop-in for a linear layer with autoregressive properties. As such, it is convenient to use
    for normalizing flows, as the determinant of such an autoregressive layer will evaluate to zero (when the weight matrix is square).

    Args:
        in_dim(int): size of the input of this layer.
        out_dim(int): size of the output of this layer.
        diag(str): When 'zero', zeros on the diagonal, when 'one', ones on the diagonal, else parameters on the diagonal.
        bias(bool): whether to use a bias vector. Defaults to True
    """

    def __init__(self, in_dim, out_dim, diag="zero", bias=True):
        super(AutoregressiveLinear, self).__init__()
        self.diag = diag
        self.weight = Parameter(torch.Tensor(out_dim, in_dim))

        if self.diag == 'one':
            self.I = torch.eye(out_dim, in_dim)

        if bias:
            self.bias = Parameter(torch.Tensor(out_dim))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        """Initializes the bias with zero, the weights with xavier/glorot normal."""
        self.weight = xavier_normal_(self.weight)

        if self.bias is not None:
            self.bias.data.zero_()

    def forward(self, input):
        """Affine transformation with lower triangular weight matrix."""
        if self.diag == "zero":
            return F.linear(input, self.weight.tril(-1), self.bias)
        elif self.diag == "one":
            return F.linear(input, self.weight.tril(-1) + input.new_tensor(self.I), self.bias)
        else:
            return F.linear(input, self.weight.tril(0), self.bias)


class AutoregressiveNetwork(nn.Module):
    """
    Simple N layer autoregressive network (or MADE) that outputs a location and scale.

    Args:
        h_dim(int): size of the hidden layers and variable
        z_dim(int): size of the latent variable.
        hidden_depth(int): number of hidden to hidden layers in the flow. Default to 1.
        bias(boolean): Whether to add bias vectors at each affine transformation. Defaults to True.
        scale(boolean): Whether to return a scale. Defaults to True.
        diag(str): The number on the diagonals of the output Jacobians. Can be "zero" or "one".
        nonlinearity(Function/Module): nonlinearity after each affine transformation. Defaults to ELU.
    """

    def __init__(self, h_dim, z_dim, hidden_depth=1, bias=True, scale=True, diag="zero", nonlinearity=nn.ELU()):
        super(AutoregressiveNetwork, self).__init__()

        self.h_dim = h_dim
        self.z_dim = z_dim
        self.h_to_h = nn.ModuleList()
        self.scale = scale
        other_diag = "one" if diag == "one" else "full"

        # First layer scales z -> h
        self.z_to_h = nn.Sequential(
            AutoregressiveLinear(self.z_dim, self.h_dim, other_diag, bias),
            nonlinearity)

        # Then N h -> h layers follow
        for i in range(hidden_depth):
            step = nn.Sequential(
                AutoregressiveLinear(self.h_dim, self.h_dim, other_diag, bias),
                nonlinearity)
            self.h_to_h.append(step)

        # Finally, we have two separate layers that produce a mu and var respectively.
        # Because these layers are lower-triangular with zeroes on the diagonal, and the outputs
        # have the same dimensionality as z, it can be determined by simple chain-rule factorization that the
        # determinants of mu and var wrt the previous z will be zero.
        self.h_to_mu = AutoregressiveLinear(self.h_dim, self.z_dim, diag, bias)
        if self.scale:
            self.h_to_var = nn.Sequential(
                AutoregressiveLinear(self.h_dim, self.z_dim, diag, bias),
                nn.Sigmoid())

    def forward(self, z, h):
        """The forward pass computes a new mu and var that have zero determinants wrt the input z.

        Args:
            z(torch.FloatTensor): previous latent vector of the flow.
            h(torch.FloatTensor): previous hidden state of the flow.

        Returns:
            mu(torch.FloatTensor): mean of the current z.
            var(torch.FloatTensor): var or gate of the current z.
        """
        h_0 = self.z_to_h(z)
        h = h_0 + h
        for step in self.h_to_h:
            h = step(h)

        mu = self.h_to_mu(h)
        if self.scale:
            var = self.h_to_var(h)
        else:
            var = h.new_tensor([1.])

        return mu, var


class PlanarStep(nn.Module):

    def __init__(self):
        super(PlanarStep, self).__init__()

        self.h = nn.Tanh()
        self.softplus = nn.Softplus()

    def _der_h(self, x):
        """Derivative of activation function h."""
        return self._der_tanh(x)

    def _der_tanh(self, x):
        "Derivative of the Tanh function."
        return 1 - self.h(x) ** 2

    def forward(self, zk, u, w, b):
        """
        Forward pass. Assumes amortized u, w and b. Conditions on diagonals of u and w for invertibility
        will be be satisfied inside this function. Computes the following transformation:
        z' = z + u h( w^T z + b)
        or actually
        z'^T = z^T + h(z^T w + b)u^T
        Assumes the following input shapes:
        shape u = (batch_size, z_dim, 1)
        shape w = (batch_size, 1, z_dim)
        shape b = (batch_size, 1, 1)
        shape z = (batch_size, z_dim).
        """

        zk = zk.unsqueeze(2)

        # reparameterize u such that the flow becomes invertible (see appendix paper)
        uw = torch.bmm(w, u)
        m_uw = -1. + self.softplus(uw)
        w_norm_sq = torch.sum(w ** 2, dim=2, keepdim=True)
        u_hat = u + ((m_uw - uw) * w.transpose(2, 1) / w_norm_sq)

        # compute flow with u_hat
        wzb = torch.bmm(w, zk) + b
        z = zk + u_hat * self.h(wzb)
        z = z.squeeze(2)

        # compute logdetJ
        psi = w * self._der_h(wzb)
        logdet = torch.log(torch.abs(1 + torch.bmm(psi, u_hat)))
        logdet = logdet.squeeze(2).squeeze(1)

        return z, logdet


class NormalizingFlow(nn.Module):
    """Base class for normalizing flows."""

    def __init__(self, h_dim, z_dim, flow_depth, hidden_depth):
        super(NormalizingFlow, self).__init__()
        self.h_dim = h_dim
        self.z_dim = z_dim
        self.flow_depth = flow_depth
        self.hidden_depth = hidden_depth

    @property
    def flow_depth(self):
        return self._flow_depth

    @flow_depth.setter
    def flow_depth(self, value):
        if not isinstance(value, int):
            raise InvalidArgumentError("flow_depth should be an integer.")
        elif value < 1:
            raise InvalidArgumentError("flow_depth should be strictly positive.")
        else:
            self._flow_depth = value

    @property
    def hidden_depth(self):
        return self._hidden_depth

    @hidden_depth.setter
    def hidden_depth(self, value):
        if not isinstance(value, int):
            raise InvalidArgumentError("hidden_depth should be an integer.")
        elif value < 0:
            raise InvalidArgumentError("hidden_depth should be positive.")
        else:
            self._hidden_depth = value


class Planar(NormalizingFlow):

    def __init__(self, h_dim, z_dim, flow_depth):
        super(Planar, self).__init__(h_dim, z_dim, flow_depth, 0)

        self.flow = PlanarStep()

        # Amortized flow parameters
        self.h_to_u = nn.Linear(self.h_dim, self.flow_depth * self.z_dim)
        self.h_to_w = nn.Linear(self.h_dim, self.flow_depth * self.z_dim)
        self.h_to_b = nn.Linear(self.h_dim, self.flow_depth)

    def forward(self, z, h):
        # compute amortized u an w for all flows
        u = self.h_to_u(h).view(-1, self.flow_depth, self.z_dim, 1)
        w = self.h_to_w(h).view(-1, self.flow_depth, 1, self.z_dim)
        b = self.h_to_b(h).view(-1, self.flow_depth, 1, 1)

        z_k = z
        logdet = 0.
        for k in range(self.flow_depth):
            z_k, ldj = self.flow(z_k, u[:, k, :, :], w[:, k, :, :], b[:, k, :, :])
            logdet += ldj

        return z_k, logdet


class IAF(NormalizingFlow):
    """The Inverse Autoregressive Flow with autoregressive MLPs.

    The default IAF module computes an IAF on its inputs with depth flow_depth and hidden_depth layers per step. Using
    the scale boolean we can define the flow to be volume-preserving.

    Args:
        h_dim(int): size of the hidden layers and variable.
        z_dim(int): size of the latent variable.
        flow_depth(int): number of steps in the flow. Defaults to 2.
        hidden_depth(int): number of hidden to hidden layers per step in the flow. Defaults to 1.
        scale(boolean): when False, a location-only transform is used. Defaults to True.
        nonlinearity(Function/Module): nonlinearity after each affine transformation. Defaults to ELU.
    """

    def __init__(self, h_dim, z_dim, flow_depth=2, hidden_depth=1, scale=True, nonlinearity=nn.ELU()):
        super(IAF, self).__init__(h_dim, z_dim, flow_depth, hidden_depth)
        self.scale = scale
        self.flow = nn.ModuleList([AutoregressiveNetwork(
            h_dim, z_dim, hidden_depth, scale=scale, nonlinearity=nonlinearity) for _ in range(flow_depth)])

    def forward(self, z, h):
        """The forward pass computes a new z for each step in the flow and registers the log-determinant.

        Args:
            z(torch.FloatTensor): initial latent vector.
            h(torch.FloatTensor): hidden state of the flow.

        Returns:
            z(torch.FloatTensor): final latent vector.
            logdet(torch.FloatTensor): the logdeterminant of the flow for each item in the batch.

        """
        logdet = 0.
        for i, step in enumerate(self.flow):
            if i % 2 == 0:
                # flip z every other step in the flow
                z = z.flip(1)
            mu, var = step(z, h)
            if self.scale:
                z = var * z + (1 - var) * mu
                logdet += torch.log(var).sum(dim=1)
            else:
                z = z + mu
            if i % 2 == 0:
                # backflip
                z = z.flip(1)

        return z, logdet


class Diag(nn.Module):
    """No Flow that passes through the given parameter."""

    def __init__(self):
        super(Diag, self).__init__()

    def forward(self, z, h):
        return z, z.new_tensor([0.])
