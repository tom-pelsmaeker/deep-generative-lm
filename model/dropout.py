from torch import nn
from torch.distributions import Bernoulli

__author__ = "Tom Pelsmaeker"
__copyright__ = "Copyright 2020"


class FlexibleDropout(nn.Module):
    """FlexibleDropout disconnects the sampling step from the masking step of dropout.

    There are two important differences between FlexibleDropout and nn.Dropout. First, FlexibleDropout exposes a
    sample_mask and apply_mask function, that allows for the same mask to be used repeatedly. Second, FlexibleDropout
    scales the input at test time with p, as opposed to scaling with 1/p at training time. This is convenient when
    Dropout is used for uncertainty estimation.
    """

    def __init__(self):
        super().__init__()

    def forward(self, input, p, shape=None):
        """Similar to F.dropout, a mask is sampled and directly applied to the input."""
        if shape is None:
            self.sample_mask(p, input.shape)
        else:
            self.sample_mask(p, shape)

        return self.apply_mask(input)

    def apply_mask(self, input):
        """Applies the sampled mask to the input."""
        return input * self._mask

    def sample_mask(self, p, shape):
        """Samples a dropout mask from a Bernoulli distribution.

        Args:
            p(float): the dropout probability [0, 1].
            shape(torch.Size): shape of the mask to be sampled.
        """
        if self.training:
            self._mask = Bernoulli(1. - p).sample(shape)
        else:
            self._mask = (1. - p)
