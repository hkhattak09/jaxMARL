import math

import jax.numpy as jnp
from jax import random as jrandom
import flax.linen as nn


class SuperLinear(nn.Module):
    """Neuron-Level Model (NLM): N independent linear transformations in one op.

    Each of the N neurons has its own weight matrix, so the layer applies a
    different linear map to each neuron's feature vector simultaneously.

    Weight layout mirrors the PyTorch implementation:
        w : (in_dims, out_dims, N)
        b : (1, N, out_dims)

    The einsum  'BDM,MHD->BDH'  contracts:
        B = batch, D = neurons (N), M = in_dims, H = out_dims

    Input:  (B, N, in_dims)
    Output: (B, N, out_dims)  — or (B, N) when out_dims == 1 (squeezed)
    """
    out_dims: int
    N: int
    dropout_rate: float = 0.0

    @nn.compact
    def __call__(self, x, deterministic: bool = True):
        in_dims = x.shape[-1]
        scale = 1.0 / math.sqrt(in_dims + self.out_dims)

        w = self.param(
            'w',
            lambda rng, shape: jrandom.uniform(rng, shape, minval=-scale, maxval=scale),
            (in_dims, self.out_dims, self.N),
        )
        b = self.param('b', nn.initializers.zeros_init(), (1, self.N, self.out_dims))

        out = nn.Dropout(rate=self.dropout_rate)(x, deterministic=deterministic)
        out = jnp.einsum('BDM,MHD->BDH', out, w) + b

        if out.shape[-1] == 1:
            out = out.squeeze(-1)
        return out
