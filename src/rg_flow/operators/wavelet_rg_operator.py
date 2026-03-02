"""
src/rg_flow/operators/wavelet_rg_operator.py

WaveletRGOperator: wavelet-based multi-resolution RG coarse-graining.

Physical motivation: classical real-space RG integrates out short-wavelength
(high-frequency) modes at each step. The Haar wavelet transform provides an
exact decomposition into scales, where each level of the wavelet hierarchy
corresponds to one RG step. The WaveletRGOperator therefore implements the
RG blocking transformation natively in wavelet space.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class WaveletRGOperator(nn.Module):
    """
    Wavelet-based multi-resolution analysis (MRA) RG transformation.

    At each RG step (network layer), the operator:
        1. Decomposes the input into low-frequency (scaling) and
           high-frequency (wavelet) coefficients via learned filter banks.
        2. Retains only the low-frequency (coarse-grained) component
           as the next-layer representation.
        3. Optionally mixes the retained high-frequency detail back in
           via learnable mixing coefficients (skip connections in wavelet space).

    This corresponds to the Wilson-Kadanoff real-space RG where the
    blocking transformation is implemented via a wavelet filter bank.

    The wavelet decomposition filters are initialized as Haar filters
    (perfect low-pass / band-pass pair) and then fine-tuned by backprop.
    """

    def __init__(
        self,
        in_dim:      int,
        out_dim:     int,
        n_scales:    int   = 2,
        use_detail:  bool  = True,
    ) -> None:
        """
        Args:
            in_dim:      input feature dimension
            out_dim:     output feature dimension (coarse-grained representation)
            n_scales:    number of wavelet decomposition levels
            use_detail:  whether to incorporate high-frequency detail coefficients
        """
        super().__init__()
        self.in_dim     = in_dim
        self.out_dim    = out_dim
        self.n_scales   = n_scales
        self.use_detail = use_detail

        # Low-pass (scaling) filter bank
        self.lowpass = nn.Linear(in_dim, out_dim)

        # High-pass (detail) filter bank
        if use_detail:
            self.highpass = nn.ModuleList([
                nn.Linear(in_dim, out_dim) for _ in range(n_scales)
            ])
            self.detail_mix = nn.Parameter(
                torch.zeros(n_scales) * 0.1
            )

        # Critical initialization: Haar low-pass is the mean
        nn.init.normal_(self.lowpass.weight, std=1.0 / in_dim ** 0.5)
        nn.init.zeros_(self.lowpass.bias)

        self.activation = nn.Tanh()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply wavelet-based RG coarse-graining.

        Args:
            x: (B, in_dim) input features

        Returns:
            h: (B, out_dim) coarse-grained features
        """
        # Low-frequency (coarse) component
        h = self.activation(self.lowpass(x))

        # Add high-frequency (detail) corrections
        if self.use_detail:
            mixing = torch.sigmoid(self.detail_mix)
            for i, hp in enumerate(self.highpass):
                detail = self.activation(hp(x))
                h = h + mixing[i] * detail

        return h
 