"""
Numeric Basis Fourier Feature Encoding (NUM-01, NUM-02).

Encodes continuous clinical scalars (vitals, lab results, inputs, outputs)
into 22 dyadic Fourier scales: 2^-7, 2^-6, ..., 2^14.
Under sine + cosine formulation, dimension is 22 * 2 = 44.
"""

from typing import Union
import numpy as np


class DyadicFourierEncoder:
    """
    Transforms continuous numeric values into 44-dimensional dyadic Fourier features.
    
    phi(v) = [sin(kappa * 2^j * g(v)), cos(kappa * 2^j * g(v))] for j in [-7, 14]
    """
    def __init__(self, kappa: float = 1.0, normalize: bool = False):
        self.kappa = kappa
        self.normalize = normalize
        # 22 dyadic scales from 2^-7 to 2^14
        self.exponents = np.arange(-7, 15, dtype=np.float32)  # length 22
        self.frequencies = np.power(2.0, self.exponents)      # [22]

    def encode(self, val: Union[float, int, np.ndarray]) -> np.ndarray:
        """
        Encode a single scalar or array of scalars into [..., 44] Fourier features.
        
        Args:
            val: float scalar or 1D array of floats of shape [N]
            
        Returns:
            np.ndarray of shape [44] for scalar, or [N, 44] for 1D array.
        """
        v = np.asarray(val, dtype=np.float32)
        is_scalar = v.ndim == 0
        if is_scalar:
            v = v.reshape(1)

        # Compute arguments: shape [N, 22]
        # v[:, None] * frequencies[None, :]
        args = self.kappa * (v[:, None] * self.frequencies[None, :])
        
        sin_part = np.sin(args)  # [N, 22]
        cos_part = np.cos(args)  # [N, 22]
        
        # Interleave or concatenate: [N, 44]
        # Candidate convention: concatenate [sin_0..21, cos_0..21]
        out = np.concatenate([sin_part, cos_part], axis=-1)
        
        if is_scalar:
            return out[0]
        return out

    @property
    def output_dim(self) -> int:
        return 44
