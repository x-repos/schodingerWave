"""1D graded damping (sponge) layer for the two-block wave system.

In one dimension the staggered damping layer is
    A = -i H - diag(sigma_v, sigma_w),
with sigma_v sampled at the nodes and sigma_w at the half-cells.  This is
numerically the class WaveEquationCPML1d of the vendored absorbing.py
(the convolutional layer collapses exactly to this diagonal damping in
1D), re-exported under its sponge name for this paper.
"""
from .absorbing import WaveEquationCPML1d


class WaveEquationSponge1d(WaveEquationCPML1d):
    """Two-block 1D wave equation with staggered diagonal damping."""
