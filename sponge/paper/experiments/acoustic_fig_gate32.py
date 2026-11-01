"""Re-plot the 21-qubit acoustic gate-level figure from figures/fig6_data.npz
(written by acoustic_gate_32x32.py) without re-running the simulation.
RdBu_r colormap (symmetric, white = 0); dashed black lines = layer strips.
Out: figures/fig6_2d_gate_level_32x32.png"""
import os
import sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)
from lib import WaveEquationSponge2d  # noqa: E402

N, NPML = 32, 8
d = np.load(os.path.join(ROOT, "figures", "fig6_data.npz"))
sponge = WaveEquationSponge2d(5, 5, npml=NPML)
bump = np.exp(-0.5 * ((np.arange(N) - N // 2) / 1.5) ** 2)   # as in acoustic_gate_32x32.py
z0 = sponge.initial_state(np.outer(bump, bump))
v0 = z0[:N * N].real.reshape(N, N).T
times = d["times"]
vmax = float(np.abs(d["fields_ref"][0]).max())   # peak of the first propagated snapshot
sys.path.insert(0, HERE)
import figstyle as fs  # noqa: E402
fs.setup()
top = [v0] + [f for f in d["fields_gate"]]
bot = [v0] + [f for f in d["fields_ref"]]
notes = [[r"error $0$"] + [rf"error ${fs.sci(e)}$" for e in d["errors"]],
         [r"energy $1$"] + [rf"energy ${E:.2g}$" for E in d["energies"]]]
titles = [r"$t=0$"] + [rf"$t={T:g}$" for T in times]
fig, axs = fs.wavefield_grid([top, bot], ["quantum\ncircuit", "classical\nPDE"], titles,
                             vmax, (NPML - 0.5, N - 0.5 - NPML), notes=notes,
                             cbar_label=r"$v/v_{\mathrm{ref}}$")
fs.panel_label(axs[0][0], "(a)", x=0.05, y=0.95)
fs.panel_label(axs[1][0], "(b)", x=0.05, y=0.95)
fs.save(fig, "fig6_2d_gate_level_32x32")
