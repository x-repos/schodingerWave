"""Figure: CX per Trotter step -- generic damping diagonal vs structured
damping (ancilla-free / one flag ancilla) vs the wave step.
Reads damping_gatecounts_results.json; writes figures/fig_damping_gatecounts.{pdf,png}."""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import figstyle as fs  # noqa: E402

fs.setup()
import matplotlib.pyplot as plt  # noqa: E402

d = json.load(open(os.path.join(HERE, "damping_gatecounts_results.json")))
series = [("cx_generic_formula", r"generic diagonal ($2^{n}-2$)", fs.BLACK, "^", "-"),
          ("cx_full_direct", r"structured, ancilla-free", fs.BLUE, "o", "-"),
          ("cx_full_flag", r"structured, one flag ancilla", fs.GREEN, "s", "-"),
          ("cx_wave_step", r"wave step (Bell, Strang)", fs.VERMILION, "D", "--")]
fig, axs = plt.subplots(1, 2, figsize=(fs.TEXTWIDTH, 2.5), sharey=True)
for k, (ax, kind) in enumerate(zip(axs, ("acoustic", "elastic"))):
    rr = sorted([r for r in d["rows"] if r["kind"] == kind], key=lambda r: r["n_total"])
    n = [r["n_total"] for r in rr]
    for key, lab, col, mk, ls in series:
        ax.semilogy(n, [r[key] for r in rr], ls=ls, marker=mk, color=col, mfc="white",
                    mew=0.9, label=lab)
    for r in rr:
        ax.annotate(rf"${r['N']}^2$", (r["n_total"], r["cx_generic_formula"]),
                    textcoords="offset points", xytext=(0, 5), ha="center", fontsize=7)
    ax.set_xlabel(r"total qubits $n=n_{\mathrm{sys}}+n_p$")
    ax.set_xticks(n)
    fs.panel_label(ax, rf"({'ab'[k]}) {kind}", x=0.03, y=0.06 if False else 0.97)
    ax.set_ylim(1.2e2, 3e8)
axs[0].set_ylabel(r"CNOTs per Trotter step")
h, l = axs[0].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.02), columnspacing=2.0)
fig.tight_layout(rect=(0, 0.13, 1, 1), w_pad=0.6)
fs.save(fig, "fig_damping_gatecounts")
