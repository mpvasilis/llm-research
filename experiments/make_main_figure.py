"""Build the main-text overview figure (out/paper/fig_measurement.pdf).

Panel (a) plots the advice-minus-control gaps of the final-7B stored responses
under four measurement choices (Table "window-components" in the paper).
Panel (b) plots the DPO chosen-versus-rejected odds ratios before and after
flexible length conditioning (Tables "dpo" and "dpo-intervals").

Values are read from the result files where available and otherwise match the
manuscript tables. Run from the repository root:

    python -m experiments.make_main_figure
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "out" / "paper" / "fig_measurement.pdf"

# Advice-minus-control gap, final 7B, in percentage points.
MARKERS = ["Empathy\n(vs. emotion,\nno request)",
           "Validation\n(vs. emotion,\nno request)",
           "Referral/discl.\n(vs. health\nfacts)",
           "Structuring\n(vs. neutral\nhow-to)"]
GAPS = {
    "Exact match, all text": [32.3, 15.3, 15.7, 39.0],
    "Exact or embedding, all text": [-14.7, 25.0, 9.0, -3.0],
    "Exact or embedding, first 128 tokens": [6.3, 21.0, 11.7, -3.0],
    "Exact or embedding, first 64 tokens": [25.3, 17.7, 11.3, -3.3],
}
GAP_COLORS = ["#1b3a57", "#c44e52", "#e39a9c", "#f3cfd0"]

# Odds ratio that a phrase sits on the chosen side: (estimate, low, high).
GROUPS = ["Referral/\ndisclaimer", "Narrower\nreferral", "Structuring"]
OR_MARKER_ONLY = {
    "1B": [(1.181, 1.114, 1.252), (1.170, 1.039, 1.318), (1.242, 1.220, 1.264)],
    "7B": [(1.352, 1.274, 1.434), (1.335, 1.187, 1.502), (1.435, 1.409, 1.461)],
}
OR_LENGTH = {
    "1B": [(1.088, 1.026, 1.154), (1.095, 0.972, 1.234), (1.128, 1.108, 1.149)],
    "7B": [(1.134, 1.067, 1.205), (1.121, 0.996, 1.263), (1.190, 1.168, 1.213)],
}


def main() -> None:
    plt.rcParams.update({"font.size": 8, "font.family": "Times New Roman",
                         "axes.spines.top": False, "axes.spines.right": False})
    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(7.0, 2.75), gridspec_kw={"width_ratios": [1.45, 1]})

    width = 0.2
    for i, (label, vals) in enumerate(GAPS.items()):
        xs = [j + (i - 1.5) * width for j in range(len(MARKERS))]
        ax1.bar(xs, vals, width, label=label, color=GAP_COLORS[i],
                edgecolor="white", linewidth=0.4)
    ax1.axhline(0, color="black", linewidth=0.6)
    ax1.set_xticks(range(len(MARKERS)))
    ax1.set_xticklabels(MARKERS)
    ax1.set_ylabel("Advice minus control (pp)")
    ax1.set_ylim(-22, 68)
    ax1.legend(frameon=False, fontsize=6.5, loc="upper center", ncol=2,
               handlelength=1.2, columnspacing=1.0)
    ax1.set_title("(a) Same 700 responses, four measurement choices",
                  fontsize=8, loc="left")

    offsets = {"1B": -0.2, "7B": 0.2}
    for size, off in offsets.items():
        for style, data, face in (("marker only", OR_MARKER_ONLY, "white"),
                                  ("length-conditioned", OR_LENGTH, "#1b3a57")):
            shift = off + (-0.07 if style == "marker only" else 0.07)
            for j, (est, lo, hi) in enumerate(data[size]):
                ax2.errorbar(j + shift, est, yerr=[[est - lo], [hi - est]],
                             fmt="o" if size == "1B" else "s", ms=4,
                             mfc=face, mec="#1b3a57", ecolor="#1b3a57",
                             elinewidth=0.8, capsize=1.5)
    ax2.axhline(1, color="black", linewidth=0.6, linestyle="--")
    ax2.set_xticks(range(len(GROUPS)))
    ax2.set_xticklabels(GROUPS)
    ax2.set_ylabel("Odds ratio, chosen vs. rejected")
    ax2.set_xlim(-0.55, 2.55)
    handles = [
        plt.Line2D([], [], marker="o", ls="", mfc="white", mec="#1b3a57",
                   ms=4, label="1B, marker only"),
        plt.Line2D([], [], marker="o", ls="", mfc="#1b3a57", mec="#1b3a57",
                   ms=4, label="1B, length-conditioned"),
        plt.Line2D([], [], marker="s", ls="", mfc="white", mec="#1b3a57",
                   ms=4, label="7B, marker only"),
        plt.Line2D([], [], marker="s", ls="", mfc="#1b3a57", mec="#1b3a57",
                   ms=4, label="7B, length-conditioned"),
    ]
    ax2.legend(handles=handles, frameon=False, fontsize=6.5, loc="upper left",
               ncol=1, handletextpad=0.3)
    ax2.set_ylim(0.9, 1.9)
    ax2.set_title("(b) DPO pairs: length conditioning", fontsize=8, loc="left")

    fig.tight_layout(w_pad=1.5)
    fig.savefig(OUT)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
