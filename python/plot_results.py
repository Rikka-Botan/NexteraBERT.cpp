#!/usr/bin/env python3
"""Draw the README figure from the JSON files in results/.

    python python/plot_results.py     # -> assets/summary.png

The look follows the NexteraBERT model card's results figure: white panels titled
"(a) ...", left / bottom axes only, one legend above the panels, NexteraBERT as the
heavy solid blue line and each baseline in its own colour and dash pattern.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent

INK, TICK, GRID, AXIS = "#111111", "#555555", "#ececec", "#2b2b2b"
BLUE = "#2a78d6"
WHITE = "#ffffff"

# One colour, dash pattern and marker per model, as in the model card's figure.
NEXTERA = dict(color=BLUE, linestyle="-", marker="o", linewidth=3.4, markersize=7.5, zorder=5)
BASELINES = {
    "modernbert": dict(label="ModernBERT-base", color="#e34948", linestyle=(0, (3, 1.6)), marker="s"),
    "lfm": dict(label="LFM2.5-Encoder", color="#1baf7a", linestyle=(0, (5, 1.5, 1.2, 1.5)), marker="^"),
    "bert": dict(label="BERT-base", color="#4a3aa7", linestyle=(0, (1, 1.4)), marker="D"),
}
BENCH_KEY = {"modernbert": "PyTorch ModernBERT-base", "lfm": "PyTorch LFM2.5-Encoder-230M",
             "bert": "PyTorch bert-base-uncased"}

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10.5,
    "figure.facecolor": WHITE, "axes.facecolor": WHITE, "savefig.facecolor": WHITE,
    "text.color": INK, "axes.labelcolor": INK, "axes.labelsize": 10.5,
    "xtick.color": TICK, "ytick.color": TICK, "xtick.labelsize": 9.8, "ytick.labelsize": 9.8,
    "axes.edgecolor": AXIS, "axes.linewidth": 1.5,
    "xtick.direction": "out", "ytick.direction": "out",
    "xtick.major.size": 5, "ytick.major.size": 5, "xtick.major.width": 1.4, "ytick.major.width": 1.4,
    "xtick.minor.size": 0, "ytick.minor.size": 0,
})


def panel(ax, title: str) -> None:
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(color=GRID, linewidth=1.0)
    ax.set_axisbelow(True)
    ax.set_title(title, loc="left", fontsize=12.5, pad=8)


def base_style(key: str) -> dict:
    s = dict(BASELINES[key])
    s.pop("label")
    return dict(s, linewidth=2.5, markersize=6.5, zorder=3)


def draw(ax, xs, ys, style: dict, **kw):
    return ax.plot(xs, ys, markeredgecolor=WHITE, markeredgewidth=1.3, solid_capstyle="round",
                   dash_capstyle="butt", **{**style, **kw})


def headline(ax, x, y, big: str, small: str, ha="right", color=BLUE, gap=0.105) -> None:
    """A large figure with a line of explanation under it (axes coordinates)."""
    ax.text(x, y, big, transform=ax.transAxes, ha=ha, va="top", fontsize=15.5, fontweight="bold", color=color)
    ax.text(x, y - gap, small, transform=ax.transAxes, ha=ha, va="top", fontsize=9.2, color=INK, linespacing=1.15)


def model_legend(fig, nextera_label: str, keys, y=0.985, ncol=4) -> None:
    handles = [Line2D([], [], markeredgecolor=WHITE, markeredgewidth=1.2, label=nextera_label,
                      **{k: v for k, v in NEXTERA.items() if k != "zorder"})]
    for k in keys:
        st = base_style(k)
        st.pop("zorder")
        handles.append(Line2D([], [], markeredgecolor=WHITE, markeredgewidth=1.2, label=BASELINES[k]["label"], **st))
    leg = fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, y), ncol=ncol, fontsize=10.2,
                     handlelength=2.6, columnspacing=1.6, borderpad=0.55, frameon=True, fancybox=True)
    leg.get_frame().set_edgecolor("#d9d9d9")
    leg.get_frame().set_linewidth(1.6)


# ---------------------------------------------------------------------------
# MLM accuracy of the quantized files, and CPU speed against other encoders
# ---------------------------------------------------------------------------

def plot_summary(verify: list[dict], mlm: list[dict], cpu: dict, out: Path, ours: str = "Q4_K_M") -> None:
    order = ["Q4_0", "Q4_K_S", "Q4_K_M", "Q5_K_M", "Q6_K", "Q8_0"]           # by file size
    v = {r["ftype"]: r for r in verify}
    xs = list(range(len(order)))

    fig, (a, b) = plt.subplots(1, 2, figsize=(7.4, 4.25))
    fig.subplots_adjust(left=0.105, right=0.975, top=0.715, bottom=0.235, wspace=0.36)

    # (a) masked-LM accuracy of each file type, against the fp32 model
    m = {r["ftype"]: r for r in mlm}                 # eval_mlm.py: tens of thousands of positions
    acc = [100 * m[k]["accuracy"] for k in order]
    ref = 100 * m[ours]["accuracy_ref"]
    # paired standard error of the difference to fp32 (same masked positions)
    a.errorbar(xs, acc, yerr=[100 * m[k]["accuracy_diff_se"] for k in order], fmt="none", ecolor=BLUE,
               elinewidth=1.4, capsize=3.5, capthick=1.4, zorder=6)
    a.axhline(ref, color=AXIS, linewidth=1.2, linestyle=(0, (1, 2)), zorder=2)
    a.text(xs[0] - 0.3, ref + 0.12, f"fp32 (PyTorch): {ref:.1f}", ha="left", va="bottom", fontsize=9.2, color=INK)
    draw(a, xs, acc, NEXTERA)
    i = order.index(ours)
    a.annotate(f"{acc[i]:.1f}", (i, acc[i]), xytext=(0, -18), textcoords="offset points", ha="center",
               fontsize=10.5, color=BLUE)
    a.set_xticks(xs, [f"{k} ({v[k]['size_mib']:.0f})" for k in order], fontsize=8.4, rotation=38,
                 ha="right", rotation_mode="anchor")
    a.set_xlim(-0.45, len(order) - 0.55)
    a.set_ylim(54.2, 58.6)
    a.set_yticks([55, 56, 57, 58])
    a.set_xlabel("File type (size in MiB)")
    a.set_ylabel("MLM accuracy (%)")
    panel(a, "(a) MLM accuracy")
    headline(a, 0.97, 0.36, f"{100 * acc[i] / ref:.1f}%",
             f"of fp32 accuracy at {ours},\n{v['F32']['size_mib'] / v[ours]['size_mib']:.1f}\u00d7 smaller file")

    # (b) CPU latency against other encoders run in PyTorch
    lens = sorted(int(k.split("x")[0]) for k in cpu[f"GGUF {ours}"]["ms"])
    g = [cpu[f"GGUF {ours}"]["ms"][f"{T}x1"] for T in lens]
    base = {k: [cpu[BENCH_KEY[k]]["ms"][f"{T}x1"] for T in lens] for k in BASELINES if BENCH_KEY[k] in cpu}
    for k, ys in base.items():
        draw(b, lens, ys, base_style(k))
    draw(b, lens, g, NEXTERA)
    # speed-up at the longest length beside each line's end, kept a text height apart (log axis)
    y_text = None
    for k, ys in sorted(base.items(), key=lambda kv: kv[1][-1]):
        y_text = ys[-1] if y_text is None else max(ys[-1], y_text * 1.38)
        b.text(lens[-1] * 1.13, y_text, f"{ys[-1] / g[-1]:.1f}\u00d7", va="center", fontsize=10.5,
               color=BASELINES[k]["color"])
    b.set_xscale("log", base=2)
    b.set_yscale("log")
    b.set_xticks(lens, [f"{T:,}" for T in lens])
    b.set_xlim(lens[0] / 1.25, lens[-1] * 1.75)
    b.set_ylim(10, 480)
    b.set_yticks([10, 30, 100, 300], ["10", "30", "100", "300"])
    b.minorticks_off()
    b.set_xlabel("Input length (tokens; log)")
    b.set_ylabel("CPU latency (ms; log)")
    panel(b, "(b) Speed (CPU)")
    ratios = [y / x for ys in base.values() for y, x in zip(ys, g)]
    headline(b, 0.97, 0.36, f"{min(ratios):.1f}\u2013{max(ratios):.1f}\u00d7",
             "faster than the PyTorch\nbaselines on the same CPU")

    model_legend(fig, f"NexteraBERT (GGUF {ours})", list(base), y=0.99, ncol=2)
    fig.savefig(out, dpi=200)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", type=Path, default=ROOT / "results")
    ap.add_argument("--out", type=Path, default=ROOT / "assets")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    def load(name):
        return json.loads((args.results / name).read_text(encoding="utf-8"))

    plot_summary(load("verify.json"), load("verify_mlm.json"), load("bench_cpu.json"), args.out / "summary.png")
    print(f"wrote {args.out / 'summary.png'}")


if __name__ == "__main__":
    main()
