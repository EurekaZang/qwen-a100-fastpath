#!/usr/bin/env python3
"""Render every figure of the report from data/measurements.json (nothing is typed in by hand here).

    .venv/bin/python report/figures/make_figures.py

Style follows the reference data-viz palette: categorical slots in fixed order, emphasis form (accent blue + gray)
for baseline-vs-ours, thin bars rounded only at the data end, 2px surface gaps, hairline solid grid, legends for
>= 2 series plus selective direct labels, text in ink tokens (never series colors). Fonts: Linux Biolinum, the
sans of the acmart class, so figures match the body text (falls back to Helvetica Neue).
"""
import glob, json, math, os
import matplotlib
matplotlib.use("pdf")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.path import Path
from matplotlib.patches import PathPatch
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator

HERE = os.path.dirname(os.path.abspath(__file__))
D = json.load(open(os.path.join(HERE, "..", "..", "data", "measurements.json")))

# ---- tokens ------------------------------------------------------------------------------------
SURFACE = "#ffffff"
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS = "#e1e0d9", "#c3c2b7"
BLUE, ORANGE, AQUA, YELLOW, MAGENTA = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"
GRAY = "#898781"             # de-emphasis (baseline) in the emphasis form
DARK = "#52514e"             # neutral single series that is not an identity
GOOD, CRITICAL = "#0ca30c", "#d03b3b"
COL, FULL = 3.33, 7.0        # acmtog column / text width (in)
PX = 1 / 96 * 72             # one CSS px in points

def setup_fonts():
    cands = glob.glob(os.path.expanduser("~/Library/Caches/TectonicProject.Tectonic/bundles/data/*/LinBiolinum_R*.otf"))
    cands += glob.glob("/usr/share/texlive/texmf-dist/fonts/opentype/public/libertine/LinBiolinum_R*.otf")
    for f in cands:
        font_manager.fontManager.addfont(f)
    fam = "Linux Biolinum O" if any("LinBiolinum_R.otf" in c for c in cands) else "Helvetica Neue"
    names = {f.name for f in font_manager.fontManager.ttflist}
    if fam not in names:
        fam = next((n for n in names if "Biolinum" in n), "Helvetica Neue")
    plt.rcParams.update({
        "font.family": fam, "font.size": 7, "axes.titlesize": 7.5, "axes.labelsize": 7, "xtick.labelsize": 6.5,
        "ytick.labelsize": 6.5, "legend.fontsize": 6.5, "axes.edgecolor": AXIS, "axes.linewidth": 0.6,
        "xtick.color": INK2, "ytick.color": INK2, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "xtick.major.size": 2.5, "ytick.major.size": 0, "xtick.minor.size": 0, "ytick.minor.size": 0,
        "axes.labelcolor": INK2, "text.color": INK, "axes.titlecolor": INK, "axes.titleweight": "bold",
        "axes.titlelocation": "left", "axes.titlepad": 5, "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE, "legend.frameon": False, "legend.handlelength": 1.4, "legend.borderaxespad": 0.2,
        "pdf.fonttype": 3, "lines.solid_capstyle": "round", "lines.solid_joinstyle": "round",
    })
    return fam

def style(ax, grid="y"):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    if grid in ("y", "both"):
        ax.yaxis.grid(True, color=GRID, linewidth=0.5, linestyle="-")
    if grid in ("x", "both"):
        ax.xaxis.grid(True, color=GRID, linewidth=0.5, linestyle="-")
    ax.set_axisbelow(True)
    if grid == "x":
        ax.spines["left"].set_visible(False)
        ax.tick_params(axis="y", length=0)
    if grid == "y":
        ax.tick_params(axis="x", length=2.5)

def panel(ax, letter, title):
    ax.set_title(f"({letter})  {title}", loc="left")

def rbar(ax, pos, start, end, thick, color, horizontal=True, r_px=3, round_end=True):
    """Bar from `start` to `end` along the value axis, square at the start, rounded data-end (r_px CSS px).
    Must be called after the axes limits and figure size are final (radius is converted to data units)."""
    fig = ax.figure; fig.canvas.draw_idle()
    bb = ax.get_window_extent()
    dpi = fig.dpi
    x0, x1 = ax.get_xlim(); y0, y1 = ax.get_ylim()
    px = r_px * dpi / 96
    rx = px * abs(x1 - x0) / bb.width; ry = px * abs(y1 - y0) / bb.height
    if horizontal:
        a, b, c0, c1, ra, rc = start, end, pos - thick / 2, pos + thick / 2, rx, ry
    else:
        a, b, c0, c1, ra, rc = start, end, pos - thick / 2, pos + thick / 2, ry, rx
    sign = 1 if b >= a else -1
    ra = min(ra, abs(b - a)); rc = min(rc, (c1 - c0) / 2)
    if not round_end:
        ra = rc = 0
    pts = [(a, c0), (b - sign * ra, c0)]
    n = 8
    for i in range(n + 1):              # corner (b, c0)
        t = -math.pi / 2 + (math.pi / 2) * i / n
        pts.append((b - sign * ra + sign * ra * math.cos(t), c0 + rc + rc * math.sin(t)))
    for i in range(n + 1):              # corner (b, c1)
        t = (math.pi / 2) * i / n
        pts.append((b - sign * ra + sign * ra * math.cos(t), c1 - rc + rc * math.sin(t)))
    pts += [(a, c1), (a, c0)]
    if not horizontal:
        pts = [(y, x) for x, y in pts]
    codes = [Path.MOVETO] + [Path.LINETO] * (len(pts) - 2) + [Path.CLOSEPOLY]
    ax.add_patch(PathPatch(Path(pts, codes), facecolor=color, edgecolor="none", lw=0))

def line(ax, x, y, color, label=None, marker="o", ms=4.2, lw=1.5, z=3, **kw):
    ax.plot(x, y, color=color, lw=lw, marker=marker, ms=ms, mfc=color, mec=SURFACE, mew=1.1, label=label,
            zorder=z, solid_capstyle="round", **kw)

def legend_handles(items):
    return [Line2D([0], [0], color=c, lw=0, marker="s", ms=5.5, label=l) for l, c in items]

def save(fig, name):
    out = os.path.join(HERE, name)
    fig.savefig(out, bbox_inches="tight", pad_inches=0.02)
    if name == "fig_teaser.pdf":           # raster copy for the README
        fig.savefig(os.path.join(HERE, "fig_teaser.png"), dpi=200, bbox_inches="tight", pad_inches=0.04)
    if os.environ.get("PREVIEW_DIR"):      # optional raster previews for checking layout
        fig.savefig(os.path.join(os.environ["PREVIEW_DIR"], name.replace(".pdf", ".png")), dpi=230,
                    bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print("wrote", os.path.relpath(out))

def mean(*v):
    return sum(v) / len(v)

# ---- derived numbers (all from the JSON) --------------------------------------------------------
dec = D["decode"]
def long_mean(cfg):
    return mean(dec[cfg]["long_cold"], dec[cfg]["long_warm"])
LIN = D["linear_kernels"]
SH = LIN["shapes"]
def per_forward_ms(table):
    return sum(table[s] * SH[s]["count"] for s in SH) / 1000.0
NS = D["nsys_38k_decode"]
steps = NS["steps"]
comp = {k: v / steps for k, v in NS["kernels_ms"].items()}
ATT = D["attention_kernel"]
ratio_att_38k = ATT["ours_Q3"][ATT["ctx"].index(38000)] / ATT["fa2_Q3"][ATT["ctx"].index(38000)]
ratio_lin = per_forward_ms(LIN["us_M3"]["ours_w8a16"]) / per_forward_ms(LIN["us_M3"]["cutlass_ref"])

# =================================================================================================
def fig_teaser():
    fig, axs = plt.subplots(1, 3, figsize=(FULL, 2.05), gridspec_kw={"width_ratios": [1.05, 1.15, 1.0], "wspace": 0.42})
    # (a) decode speed, baseline vs ours
    ax = axs[0]
    rows = [("Story (EN)", dec["baseline"]["story"], dec["greedy_k4"]["story"]),
            ("Code", dec["baseline"]["code"], dec["greedy_k4"]["code"]),
            ("Essay (ZH)", dec["baseline"]["zh"], dec["greedy_k4"]["zh"]),
            ("38K context", long_mean("baseline"), long_mean("greedy_k4"))]
    ax.set_xlim(0, 165); ax.set_ylim(-1.15, len(rows) - 0.4); ax.invert_yaxis()
    style(ax, "x")
    th = 0.30
    for i, (lab, b, o) in enumerate(rows):
        rbar(ax, i - 0.17, 0, b, th, GRAY)
        rbar(ax, i + 0.17, 0, o, th, BLUE)
        ax.text(b + 2, i - 0.17, f"{b:.0f}", va="center", ha="left", fontsize=6, color=INK2)
        ax.text(o + 2, i + 0.17, f"{o:.0f}", va="center", ha="left", fontsize=6, color=INK)
        ax.text(o + 18, i + 0.17, f"{o / b:.2f}×", va="center", ha="left", fontsize=6.5, color=INK, fontweight="bold")
    ax.set_yticks(range(len(rows))); ax.set_yticklabels([r[0] for r in rows])
    ax.set_xticks([0, 50, 100, 150])
    ax.set_xlabel("decode throughput, single stream (tok/s)")
    ax.legend(handles=legend_handles([("stock vLLM", GRAY), ("this work", BLUE)]), loc="upper left",
              bbox_to_anchor=(0.0, 1.0), ncol=2, columnspacing=1.0, handletextpad=0.3)
    panel(ax, "a", "End-to-end decode speed")
    # (b) per-step time at 38K
    ax = axs[1]
    order = list(NS["kernels_ms"].keys())
    colors = [BLUE, ORANGE, AQUA, YELLOW, MAGENTA]
    base = [comp[k] for k in order]
    proj = [comp[order[0]] * ratio_att_38k, comp[order[1]] * ratio_lin, comp[order[2]], 0.0, comp[order[4]]]
    acc = 2.04
    measured = acc / long_mean("attn_linear_k2") * 1000
    XMAX = 98
    ax.set_xlim(0, XMAX); ax.set_ylim(-0.6, 2.6); ax.invert_yaxis()
    style(ax, "x")
    gap = 2 * PX  # surface gap in points -> converted below
    fig.canvas.draw()
    bbw = ax.get_window_extent().width
    gap_d = (2 * fig.dpi / 96) * XMAX / bbw
    for row, vals in ((0, base), (1, proj)):
        x = 0.0
        last = max(i for i, v in enumerate(vals) if v > 0)
        for i, (v, c) in enumerate(zip(vals, colors)):
            if v <= 0:
                continue
            s = x + (gap_d / 2 if i else 0); e = x + v - (gap_d / 2 if i != last else 0)
            rbar(ax, row, s, e, 0.42, c, round_end=(i == last))
            x += v
        ax.text(x + 1.2, row, f"{x:.1f} ms", va="center", ha="left", fontsize=6.5, color=INK)
    rbar(ax, 2, 0, measured, 0.42, DARK)
    ax.text(measured + 1.2, 2, f"{measured:.1f} ms", va="center", ha="left", fontsize=6.5, color=INK)
    ax.set_yticks([0, 1, 2]); ax.set_yticklabels(["stock (nsys)", "this work\n(projected)", "this work\n(measured)"])
    ax.set_xlabel("GPU time per verification step at 38K context (ms)")
    labs = ["attention", "INT8 GEMM", "bf16 GEMM\n(lm_head, MTP)", "act. quant", "other"]
    ax.set_xticks([0, 20, 40, 60])
    ax.legend(handles=legend_handles(list(zip(labs, colors))), loc="lower right", bbox_to_anchor=(1.03, 0.0),
              ncol=1, fontsize=5.6, handletextpad=0.3, labelspacing=0.3)
    panel(ax, "b", "Where a decode step goes")
    # (c) agent loop TTFT
    ax = axs[2]
    A = D["agent_session"]
    xs = A["steps"][1:]
    style(ax, "y")
    ax.set_yscale("log"); ax.set_ylim(2, 90); ax.set_xlim(0.6, 6.4)
    ax.yaxis.set_major_locator(FixedLocator([2, 5, 10, 20, 50]))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}")); ax.yaxis.set_minor_locator(NullLocator())
    line(ax, xs, A["merged_ttft_s"][1:], GRAY, "system prompt rewritten each step")
    line(ax, xs, A["inline_ttft_s"][1:], BLUE, "reminders rendered in place")
    tot_m, tot_i = sum(A["merged_ttft_s"][1:]), sum(A["inline_ttft_s"][1:])
    ax.text(6.35, A["merged_ttft_s"][6] * 1.17, f"prefix hit {min(A['merged_hit'][2:]):.0%}–{max(A['merged_hit'][2:]):.0%}",
            ha="right", va="bottom", fontsize=6, color=INK2)
    ax.text(6.35, A["inline_ttft_s"][6] * 1.17, f"prefix hit {min(A['inline_hit'][1:]):.0%}–{max(A['inline_hit'][1:]):.0%}",
            ha="right", va="bottom", fontsize=6, color=INK2)
    ax.text(0.7, 3.0, f"steps 1–6: {tot_m:.0f} s → {tot_i:.0f} s ({tot_m / tot_i:.1f}×)", fontsize=6.5, color=INK, fontweight="bold")
    ax.set_xticks(xs); ax.set_xlabel("agent tool step (41–48K-token prompt)")
    ax.set_ylabel("time to first token (s, log)")
    ax.legend(loc="center left", bbox_to_anchor=(0.0, 0.52), fontsize=5.8, handlelength=1.6)
    panel(ax, "c", "Claude Code agent loop")
    save(fig, "fig_teaser.pdf")

# =================================================================================================
def fig_attention():
    fig, axs = plt.subplots(1, 3, figsize=(FULL, 1.95), gridspec_kw={"width_ratios": [1.1, 1.1, 1.0], "wspace": 0.45})
    ctx = ATT["ctx"]
    # (a) latency
    ax = axs[0]
    keep = [i for i, c in enumerate(ctx) if c >= 1]          # ctx = 0 sits on top of ctx = 1; it stays in the table
    xs = [ctx[i] for i in keep]
    pick = lambda k: [ATT[k][i] for i in keep]
    style(ax, "both")
    ax.set_xscale("log"); ax.set_yscale("log")
    line(ax, xs, pick("fa2_Q3"), GRAY, "FA-2 varlen")
    line(ax, xs, pick("ours_Q3"), BLUE, "ours")
    line(ax, xs, pick("ours_Q3_B2"), AQUA, "ours, 2 req.", ms=3.4, lw=1.1)
    ax.set_xlim(0.7, 6e5); ax.set_ylim(4, 8000)
    ax.xaxis.set_major_locator(FixedLocator([1, 10, 100, 1e3, 1e4, 1e5]))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: {1: "≤1", 10: "10", 100: "100", 1e3: "1K", 1e4: "10K", 1e5: "100K"}.get(v, "")))
    ax.yaxis.set_major_locator(FixedLocator([10, 100, 1000]))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.xaxis.set_minor_locator(NullLocator()); ax.yaxis.set_minor_locator(NullLocator())
    for c, dx, ha in ((100000, 1.25, "left"),):
        i = ctx.index(c)
        ax.text(c * dx, math.sqrt(ATT["fa2_Q3"][i] * ATT["ours_Q3"][i]), f"{ATT['fa2_Q3'][i] / ATT['ours_Q3'][i]:.1f}×", ha=ha, va="center",
                fontsize=6.5, color=INK, fontweight="bold")
    ax.set_xlabel("cached context (tokens)"); ax.set_ylabel("latency per layer (µs, log)")
    ax.legend(loc="upper left", bbox_to_anchor=(0.0, 1.0), fontsize=5.8)
    panel(ax, "a", "One full-attention layer, Q = 3")
    # (b) effective bandwidth
    ax = axs[1]
    kvb = D["model"]["kv_bytes_per_token_full_attn_layer"]
    sel = [i for i, c in enumerate(ctx) if c >= 1000]
    xs = [ctx[i] for i in sel]
    bw = lambda t, c: kvb * c / (t * 1e-6) / 1e9
    style(ax, "y")
    ax.set_xscale("log"); ax.set_ylim(0, 1900); ax.set_xlim(700, 1.5e5)
    hb = D["hardware"]["copy_bandwidth_gbs"]
    ax.axhline(hb, color=AXIS, lw=0.8, zorder=1)
    ax.text(760, hb + 40, f"device copy bandwidth {hb} GB/s", fontsize=6, color=INK2, va="bottom")
    line(ax, xs, [bw(ATT["fa2_Q3"][i], ctx[i]) for i in sel], GRAY, "FA-2 varlen")
    line(ax, xs, [bw(ATT["ours_Q3"][i], ctx[i]) for i in sel], BLUE, "ours")
    i = ctx.index(100000)
    ax.text(1e5, bw(ATT["ours_Q3"][i], 1e5) - 140, f"{bw(ATT['ours_Q3'][i], 1e5):,.0f}", ha="center", va="top", fontsize=6.5, color=INK)
    ax.text(1e5, bw(ATT["fa2_Q3"][i], 1e5) + 70, f"{bw(ATT['fa2_Q3'][i], 1e5):,.0f}", ha="center", va="bottom", fontsize=6.5, color=INK2)
    ax.xaxis.set_major_locator(FixedLocator([1e3, 1e4, 1e5]))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: {1e3: "1K", 1e4: "10K", 1e5: "100K"}[v]))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}"))
    ax.set_xlabel("cached context (tokens)"); ax.set_ylabel("KV bytes read once / latency (GB/s)")
    ax.legend(loc="upper left", bbox_to_anchor=(0.0, 0.88), fontsize=5.8)
    panel(ax, "b", "Effective KV-cache bandwidth")
    # (c) end-to-end by backend
    ax = axs[2]
    rows = [("FA-2", long_mean("baseline"), GRAY), ("Triton", long_mean("triton_attn"), GRAY),
            ("FlashInfer*", long_mean("flashinfer_attn"), GRAY), ("ours", long_mean("attn_only"), BLUE)]
    ax.set_xlim(0, 80); ax.set_ylim(-0.6, len(rows) - 0.4); ax.invert_yaxis()
    style(ax, "x")
    for i, (lab, v, c) in enumerate(rows):
        rbar(ax, i, 0, v, 0.46, c)
        ax.text(v + 1.5, i, f"{v:.1f}", va="center", fontsize=6.5, color=INK if c == BLUE else INK2)
    ax.set_yticks(range(len(rows))); ax.set_yticklabels([r[0] for r in rows])
    ax.set_xlabel("decode at 38K context (tok/s)")
    panel(ax, "c", "End-to-end, by attention backend")
    save(fig, "fig_attention.pdf")

# =================================================================================================
def fig_linear():
    fig, axs = plt.subplots(1, 3, figsize=(FULL, 1.9), gridspec_kw={"width_ratios": [1.0, 1.0, 1.15], "wspace": 0.75})
    names = {"bf16_cublas": "bf16 (cuBLAS)", "cutlass_w8a8": "CUTLASS W8A8", "triton_w8a8": "Triton W8A8",
             "marlin_w8a16": "Marlin W8A16", "ours_w8a16": "ours W8A16"}
    # (a) decode
    ax = axs[0]
    keys = ["bf16_cublas", "cutlass_w8a8", "triton_w8a8", "marlin_w8a16", "ours_w8a16"]
    vals = [per_forward_ms(LIN["us_M3"][k]) for k in keys]
    ax.set_xlim(0, 38); ax.set_ylim(-0.6, len(keys) - 0.4); ax.invert_yaxis()
    style(ax, "x")
    for i, (k, v) in enumerate(zip(keys, vals)):
        c = BLUE if k == "ours_w8a16" else GRAY
        rbar(ax, i, 0, v, 0.5, c)
        ax.text(v + 0.6, i, f"{v:.1f}", va="center", fontsize=6.5, color=INK if c == BLUE else INK2)
    ax.set_yticks(range(len(keys))); ax.set_yticklabels([names[k] for k in keys])
    ax.set_xlabel("GEMM time per forward pass (ms)")
    panel(ax, "a", "Decode, M = 3 tokens")
    # (b) prefill
    ax = axs[1]
    keys2 = ["bf16_cublas", "cutlass_w8a8", "triton_w8a8", "marlin_w8a16"]
    vals2 = [per_forward_ms(LIN["us_M2048"][k]) for k in keys2]
    ax.set_xlim(0, 760); ax.set_ylim(-0.6, len(keys2) - 0.4); ax.invert_yaxis()
    style(ax, "x")
    for i, (k, v) in enumerate(zip(keys2, vals2)):
        c = BLUE if k == "cutlass_w8a8" else GRAY
        rbar(ax, i, 0, v, 0.5, c)
        ax.text(v + 12, i, f"{v:.0f}", va="center", fontsize=6.5, color=INK if c == BLUE else INK2)
    ax.set_yticks(range(len(keys2))); ax.set_yticklabels([names[k] + (" †" if k == "cutlass_w8a8" else "") for k in keys2])
    ax.set_xlabel("GEMM time per forward pass (ms)")
    panel(ax, "b", "Prefill chunk, M = 2048")
    # (c) error
    ax = axs[2]
    E = LIN["rel_error_M1"]
    shapes = list(E["w8a8_cutlass"].keys())
    style(ax, "x")
    ax.set_xlim(0, 0.0118); ax.set_ylim(-1.35, len(shapes) - 0.4); ax.invert_yaxis()
    for i, s in enumerate(shapes):
        a, b = E["w8a8_cutlass"][s], E["w8a16_ours"][s]
        ax.plot([b, a], [i, i], color=GRID, lw=1.5, zorder=1, solid_capstyle="round")
        ax.plot([a], [i], "o", color=GRAY, ms=4.6, mec=SURFACE, mew=1.1, zorder=3)
        ax.plot([b], [i], "o", color=BLUE, ms=4.6, mec=SURFACE, mew=1.1, zorder=3)
    ax.set_yticks(range(len(shapes))); ax.set_yticklabels(shapes)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v * 100:.1f}%"))
    ax.xaxis.set_major_locator(FixedLocator([0, 0.004, 0.008]))
    ax.set_xlabel("relative error vs exact product")
    ax.legend(handles=[Line2D([0], [0], color=c, lw=0, marker="o", ms=4.6, label=l) for l, c in
                       (("CUTLASS W8A8", GRAY), ("ours W8A16", BLUE))], loc="upper left", bbox_to_anchor=(0.0, 1.0),
              ncol=2, fontsize=5.8, columnspacing=1.0, handletextpad=0.2)
    panel(ax, "c", "Numerical error, same int8 weights")
    save(fig, "fig_linear.pdf")

# =================================================================================================
def fig_spec_accept():
    fig, axs = plt.subplots(1, 3, figsize=(COL, 1.55), sharey=True, gridspec_kw={"wspace": 0.12})
    wl = [("story", "Story (EN)"), ("code", "Code"), ("zh", "Essay (ZH)")]
    for ax, (k, title) in zip(axs, wl):
        style(ax, "y")
        g = [dec["attn_linear_k2"]["acc"][k], dec["greedy_k3"]["acc"][k], dec["greedy_k4"]["acc"][k]]
        p = [dec["prob_k3"]["acc"][k], dec["prob_k4"]["acc"][k]]
        line(ax, [2, 3, 4], g, GRAY, "greedy drafts")
        line(ax, [3, 4], p, BLUE, "probabilistic drafts")
        ax.text(4.12, p[-1], f"{p[-1]:.2f}", va="center", fontsize=6, color=INK)
        ax.text(4.12, g[-1], f"{g[-1]:.2f}", va="center", fontsize=6, color=INK2)
        ax.set_xlim(1.7, 4.75); ax.set_ylim(1.5, 4.3); ax.set_xticks([2, 3, 4])
        ax.set_title(title, loc="left", fontsize=7)
        ax.set_xlabel("draft tokens k")
    axs[0].set_ylabel("tokens per verification step")
    axs[0].legend(loc="upper left", fontsize=5.6, handlelength=1.4)
    save(fig, "fig_spec_accept.pdf")

def fig_k_sweep():
    fig, axs = plt.subplots(1, 3, figsize=(FULL, 1.75), gridspec_kw={"wspace": 0.38})
    W = [("story", "Story (EN)", BLUE), ("code", "Code", ORANGE), ("zh", "Essay (ZH)", AQUA)]
    ax = axs[0]; style(ax, "y")
    for k, lab, c in W:
        ys = [dec["attn_linear_k2"][k], dec["greedy_k3"][k], dec["greedy_k4"][k]]
        line(ax, [2, 3, 4], ys, c, lab)
        ax.text(4.1, ys[-1], f"{ys[-1]:.0f}", va="center", fontsize=6.5, color=INK)
    ax.set_xlim(1.8, 4.5); ax.set_ylim(35, 135); ax.set_xticks([2, 3, 4])
    ax.set_xlabel("draft tokens k (greedy)"); ax.set_ylabel("decode (tok/s)")
    ax.legend(loc="lower right", fontsize=5.8, ncol=3, columnspacing=0.8, handletextpad=0.3)
    panel(ax, "a", "Idle GPU, all fast paths")
    S = D["first_deployment_k_sweep"]
    ax = axs[1]; style(ax, "y")
    for k, lab, c in W[:2]:
        line(ax, S["k"], S[k], c, lab)
        ax.text(3.1, S[k][-1], f"{S[k][-1]:.0f}", va="center", fontsize=6.5, color=INK)
    ax.set_xlim(-0.3, 3.5); ax.set_ylim(0, 60); ax.set_xticks(S["k"])
    ax.set_xlabel("draft tokens k"); ax.set_ylabel("decode, single stream (tok/s)")
    ax.legend(loc="upper left", fontsize=5.8)
    panel(ax, "b", "Time-shared GPU, stock kernels")
    ax = axs[2]; style(ax, "y")
    line(ax, S["k"], S["agg8"], DARK)
    best = max(range(len(S["k"])), key=lambda i: S["agg8"][i])
    ax.text(S["k"][best], S["agg8"][best] + 8, f"peak {S['agg8'][best]} at k={S['k'][best]}", ha="center", fontsize=6.5, color=INK)
    ax.set_xlim(-0.3, 3.3); ax.set_ylim(0, 260); ax.set_xticks(S["k"])
    ax.set_xlabel("draft tokens k"); ax.set_ylabel("aggregate, 8 streams (tok/s)")
    panel(ax, "c", "Same GPU, 8 concurrent streams")
    save(fig, "fig_k_sweep.pdf")

# =================================================================================================
def fig_prefix():
    A = D["agent_session"]
    fig, ax = plt.subplots(figsize=(COL, 1.55))
    style(ax, "y")
    xs = A["steps"]
    line(ax, xs, [h * 100 for h in A["merged_hit"]], GRAY, "merged into system prompt (stock)")
    line(ax, xs, [h * 100 for h in A["inline_hit"]], BLUE, "rendered in place (patched template)")
    ax.set_ylim(-4, 104); ax.set_xlim(-0.3, 6.3); ax.set_xticks(xs)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0f}%"))
    ax.set_xlabel("agent step"); ax.set_ylabel("prefix-cache hit rate")
    ax.legend(loc="center right", bbox_to_anchor=(1.0, 0.64), fontsize=5.6)
    save(fig, "fig_prefix.pdf")

def fig_ablation():
    fig, ax = plt.subplots(figsize=(COL, 1.85))
    style(ax, "y")
    cfgs = ["baseline", "attn_only", "attn_linear_k2", "greedy_k4"]
    xl = ["stock\nk=2", "+ split-KV\nattention", "+ W8A16\ndecode GEMM", "+ k=4, int8\ndraft head"]
    W = [("story", "Story (EN)", BLUE), ("code", "Code", ORANGE), ("zh", "Essay (ZH)", AQUA), ("long", "38K context", YELLOW)]
    for k, lab, c in W:
        ys = [long_mean(cf) if k == "long" else dec[cf][k] for cf in cfgs]
        line(ax, range(4), ys, c, lab)
        if k == "code":
            ax.text(3.12, ys[-1], f"{ys[-1]:.0f}", va="center", fontsize=6.5, color=INK)
        if k == "long":     # the 38K curve is the story of this figure: label both ends
            ax.text(0, ys[0] - 4, f"{ys[0]:.0f}", ha="center", va="top", fontsize=6.5, color=INK)
            ax.text(3, ys[-1] - 4, f"{ys[-1]:.0f} ({ys[-1] / ys[0]:.1f}×)", ha="center", va="top", fontsize=6.5, color=INK)
    ax.set_xticks(range(4)); ax.set_xticklabels(xl, fontsize=6)
    ax.set_xlim(-0.25, 3.45); ax.set_ylim(20, 135)
    ax.set_ylabel("decode (tok/s)")
    ax.legend(loc="upper left", fontsize=5.6, ncol=2, columnspacing=0.8)
    save(fig, "fig_ablation.pdf")

def fig_contention():
    C = D["contention"]
    fig, ax = plt.subplots(figsize=(COL, 1.45))
    rows = [("Story (EN)", "story"), ("Code", "code"), ("Essay (ZH)", "zh"), ("38K context", "long")]
    idle_long = long_mean("greedy_k4"); cont_long = long_mean("prob_k4")
    ax.set_xlim(0, 150); ax.set_ylim(-1.15, len(rows) - 0.4); ax.invert_yaxis()
    style(ax, "x")
    for i, (lab, k) in enumerate(rows):
        a = idle_long if k == "long" else C["idle"][k]
        b = cont_long if k == "long" else C["contended"][k]
        rbar(ax, i - 0.17, 0, a, 0.30, BLUE)
        rbar(ax, i + 0.17, 0, b, 0.30, ORANGE)
        ax.text(a + 2, i - 0.17, f"{a:.0f}", va="center", fontsize=6, color=INK)
        ax.text(b + 2, i + 0.17, f"{b:.1f}", va="center", fontsize=6, color=INK2)
        ax.text(148, i, f"÷{a / b:.1f}", va="center", ha="right", fontsize=7, color=INK, fontweight="bold")
    ax.set_yticks(range(len(rows))); ax.set_yticklabels([r[0] for r in rows])
    ax.set_xlabel("decode (tok/s)")
    ax.legend(handles=legend_handles([("GPU to ourselves", BLUE), ("6 co-tenant processes", ORANGE)]),
              loc="upper left", bbox_to_anchor=(0.0, 1.0), ncol=2, fontsize=5.8, columnspacing=1.0, handletextpad=0.3)
    save(fig, "fig_contention.pdf")

def fig_tunnel():
    T = D["tunnel_recovery"]
    fig, ax = plt.subplots(figsize=(COL, 0.95))
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.set_yticks([]); ax.set_ylim(-1, 1.6); ax.set_xlim(-1.5, 64)
    for t, ok in zip(T["t_s"], T["ok"]):
        if ok:
            ax.plot([t], [0], "o", color=GOOD, ms=5, mec=SURFACE, mew=1.1, zorder=3)
        else:
            ax.plot([t], [0], "X", color=CRITICAL, ms=6, mec=SURFACE, mew=0.8, zorder=3)
    for x, lab in ((T["kill_t_s"], "ssh killed"), (T["reconnect_t_s"], "watchdog reconnects")):
        ax.axvline(x, color=AXIS, lw=0.8, zorder=1)
    ax.text(T["kill_t_s"] - 0.6, 0.95, "tunnel ssh killed", ha="right", fontsize=6, color=INK2)
    ax.text(T["reconnect_t_s"] + 0.6, 0.95, "watchdog reopens tunnel", ha="left", fontsize=6, color=INK2)
    ax.set_xlabel("time (s)")
    ax.legend(handles=[Line2D([0], [0], color=GOOD, lw=0, marker="o", ms=5, label="200 OK"),
                       Line2D([0], [0], color=CRITICAL, lw=0, marker="X", ms=5.5, label="connection refused")],
              loc="upper right", bbox_to_anchor=(1.0, 1.25), ncol=2, fontsize=5.8)
    save(fig, "fig_tunnel.pdf")

if __name__ == "__main__":
    print("font:", setup_fonts())
    fig_teaser(); fig_attention(); fig_linear(); fig_spec_accept(); fig_k_sweep()
    fig_prefix(); fig_ablation(); fig_contention(); fig_tunnel()
    print(f"derived: attention ratio at 38K {ratio_att_38k:.3f}, W8A16/CUTLASS per-forward {ratio_lin:.3f}")
