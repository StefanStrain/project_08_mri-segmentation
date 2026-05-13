"""Generate a clean architectural diagram of the 3D U-Net and save to docs/figures/.

Usage (from project root, with venv active):
    python scripts/visualize_model.py

Outputs:
    docs/figures/unet3d_architecture.png  -- high-res diagram
    docs/figures/unet3d_summary.txt       -- torchinfo text summary (if torchinfo installed)
"""
import sys
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

# ── colours ───────────────────────────────────────────────────────────────────
C_INPUT   = "#AED6F1"
C_ENC     = "#2980B9"
C_BOTTLE  = "#1A5276"
C_DEC     = "#E67E22"
C_OUTPUT  = "#A9DFBF"
C_BG      = "#F4F6F7"
C_POOL    = "#717D7E"
C_UP      = "#8E44AD"
C_SKIP    = "#27AE60"

# ── architecture spec ─────────────────────────────────────────────────────────
IN_CH  = 4
OUT_CH = 3
FEATS  = [32, 64, 128, 256]
BOTTLE = FEATS[-1] * 2        # 512
PATCH  = 128

def sp(depth):
    return PATCH // (2 ** depth)

# ── layout: left col = encoders top-to-bottom, right col = decoders bottom-to-top
#    bottleneck at bottom centre
N       = len(FEATS)            # 4
BW      = 1.8                   # box width
BH      = 0.60                  # box height
XENC    = 1.5                   # encoder column centre-x
XDEC    = 9.5                   # decoder column centre-x
XBOT    = (XENC + XDEC) / 2    # bottleneck centre-x
YSTEP   = 2.0                   # vertical step between levels
YTOP    = 9.0                   # y of level-0 (shallowest)

# level i  ->  y = YTOP - i * YSTEP
enc_y   = [YTOP - i * YSTEP for i in range(N)]
dec_y   = list(reversed(enc_y))          # decoder mirrors encoder levels
bot_y   = enc_y[-1] - YSTEP             # below deepest encoder

FIG_W   = 13.0
FIG_H   = 12.5


def box(ax, cx, cy, title, sub, color, fs=9.0):
    rect = mpatches.FancyBboxPatch(
        (cx - BW / 2, cy - BH / 2), BW, BH,
        boxstyle="round,pad=0.06",
        linewidth=1.3, edgecolor="black",
        facecolor=color, zorder=3,
    )
    ax.add_patch(rect)
    ax.text(cx, cy + 0.10, title, ha="center", va="center",
            fontsize=fs, fontweight="bold", zorder=4)
    ax.text(cx, cy - 0.18, sub, ha="center", va="center",
            fontsize=7.0, color="#2C2C2C", zorder=4)


def arr(ax, x0, y0, x1, y1, color, lw=1.3, dashed=False):
    style = "--" if dashed else "-"
    ax.annotate("",
        xy=(x1, y1), xytext=(x0, y0),
        arrowprops=dict(arrowstyle="-|>", color=color, lw=lw,
                        linestyle=style, mutation_scale=10),
        zorder=2,
    )


def make_figure():
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    ax.set_xlim(0, FIG_W)
    ax.set_ylim(bot_y - 1.5, YTOP + 2.0)
    ax.axis("off")
    ax.set_facecolor(C_BG)
    fig.patch.set_facecolor(C_BG)

    # ── input volume ──────────────────────────────────────────────────────────
    in_cx = XENC - BW - 0.6
    in_cy = enc_y[0]
    box(ax, in_cx, in_cy, "Input", f"{IN_CH}ch  {sp(0)}^3", C_INPUT, fs=8.5)
    arr(ax, in_cx + BW / 2, in_cy, XENC - BW / 2, in_cy, "black")

    # ── encoder levels ────────────────────────────────────────────────────────
    for i in range(N):
        cy   = enc_y[i]
        ch   = FEATS[i]
        spc  = sp(i)
        box(ax, XENC, cy,
            f"Encoder {i+1}",
            f"DoubleConv  {ch}ch  {spc}^3",
            C_ENC)

        # MaxPool arrow down
        next_y = enc_y[i + 1] if i < N - 1 else bot_y
        mid_y  = (cy - BH / 2 + next_y + BH / 2) / 2
        arr(ax, XENC, cy - BH / 2, XENC, next_y + BH / 2, C_POOL, lw=1.2)
        ax.text(XENC + 0.14, mid_y, "MaxPool 2x",
                fontsize=6.5, color=C_POOL, ha="left", va="center")

    # ── bottleneck ────────────────────────────────────────────────────────────
    box(ax, XBOT, bot_y,
        "Bottleneck",
        f"DoubleConv  {BOTTLE}ch  {sp(N)}^3",
        C_BOTTLE, fs=9.0)

    # ── decoder levels ────────────────────────────────────────────────────────
    for i in range(N):
        cy     = dec_y[i]
        depth  = N - 1 - i         # 3 -> 0
        ch     = FEATS[depth]
        spc    = sp(depth)
        box(ax, XDEC, cy,
            f"Decoder {i+1}",
            f"DoubleConv  {ch}ch  {spc}^3",
            C_DEC)

        # ConvTranspose arrow up from previous (bottleneck or decoder below)
        if i == 0:
            src_cx, src_cy = XBOT, bot_y
            # diagonal from bottleneck to first decoder
            arr(ax, src_cx + BW / 2, src_cy, XDEC - BW / 2, cy, C_UP, lw=1.4)
            mx = (src_cx + BW / 2 + XDEC - BW / 2) / 2
            my = (src_cy + cy) / 2
            ax.text(mx, my + 0.15, "ConvT 2x",
                    fontsize=6.5, color=C_UP, ha="center", va="bottom")
        else:
            prev_cy = dec_y[i - 1]
            arr(ax, XDEC, prev_cy - BH / 2, XDEC, cy + BH / 2, C_UP, lw=1.4)
            mid_y2 = (prev_cy - BH / 2 + cy + BH / 2) / 2
            ax.text(XDEC + 0.14, mid_y2, "ConvT 2x",
                    fontsize=6.5, color=C_UP, ha="left", va="center")

    # ── skip connections (horizontal, same y level) ───────────────────────────
    for i in range(N):
        enc_cx = XENC
        dec_cx = XDEC
        y      = enc_y[i]           # encoder level i  <->  decoder level (N-1-i)
        dec_level = N - 1 - i
        dec_yy = dec_y[dec_level]   # should equal enc_y[i]

        arr(ax, enc_cx + BW / 2, y, dec_cx - BW / 2, y,
            C_SKIP, lw=1.1, dashed=True)
        ax.text((enc_cx + BW / 2 + dec_cx - BW / 2) / 2, y + 0.15,
                "skip (concat)",
                fontsize=6.5, color=C_SKIP, ha="center", va="bottom")

    # ── output head ───────────────────────────────────────────────────────────
    out_cx = XDEC + BW + 0.6
    out_cy = dec_y[-1]
    box(ax, out_cx, out_cy, "Output",
        f"Conv1x1  {OUT_CH}ch  {sp(0)}^3", C_OUTPUT, fs=8.5)
    arr(ax, XDEC + BW / 2, out_cy, out_cx - BW / 2, out_cy, "black")
    ax.text(out_cx, out_cy - BH / 2 - 0.22,
            "NCR  |  ED  |  ET",
            fontsize=7.0, color="#1D8348", ha="center", va="top",
            fontstyle="italic")

    # ── channel-dim annotations alongside encoder ──────────────────────────
    ax.text(XENC - BW / 2 - 0.15, enc_y[0] + 0.05,
            f"ch: {IN_CH}", fontsize=6.5, color="#555", ha="right")
    for i, ch in enumerate(FEATS):
        ax.text(XENC - BW / 2 - 0.15, enc_y[i] - 0.15,
                f"-> {ch}", fontsize=6.5, color="#555", ha="right")
    ax.text(XENC - BW / 2 - 0.15, bot_y,
            f"-> {BOTTLE}", fontsize=6.5, color="#ccc", ha="right")

    # ── title ─────────────────────────────────────────────────────────────────
    ax.set_title(
        "3D U-Net  --  BraTS2021 Brain Tumor Segmentation\n"
        f"features={FEATS}   patch={PATCH}^3   "
        f"in={IN_CH}ch (FLAIR/T1/T1CE/T2)   out={OUT_CH}ch (NCR/ED/ET)",
        fontsize=11.5, fontweight="bold", pad=12,
    )

    # ── legend ────────────────────────────────────────────────────────────────
    legend_handles = [
        mpatches.Patch(color=C_ENC,    label="Encoder  (DoubleConv3d x2)"),
        mpatches.Patch(color=C_BOTTLE, label="Bottleneck  (512ch)"),
        mpatches.Patch(color=C_DEC,    label="Decoder  (DoubleConv3d x2)"),
        mpatches.Patch(color=C_SKIP,   label="Skip connection  (concat)"),
        mpatches.Patch(color=C_POOL,   label="MaxPool3d  /2"),
        mpatches.Patch(color=C_UP,     label="ConvTranspose3d  x2"),
    ]
    ax.legend(handles=legend_handles, loc="lower right",
              fontsize=8.5, framealpha=0.92, ncol=2,
              title="Legend", title_fontsize=9)

    return fig


def try_torchinfo_summary(out_path: Path) -> bool:
    try:
        import torchinfo
    except ImportError:
        return False

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from src.models.unet3d import UNet3D
    import torch

    model = UNet3D(in_channels=4, out_channels=3, features=[32, 64, 128, 256])
    summary = torchinfo.summary(
        model,
        input_size=(1, 4, 128, 128, 128),
        device="cpu",
        verbose=0,
    )
    out_path.write_text(str(summary))
    return True


def main():
    out_dir = Path(__file__).resolve().parent.parent / "docs" / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Generating architecture diagram...")
    fig = make_figure()
    png_path = out_dir / "unet3d_architecture.png"
    fig.savefig(png_path, dpi=180, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Saved -> {png_path}")

    txt_path = out_dir / "unet3d_summary.txt"
    if try_torchinfo_summary(txt_path):
        print(f"  Saved -> {txt_path}")
    else:
        print("  torchinfo not installed -- skipping text summary.")
        print("  To add: python -m pip install torchinfo")

    print("\nDone.")


if __name__ == "__main__":
    main()
