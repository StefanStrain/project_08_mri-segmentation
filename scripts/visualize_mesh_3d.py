"""
Generate an interactive 3D tumour mesh HTML using Plotly + Marching Cubes.

Output: docs/figures/tumor_mesh_3d.html

Usage:
  python scripts/visualize_mesh_3d.py
  python scripts/visualize_mesh_3d.py --case BraTS2021_01553
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from skimage.measure import marching_cubes
from scipy.ndimage import gaussian_filter

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.visualize_segmentation import get_val_cases, pick_best_case, NPY_DIR, OUT_DIR


def _add_mesh(fig, mask_f32, name, color, opacity, step_size=2):
    from skimage.measure import marching_cubes
    try:
        verts, faces, _, _ = marching_cubes(mask_f32, level=0.5, step_size=step_size)
        x, y, z = verts[:, 0], verts[:, 1], verts[:, 2]
        i, j, k = faces[:, 0], faces[:, 1], faces[:, 2]
        fig.add_trace(go.Mesh3d(
            x=x, y=y, z=z, i=i, j=j, k=k,
            color=color, opacity=opacity,
            name=name, legendgroup=name, showlegend=True,
            hoverinfo="name",
        ))
        print(f"  {name}: {len(verts):,} verts, {len(faces):,} faces")
    except Exception as e:
        print(f"  {name} failed: {e}")


def generate_mesh_html(case_id: str, out_path: Path):

    image = np.load(NPY_DIR / f"{case_id}_image.npy").astype(np.float32)  # (4, H, W, D)
    label = np.load(NPY_DIR / f"{case_id}_label.npy").astype(bool)         # (3, H, W, D)

    # Decode MONAI TC/WT/ET → disjoint NCR/ED/ET
    tc, wt, et = label[0], label[1], label[2]
    decoded_masks = [tc & ~et, wt & ~tc, et]

    # Brain mask: FLAIR is skull-stripped so background == 0 exactly
    # Smooth lightly before marching cubes to remove surface jaggedness
    flair = image[0]
    brain_mask = (flair > 0.05).astype(np.float32)
    brain_smooth = gaussian_filter(brain_mask, sigma=1.5)

    fig = go.Figure()

    # Brain surface — rendered first so it sits behind the tumour traces
    print("  Brain surface...")
    _add_mesh(fig, brain_smooth, "Brain", "rgb(190,185,180)", opacity=0.07, step_size=5)

    # Tumour sub-regions — rendered on top
    region_names  = ["NCR", "ED", "ET"]
    region_colors = ["rgb(220,60,60)", "rgb(220,210,30)", "rgb(30,200,240)"]
    for mask, name, color in zip(decoded_masks, region_names, region_colors):
        if mask.sum() < 100:
            print(f"  Skipping {name} — too few voxels")
            continue
        _add_mesh(fig, mask.astype(np.float32), name, color, opacity=0.75, step_size=2)

    fig.update_layout(
        title=dict(
            text=f"3D Tumour Mesh — {case_id} (Ground Truth)",
            font=dict(color="white", size=14),
        ),
        scene=dict(
            xaxis_title="", yaxis_title="", zaxis_title="",
            aspectmode="data",
            bgcolor="rgb(13,13,20)",
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False,
                       showspikes=False, visible=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False,
                       showspikes=False, visible=False),
            zaxis=dict(showgrid=False, zeroline=False, showticklabels=False,
                       showspikes=False, visible=False),
        ),
        paper_bgcolor="rgb(13,13,20)",
        plot_bgcolor="rgb(13,13,20)",
        font=dict(color="white", family="Inter, sans-serif"),
        height=560,
        margin=dict(l=0, r=0, t=40, b=0),
        legend=dict(
            x=0.01, y=0.99,
            bgcolor="rgba(0,0,0,0.4)",
            bordercolor="rgba(255,255,255,0.1)",
            borderwidth=1,
        ),
    )

    fig.write_html(out_path, include_plotlyjs="cdn")
    print(f"  Saved {out_path.name}")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--case", default=None)
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    val_cases = get_val_cases()
    case_id   = args.case or pick_best_case(val_cases)

    print(f"\nGenerating 3D mesh for {case_id}...")
    generate_mesh_html(case_id, OUT_DIR / "tumor_mesh_3d.html")
    print(f"\nDone.")


if __name__ == "__main__":
    main()
