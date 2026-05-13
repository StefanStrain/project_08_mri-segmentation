"""
    Generate interactive Plotly charts for the portfolio page.

    Outputs (in docs/figures/):
        training_curves.html  - loss + Dice curves per model over epochs (from MLflow)
        model_comparison.html - grouped bar chart: WT/TC/ET Dice across all models
        radar_comparison.html - radar/spider chart across all models
        params_vs_dice.html   - scatter: param count vs mean Dice (efficiency plot)

    Usage:
        python scripts/plot_charts.py
        python scripts/plot_charts.py --no-mlflow  # skip training curves if MLflow unavailable
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

OUT_DIR = ROOT / "docs/figures"

# update these as models finish training
RESULTS = {
    "3D U-Net": {
        "WT": 0.876, "TC": 0.877, "ET": 0.869, "mean": 0.874,
        "params_m": 22.58, "train_hours": 7,
    },
    "Attention U-Net": {
        "WT": 0.886, "TC": 0.884, "ET": 0.875, "mean": 0.882,
        "params_m": 22.66, "train_hours": 7,
    },
    "Swin UNETR": {
        "WT": 0.882, "TC": 0.863, "ET": 0.862, "mean": 0.869,
        "params_m": 62.2, "train_hours": 17.0,
    },
    "KAN U-Net": {
        "WT": 0.878, "TC": 0.873, "ET": 0.856, "mean": 0.869,
        "params_m": 2.42, "train_hours": 7,
    },
    "KAN 3D U-Net": {
        "WT": 0.879, "TC": 0.885, "ET": 0.869, "mean": 0.878,
        "params_m": 22.59, "train_hours": 7,
    },
}

MODEL_COLORS = {
    "3D U-Net": "#636EFA",
    "Attention U-Net": "#EF553B",
    "Swin UNETR": "#00CC96",
    "KAN U-Net": "#AB63FA",
    "KAN 3D U-Net": "#FFA15A",
}


def get_mlflow_metrics(run_name_fragment: str) -> dict:
    """Pull per-epoch metrics from MLflow for a run whose name contains the fragment."""
    try:
        import mlflow
        mlflow.set_tracking_uri(f"sqlite:///{ROOT}/mlflow.db")
        client = mlflow.tracking.MlflowClient()

        runs = client.search_runs(
            experiment_ids=[
                e.experiment_id
                for e in client.search_experiments()
                if e.name == "brain_tumor_seg"
            ],
            filter_string=f"tags.mlflow.runName LIKE '%{run_name_fragment}%'",
            order_by=["start_time DESC"],
            max_results=1,
        )
        if not runs:
            return {}

        run_id = runs[0].info.run_id
        metrics = {}
        for key in ["train/loss", "val/dice_WT", "val/dice_TC", "val/dice_ET", "val/dice_mean"]:
            history = client.get_metric_history(run_id, key)
            if history:
                metrics[key] = [(h.step, h.value) for h in sorted(history, key=lambda x: x.step)]
        return metrics
    except Exception as e:
        print(f"  MLflow query failed for '{run_name_fragment}': {e}")
        return {}


def plot_training_curves(out_path: Path):
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError:
        print("  plotly not installed, skipping. Install with: pip install plotly")
        return

    run_fragments = {
        "3D U-Net": "unet3d_baseline",
        "Attention U-Net": "attention_unet_baseline",
        "Swin UNETR": "swin_unetr_baseline",
        "KAN U-Net": "kan_unet3d_baseline",
        "KAN 3D U-Net": "unet3d_kan_ablation",
    }

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=["Training Loss", "Val Dice - WT", "Val Dice - TC", "Val Dice - ET"],
        shared_xaxes=False,
    )

    metric_map = [
        ("train/loss", 1, 1),
        ("val/dice_WT", 1, 2),
        ("val/dice_TC", 2, 1),
        ("val/dice_ET", 2, 2),
    ]

    any_data = False
    for model_name, fragment in run_fragments.items():
        print(f"  Fetching MLflow metrics for {model_name}...")
        metrics = get_mlflow_metrics(fragment)
        if not metrics:
            continue
        any_data = True
        color = MODEL_COLORS[model_name]
        first = True
        for metric_key, row, col in metric_map:
            if metric_key not in metrics:
                continue
            steps, values = zip(*metrics[metric_key])
            fig.add_trace(
                go.Scatter(
                    x=steps, y=values,
                    name=model_name,
                    line=dict(color=color, width=2),
                    legendgroup=model_name,
                    showlegend=first,
                    hovertemplate=f"{model_name}<br>Epoch %{{x}}<br>Value %{{y:.4f}}<extra></extra>",
                ),
                row=row, col=col,
            )
            first = False

    if not any_data:
        print("  No MLflow data found, training curves not generated.")
        return

    fig.update_layout(
        title="Training Curves - All Models",
        height=600,
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        font=dict(family="Inter, sans-serif", size=12),
    )
    fig.update_xaxes(title_text="Epoch")
    fig.write_html(out_path, include_plotlyjs="cdn")
    print(f"  Saved {out_path.name}")


def plot_model_comparison(out_path: Path):
    try:
        import plotly.graph_objects as go
    except ImportError:
        print("  plotly not installed, skipping.")
        return

    models = [m for m, r in RESULTS.items() if r["mean"] is not None]
    regions = ["WT", "TC", "ET"]
    region_colors = ["#4C78A8", "#F58518", "#E45756"]

    fig = go.Figure()
    for region, color in zip(regions, region_colors):
        fig.add_trace(go.Bar(
            name=region,
            x=models,
            y=[RESULTS[m][region] for m in models],
            marker_color=color,
            text=[f"{RESULTS[m][region]:.3f}" for m in models],
            textposition="outside",
            hovertemplate=f"{region}: %{{y:.4f}}<extra></extra>",
        ))

    fig.update_layout(
        title="Full-Volume Dice Score Comparison (50 val cases, sliding window inference)",
        barmode="group",
        yaxis=dict(title="Dice Score", range=[0.8, 0.95], tickformat=".3f"),
        xaxis_title="Model",
        template="plotly_white",
        legend_title="Region",
        height=480,
        font=dict(family="Inter, sans-serif", size=13),
    )
    fig.write_html(out_path, include_plotlyjs="cdn")
    print(f"  Saved {out_path.name}")


def plot_radar(out_path: Path):
    try:
        import plotly.graph_objects as go
    except ImportError:
        print("  plotly not installed, skipping.")
        return

    categories = ["WT", "TC", "ET", "WT"] # close the loop

    fig = go.Figure()
    for model_name, results in RESULTS.items():
        if results["mean"] is None:
            continue
        values = [results["WT"], results["TC"], results["ET"], results["WT"]]
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=categories,
            fill="toself",
            name=model_name,
            line_color=MODEL_COLORS[model_name],
            opacity=0.6,
        ))

    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0.83, 0.92])),
        title="Model Comparison - Radar Chart (Dice per Region)",
        template="plotly_white",
        height=500,
        font=dict(family="Inter, sans-serif", size=13),
    )
    fig.write_html(out_path, include_plotlyjs="cdn")
    print(f"  Saved {out_path.name}")


def plot_params_vs_dice(out_path: Path):
    try:
        import plotly.graph_objects as go
    except ImportError:
        print("  plotly not installed, skipping.")
        return

    fig = go.Figure()
    for model_name, results in RESULTS.items():
        if results["mean"] is None or results["params_m"] is None:
            continue
        fig.add_trace(go.Scatter(
            x=[results["params_m"]],
            y=[results["mean"]],
            mode="markers+text",
            name=model_name,
            text=[model_name],
            textposition="top center",
            marker=dict(size=14, color=MODEL_COLORS[model_name]),
            hovertemplate=(
                f"<b>{model_name}</b><br>"
                f"Params: %{{x:.1f}}M<br>"
                f"Mean Dice: %{{y:.4f}}<extra></extra>"
            ),
        ))

    fig.update_layout(
        title="Parameter Efficiency - Mean Dice vs Model Size",
        xaxis=dict(title="Parameters (M)", type="log"),
        yaxis=dict(title="Mean Dice (full-volume)", tickformat=".3f"),
        template="plotly_white",
        showlegend=False,
        height=450,
        font=dict(family="Inter, sans-serif", size=13),
    )
    fig.write_html(out_path, include_plotlyjs="cdn")
    print(f"  Saved {out_path.name}")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--no-mlflow", action="store_true",
                        help="Skip training curves (MLflow queries)")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if not args.no_mlflow:
        print("\n--- Training curves ---")
        plot_training_curves(OUT_DIR / "training_curves.html")

    print("\n--- Model comparison bar chart ---")
    plot_model_comparison(OUT_DIR / "model_comparison.html")

    print("\n--- Radar chart ---")
    plot_radar(OUT_DIR / "radar_comparison.html")

    print("\n--- Params vs Dice scatter ---")
    plot_params_vs_dice(OUT_DIR / "params_vs_dice.html")

    print(f"\nAll charts saved to {OUT_DIR}/")


if __name__ == "__main__":
    main()
