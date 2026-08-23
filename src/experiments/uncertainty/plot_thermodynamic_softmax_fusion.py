"""
3D Thermodynamic Surface and 2D Heatmap of Adaptive Softmax Fusion.

Visualizes the smooth non-linear fusion surface combining Baseline (DARK) uncertainty
and GMM Covariance Volumetric uncertainty projected via the thermodynamic beta parameter.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
OUTPUT_FIGURES_DIR = PROJECT_ROOT / "outputs" / "figures"


def adaptive_softmax_fusion(u_base: np.ndarray, u_gmm: np.ndarray) -> np.ndarray:
    """Compute Softmax Fusion of baseline uncertainty and GMM volumetric uncertainty."""
    num = np.exp(u_base) * u_base + np.exp(u_gmm) * u_gmm
    den = np.exp(u_base) + np.exp(u_gmm)
    return num / den


def u_gmm_projection(volume: np.ndarray, beta: float = 0.1) -> np.ndarray:
    """Project unbounded covariance volume into [0, 1] bounded uncertainty space."""
    return 1.0 - np.exp(-beta * volume)


def generate_plots(output_dir: Path = OUTPUT_FIGURES_DIR, beta: float = 0.1) -> Tuple[Path, Path]:
    """Generate both 3D Thermodynamic Surface and 2D Heatmap of Softmax Fusion."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Set up evaluation grid
    u_base_vals = np.linspace(0, 1, 100)
    volume_vals = np.linspace(0, 50, 100)
    u_base_grid, volume_grid = np.meshgrid(u_base_vals, volume_vals)

    # Project volume and apply fusion
    u_gmm_grid = u_gmm_projection(volume_grid, beta=beta)
    u_adapt_grid = adaptive_softmax_fusion(u_base_grid, u_gmm_grid)

    # ---------------------------------------------------------
    # 1. 3D Surface Plot
    # ---------------------------------------------------------
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    surf = ax.plot_surface(
        u_base_grid,
        volume_grid,
        u_adapt_grid,
        cmap=cm.inferno,
        linewidth=0,
        antialiased=True,
        alpha=0.9,
    )

    ax.set_xlabel(r"Baseline Uncertainty ($U_{base}$)", fontsize=12, labelpad=10)
    ax.set_ylabel(r"GMM Covariance Volume ($|\Sigma_{final}|$)", fontsize=12, labelpad=10)
    ax.set_zlabel(r"Softmax Fusion ($U_{adapt}$)", fontsize=12, labelpad=10)
    ax.set_title(rf"3D Thermodynamic Surface of Adaptive Softmax Fusion ($\beta = {beta}$)", fontsize=14, pad=20)

    fig.colorbar(surf, shrink=0.5, aspect=10, pad=0.1, label=r"Softmax Fusion ($U_{adapt}$)")
    ax.view_init(elev=25, azim=135)

    out_3d_png = output_dir / "softmax_fusion_3d.png"
    out_3d_pdf = output_dir / "softmax_fusion_3d.pdf"
    plt.savefig(out_3d_png, dpi=300, bbox_inches="tight")
    plt.savefig(out_3d_pdf, bbox_inches="tight")
    plt.close(fig)

    logger.info("[SUCCESS] Saved 3D Surface plot to: %s", out_3d_png)

    # ---------------------------------------------------------
    # 2. 2D Heatmap Plot
    # ---------------------------------------------------------
    fig2, ax2 = plt.subplots(figsize=(8, 6))
    contour = ax2.contourf(u_base_grid, volume_grid, u_adapt_grid, levels=100, cmap=cm.inferno)

    ax2.set_xlabel(r"Baseline Uncertainty ($U_{base} = $ DARK ($1-P_{DARK}$))", fontsize=12)
    ax2.set_ylabel(r"GMM Covariance Volume ($|\Sigma_{final}|$)", fontsize=12)
    ax2.set_title(rf"2D Heatmap of Adaptive Softmax Fusion ($\beta = {beta}$)", fontsize=14)
    fig2.colorbar(contour, label=r"Final Uncertainty ($U_{adapt}$)")

    out_2d_png = output_dir / "softmax_fusion_2d.png"
    out_2d_pdf = output_dir / "softmax_fusion_2d.pdf"
    plt.savefig(out_2d_png, dpi=300, bbox_inches="tight")
    plt.savefig(out_2d_pdf, bbox_inches="tight")
    plt.close(fig2)

    logger.info("[SUCCESS] Saved 2D Heatmap plot to: %s", out_2d_png)
    return out_3d_png, out_2d_png


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate 3D Thermodynamic Surface and 2D Heatmap.")
    parser.add_argument("--beta", type=float, default=0.1, help="Beta projection hyperparameter (default: 0.1)")
    parser.add_argument("--out-dir", type=str, default=str(OUTPUT_FIGURES_DIR), help="Output directory")
    args = parser.parse_args()

    generate_plots(output_dir=Path(args.out_dir), beta=args.beta)


if __name__ == "__main__":
    main()
