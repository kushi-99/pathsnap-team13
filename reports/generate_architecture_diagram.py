"""Generate clean architecture diagrams for the PathSnap Review 2 Technical Report PDF."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "pathsnap_mpl"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch


def _box(ax, x, y, w, h, title, body, fc="#eef5ff", ec="#1f4e79", title_color="#102a43"):
    rect = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.015,rounding_size=0.02",
        facecolor=fc,
        edgecolor=ec,
        linewidth=1.6,
    )
    ax.add_patch(rect)
    ax.text(
        x + w / 2,
        y + h - 0.032,
        title,
        ha="center",
        va="top",
        fontsize=9.5,
        fontweight="bold",
        color=title_color,
    )
    ax.text(
        x + w / 2,
        y + h / 2 - 0.02,
        body,
        ha="center",
        va="center",
        fontsize=7.8,
        color="#243b53",
        linespacing=1.35,
    )


def _arrow(ax, x1, y1, x2, y2, label="", color="#334e68", label_below=False):
    arr = FancyArrowPatch(
        (x1, y1),
        (x2, y2),
        arrowstyle="-|>,head_length=7,head_width=4",
        color=color,
        linewidth=1.6,
        mutation_scale=1.0,
    )
    ax.add_patch(arr)
    if label:
        dy = -0.022 if label_below else 0.022
        va = "top" if label_below else "bottom"
        ax.text(
            (x1 + x2) / 2,
            (y1 + y2) / 2 + dy,
            label,
            ha="center",
            va=va,
            fontsize=7.5,
            fontweight="bold",
            color=color,
            bbox=dict(boxstyle="round,pad=0.18", facecolor="white", edgecolor="none", alpha=0.9),
        )


def generate_two_laptop_architecture_diagram(out_path: Path) -> Path:
    """Create a high-resolution diagram of PathSnap's Two-Laptop + ML Inference Architecture."""
    fig, ax = plt.subplots(figsize=(11.5, 5.4), dpi=180)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(
        0.5,
        0.96,
        "PathSnap — Distributed Two-Laptop Map-Matching & Evaluation Architecture",
        ha="center",
        va="top",
        fontsize=13,
        fontweight="bold",
        color="#102a43",
    )

    # Laptop 2 container
    l2 = FancyBboxPatch(
        (0.02, 0.07),
        0.34,
        0.82,
        boxstyle="round,pad=0.015,rounding_size=0.025",
        facecolor="#f7fbff",
        edgecolor="#2b6cb0",
        linewidth=2.0,
        linestyle="--",
    )
    ax.add_patch(l2)
    ax.text(0.19, 0.865, "LAPTOP 2 — Evaluator Client UI", ha="center", va="top", fontsize=10.5, fontweight="bold", color="#2b6cb0")

    _box(
        ax, 0.045, 0.62, 0.29, 0.19,
        "Streamlit 5-Page Frontend",
        "1. Dashboard  2. Model Comparison\n3. GPS Map Matching  4. Evaluation\n5. System Info (ui/evaluator_dashboard/)",
        fc="#ebf8ff", ec="#2b6cb0",
    )
    _box(
        ax, 0.045, 0.36, 0.29, 0.19,
        "Folium Interactive Map Renderer",
        "OpenStreetMap Tiles + LayerControl\nGPS Track | Predicted Route | Candidates\nOptional Dev Reference (No Eval Truth)",
        fc="#e6fffa", ec="#2c7a7b",
    )
    _box(
        ax, 0.045, 0.11, 0.29, 0.18,
        "Configurable HTTP API Client",
        "Env: PATHSNAP_API_URL=http://<IP>:8000\nHealth Checks, Timeouts, Offline Guards\n(Zero Model Weights / Raw Data Needed)",
        fc="#fefcbf", ec="#b7791f",
    )
    _arrow(ax, 0.19, 0.62, 0.19, 0.55)
    _arrow(ax, 0.19, 0.36, 0.19, 0.29)

    # Laptop 1 container
    l1 = FancyBboxPatch(
        (0.45, 0.07),
        0.53,
        0.82,
        boxstyle="round,pad=0.015,rounding_size=0.025",
        facecolor="#f8fafc",
        edgecolor="#2f855a",
        linewidth=2.0,
        linestyle="--",
    )
    ax.add_patch(l1)
    ax.text(0.715, 0.865, "LAPTOP 1 — FastAPI Inference & Evaluation Server", ha="center", va="top", fontsize=10.5, fontweight="bold", color="#2f855a")

    _box(
        ax, 0.475, 0.62, 0.48, 0.19,
        "FastAPI + Uvicorn REST API (0.0.0.0:8000)",
        "GET /health | GET /models | GET /trips | GET /trips/{id}\nPOST /predict | POST /evaluate (Pydantic v2 + Request Logs)",
        fc="#f0fff4", ec="#2f855a",
    )
    _box(
        ax, 0.475, 0.36, 0.225, 0.19,
        "Startup Model Registry",
        "Loaded ONCE at Startup:\nMLP (128-64-32, PyTorch)\nHistGB | Random Forest | LogReg",
        fc="#faf5ff", ec="#6b46c1",
    )
    _box(
        ax, 0.73, 0.36, 0.225, 0.19,
        "Candidate & Feature Engine",
        "Local Tangent-Plane cKDTree\n28 Observable Features\n(Geometry, Heading, Topology)",
        fc="#fff5f5", ec="#c53030",
    )
    _box(
        ax, 0.475, 0.11, 0.225, 0.18,
        "Server-Side Evaluation",
        "Cand Acc, Macro-F1, Top-1,\nLog Loss, ROC/PR, Continuity\n(Eval .route Stays Server-Side)",
        fc="#edf2f7", ec="#4a5568",
    )
    _box(
        ax, 0.73, 0.11, 0.225, 0.18,
        "Road Network & Datasets",
        "80 Train / 10 Student-Test\n(.track, .nodes, .arcs, .route)\nEnv: PATHSNAP_DATA_ROOT",
        fc="#fffaf0", ec="#c05621",
    )

    # Internal arrows on Laptop 1
    _arrow(ax, 0.587, 0.62, 0.587, 0.55)
    _arrow(ax, 0.842, 0.62, 0.842, 0.55)
    _arrow(ax, 0.842, 0.29, 0.842, 0.36)
    _arrow(ax, 0.73, 0.455, 0.70, 0.455)
    _arrow(ax, 0.587, 0.36, 0.587, 0.29)

    # Cross-laptop HTTP LAN arrows
    _arrow(ax, 0.335, 0.745, 0.475, 0.745, label="HTTP / JSON\n(LAN Wi-Fi)", color="#1a365d")
    _arrow(ax, 0.475, 0.675, 0.335, 0.675, label="Predicted Route\n& Metrics", color="#2f855a", label_below=True)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


if __name__ == "__main__":
    out = Path(__file__).resolve().parent / "assets" / "two_laptop_architecture.png"
    generate_two_laptop_architecture_diagram(out)
    print(f"Generated architecture diagram: {out}")
