"""One visual system for every figure in the project (report, deck, app).

Palette = validated reference palette (dataviz skill): fixed categorical order,
single-hue blue sequential ramp, recessive grid, thin marks.
"""
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from . import config

SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT_2 = "#52514e"
MUTED = "#8a8984"
GRID = "#e6e5e1"
SPLIT_FILL = {"train": "#f3f2ef", "val": "#e9e8e4", "test": "#dfded9"}

# Categorical slots — always assigned in this order, never cycled.
C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
BLUE_RAMP = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7",
             "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"]
SEQ = LinearSegmentedColormap.from_list("seq_blue", BLUE_RAMP)
DIVERGING = LinearSegmentedColormap.from_list("div", ["#2a78d6", "#f0efec", "#e34948"])


def apply():
    mpl.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "figure.dpi": 110, "savefig.dpi": 200, "savefig.bbox": "tight",
        "font.size": 10, "font.family": "DejaVu Sans",
        "axes.titlesize": 12, "axes.titleweight": "bold", "axes.titlelocation": "left",
        "axes.titlecolor": TEXT, "axes.labelcolor": TEXT_2, "axes.labelsize": 10,
        "axes.edgecolor": GRID, "axes.linewidth": 0.8,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "axes.axisbelow": True, "grid.color": GRID, "grid.linewidth": 0.6,
        "xtick.color": TEXT_2, "ytick.color": TEXT_2,
        "xtick.major.size": 0, "ytick.major.size": 0,
        "lines.linewidth": 2.0, "legend.frameon": False, "legend.fontsize": 9,
        "axes.prop_cycle": mpl.cycler(color=C),
    })


def subtitle(ax, text, width=None):
    """Grey takeaway under the title (the 'so what'), wrapped to the axes width.
    Returns the number of lines so the caller can size the title pad."""
    import textwrap
    if width is None:
        w_in = ax.get_position().width * ax.figure.get_size_inches()[0]
        width = int(w_in * 13.5)          # ~13.5 chars per inch at 9 pt
    wrapped = textwrap.fill(text, width)
    ax.text(0, 1.02, wrapped, transform=ax.transAxes, fontsize=9, color=TEXT_2, va="bottom")
    return wrapped.count("\n") + 1


def save(fig, name):
    path = config.FIG_DIR / f"{name}.png"
    fig.savefig(path)
    plt.close(fig)
    return path


# One colour per model, fixed for the whole project (colour follows the entity, never its rank).
MODEL_COLORS = {
    "Naive-1 (persistence)": C[0],
    "Seasonal naive-24": C[1],
    "Naive-1 + hourly step": C[2],
    "Seasonal naive-168": MUTED,          # retired after Phase 3 (its slot is reused by Random Forest)
    "Calendar climatology": MUTED,        # retired after Phase 3
    # Phase 4 — best linear model (the one carried into the final comparison)
    "Lasso + hour×month": C[5],
    # Final contenders (Phase 7 onwards), one fixed colour each
    "Random Forest": C[3],
    "XGBoost": C[6],
    "MLP v4 [64,32] Δ-target": C[7],
}

# Comparison bar charts colour by model FAMILY (9+ models would exceed the 8 categorical slots).
FAMILY_COLORS = {"Baseline": MUTED, "Linear": C[5], "Tree": C[6], "Neural": C[7]}


def family(model: str) -> str:
    if model.startswith(("Naive", "Seasonal", "Calendar")):
        return "Baseline"
    if any(k in model for k in ("OLS", "Ridge", "Lasso")):
        return "Linear"
    if any(k in model for k in ("Tree", "Forest", "XGBoost")):
        return "Tree"
    return "Neural"
