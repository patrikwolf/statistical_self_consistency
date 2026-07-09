import matplotlib
import matplotlib.pyplot as plt


def set_plot_style(
        sans_serif: bool,
        factor: float = 1.0):
    # Use white background for plots
    plt.style.use("default")

    if sans_serif:
        font = "sans-serif"
        packages = r"""
            \usepackage{amsfonts}
            \usepackage{sfmath}
            \usepackage{xcolor}
        """
    else:
        font = "serif"
        packages = r"""
            \usepackage{amsfonts}
            \usepackage{xcolor}
        """

    # LaTeX-style plotting
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": font,
        "text.latex.preamble": packages,
        "pgf.preamble": packages,
        "font.size": int(16 * factor),
        "axes.titlesize": int(22 * factor),
        "axes.labelsize": int(18 * factor),
        "xtick.labelsize": int(16 * factor),
        "ytick.labelsize": int(16 * factor),
        "legend.fontsize": int(16 * factor),
    })


def setup_pgf_plots(font_size: int = 11, line_scaling: float = 1.0):
    GRID = "#D0D0D0"
    matplotlib.use("pgf")
    matplotlib.rcParams.update({
        "pgf.texsystem": "pdflatex",
        "font.family": "serif",
        "font.serif": "Times New Roman",
        "font.size": font_size,
        "text.usetex": True,
        "text.latex.preamble": r"\usepackage{amsmath}\usepackage{amssymb}",
        "pgf.preamble": r"\usepackage{amsmath}\usepackage{amssymb}",
        "pgf.rcfonts": False,
        "lines.linewidth": 1.75,
        "lines.markersize": 3,
        "figure.labelsize": font_size,
        "axes.titlesize": font_size,
        "axes.labelsize": font_size,
        "xtick.labelsize": font_size * 0.9,
        "ytick.labelsize": font_size * 0.9,
        "axes.linewidth": 1.25 * line_scaling,         # Adjust axis boundary thickness
        "xtick.major.size": 6 * line_scaling,          # Length of major ticks on x-axis
        "xtick.major.width": 1.25 * line_scaling,      # Thickness of major ticks on x-axis
        "ytick.major.size": 6 * line_scaling,          # Length of major ticks on y-axis
        "ytick.major.width": 1.25 * line_scaling,      # Thickness of major ticks on y-axis
        "xtick.minor.size": 3 * line_scaling,          # Length of minor ticks on x-axis
        "xtick.minor.width": 1 * line_scaling,         # Thickness of minor ticks on x-axis
        "ytick.minor.size": 3 * line_scaling,          # Length of minor ticks on y-axis
        "ytick.minor.width": 1 * line_scaling,         # Thickness of minor ticks on y-axis
        "axes.facecolor": "white",     # Set light gray background for all axes
        "figure.facecolor": "white",    # Optionally set figure background (around the plot) to white
        "grid.color": GRID,             # Set grid lines color to white
        "grid.linewidth": 1.25 * line_scaling,         # Set grid line width
        "grid.alpha": 1.0,              # Set the opacity of the grid lines (1.0 for solid)
        "grid.linestyle": "-",          # Set solid grid line style
        "axes.grid": True,              # Enable grid by default
        "axes.grid.which": "major",     # Show grid for both major and minor ticks
    })
