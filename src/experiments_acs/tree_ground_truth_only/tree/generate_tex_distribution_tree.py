import json
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from datetime import datetime
from experiments_acs.tree_income_distribution.latex.generate_tex_distribution_tree import subsample_bins
from utility.directories import get_results_dir, get_latex_template_dir
from utility.time_helper import convert_timestamp_to_readable_format


# Use white background for plots
plt.style.use("default")

# LaTeX-style plotting
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "roman",
    "text.latex.preamble": r"""
        \usepackage{amsfonts}
    """,
    "font.size": 16,
    "axes.titlesize": 22,
    "axes.labelsize": 18,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "legend.fontsize": 16,
})


def generate_binary_distribution_tree(
        gt_json_path: Path,
        template_path: Path,
        out_path: Path,
        subsampling_factor: int,
        lower_limit: int,
        upper_limit: int,
):
    # Load metadata
    gt_metadata_path = gt_json_path / "meta.json"
    gt_metadata = json.loads(gt_metadata_path.read_text())

    # Extract timestamp
    timestamp = convert_timestamp_to_readable_format(gt_metadata["timestamp"])

    # Create subtitle from metadata
    subtitle = (f"Experiment timestamp: {timestamp}"
                r"\\ "
                f"Survey: {gt_metadata['survey_name']} ({gt_metadata['survey_year']})"
                r"\\ "
                f"Target: {gt_metadata['target_description']} ({gt_metadata['target_column']})"
                r"\\ "
                f"Bin subsampling factor: {subsampling_factor}"
                )

    # Load all level files with ground-truth results
    gt_level_files = sorted(gt_json_path.glob("level_*.json"), key=lambda p: int(p.stem.split("_")[1]))
    print(f"Found {len(gt_level_files)} level files with GT results: {[f.name for f in gt_level_files]}")

    # Build node and children mappings
    tree = build_tree(gt_level_files=gt_level_files)

    # Emit TikZ
    tikz_tree = emit_node(tree=tree, out_path=out_path, lower_limit=lower_limit,
                          upper_limit=upper_limit, subsampling_factor=subsampling_factor, node_id=0, root=True) + ";"

    # Fill template
    template = Path(template_path).read_text()
    tex = template.replace("{{SUBTITLE}}", subtitle)
    tex = tex.replace("{{TREE}}", tikz_tree)
    tex = tex.replace("{{ADDITIONAL}}", "")

    # Write output
    out_path.parent.mkdir(exist_ok=True)
    Path(out_path).write_text(tex)
    print(f"Generated {out_path}")


def build_tree(gt_level_files: list) -> dict:
    """
    Build a single dict mapping node_id → node_data including children list.
    Assumes levels are ordered: root first, then level 1, level 2, ...
    """
    tree = {}

    for level_file in gt_level_files:
        data = json.loads(Path(level_file).read_text())
        total_weight = sum(node["weight_after_filtering"] for node in data) or 1
        for node in data:
            node_id = node["id"]
            tree[node_id] = {}

            # Compute prior
            prior = node["weight_after_filtering"] / total_weight
            assert prior == node["weighted_prior"], "Prior mismatch"

            # Copy node, add prior, and initialize children
            tree[node_id]["ground_truth"] = dict(node)
            tree[node_id]["prior"] = prior
            tree[node_id]["children"] = []

            # Link to parent if it exists
            if node["parent_id"] is not None:
                tree[node["parent_id"]]["children"].append(node_id)

    return tree


def emit_node(
        tree: dict,
        node_id: int,
        out_path: Path,
        lower_limit: int,
        upper_limit: int,
        subsampling_factor: int,
        indent: int = 0,
        root: bool = False):
    """
    Recursively generate TikZ code from the tree dict.
    """
    single_pad = "   "
    pad = single_pad * indent
    node = tree[node_id]

    # Description
    prob_mass = node["ground_truth"]["probability_mass"]
    prior = node["prior"]
    mean = node["ground_truth"]["target_mean"]
    std = node["ground_truth"]["target_std"]

    # Subsampling
    bin_edges, counts, sub_prob = subsample_bins(
        bin_edges=np.array(node["ground_truth"]["bin_edges"]),
        counts=np.array(node["ground_truth"]["counts"]),
        lower_limit=lower_limit,
        upper_limit=upper_limit,
        subsampling_factor=subsampling_factor)

    # Description
    description = (f"{pad + single_pad}GT probability = {100 * prob_mass * sub_prob:.0f}" + r"\% \\[1mm]" + "\n"
                   + f"{pad + single_pad}Prior = {100 * prior:.2f}" + r"\% \\[1mm]" + "\n"
                   + f"{pad + single_pad}Standard dev = ${std / 1000:.1f}" + r"\mathrm{e}3$\\[1mm]" + "\n")

    # Create histogram
    plt_path = create_income_distr_histogram(
        bin_edges=bin_edges,
        counts=counts,
        mean=mean,
        out_path=out_path.parent,
        node_id=node_id)
    figure = r"\includegraphics[width=70mm]{" + str(plt_path) + "}"

    # Root node needs leading backslash
    if root:
        node_tex = ("\\node[treenode]" + r" {\textbf{Root}\\[1mm]" + f"\n{description}"
                    + f"{pad + single_pad}{figure}" + "\n}")
    else:
        short_label = node["ground_truth"]["short_label"].replace("_", "-").replace("PARTITIONED", "PART")
        node_tex = ("node[treenode] " + r"{\textbf{" + short_label + r"}\\[1mm]" + f"\n{description}"
                    + f"{pad + single_pad}{figure}" + f"\n{pad}}}")

    # Recurse through children
    for child_id in sorted(node.get("children", [])):
        child = emit_node(tree=tree, node_id=child_id, out_path=out_path,
                          lower_limit=lower_limit, upper_limit=upper_limit, subsampling_factor=subsampling_factor,
                          indent=indent + 2)
        node_tex += (f"\n{pad + single_pad}child {{"
                     f"\n{pad + 2 * single_pad}{child}" + f"\n{pad + single_pad}" + "}")

    return node_tex


def create_income_distr_histogram(
        bin_edges: np.ndarray,
        counts: np.ndarray,
        mean: float,
        out_path: Path,
        node_id: int) -> Path:
    plt.figure(figsize=(8, 6))

    # Ground-truth
    width = 0.9 * (bin_edges[1] - bin_edges[0])
    gt_bin_centers = (bin_edges[1:] + bin_edges[:-1]) / 2
    plt.bar(gt_bin_centers, counts, width=width, align="edge", alpha=0.6, label="GT", color="blue")
    plt.axvline(mean, color="blue", linestyle="--", linewidth=2, label="GT mean")

    # Format
    plt.ylabel("Density", labelpad=10)
    plt.legend()

    # Save
    plot_name = f"hist_{node_id}.pdf"
    plot_path = out_path / plot_name
    plt.savefig(plot_path, bbox_inches="tight")

    return plot_path


if __name__ == "__main__":
    gt_experiment_name = "ground_truth_income"
    gt_datetime = "2026-04-16__18-20-00"
    gt_cluster = True

    # Construct paths
    gt_base_path = get_results_dir(experiment_name=gt_experiment_name, cluster=gt_cluster, use_timestamp=True,
                                   timestamp=gt_datetime)
    gt_json_path = gt_base_path / "tree"

    combined_name = "gt_income_tree"
    datetime = datetime.now().strftime("%Y-%m-%d__%H-%M-%S")
    base_path = get_results_dir(
        experiment_name=combined_name,
        cluster=False,
        use_timestamp=True,
        create_dir=True,
    )
    template_dir = get_latex_template_dir()
    template_path = template_dir / "distribution_tree/template_variance_tree_three.tex"
    out_path = base_path / "main.tex"

    # Generate TikZ tree from JSON and template
    generate_binary_distribution_tree(
        gt_json_path=gt_json_path,
        template_path=template_path,
        out_path=out_path,
        lower_limit=-20_000,
        upper_limit=300_000,
        subsampling_factor=5,
    )
