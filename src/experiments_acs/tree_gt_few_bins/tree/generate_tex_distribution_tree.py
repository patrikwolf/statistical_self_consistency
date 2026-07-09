import json
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from datetime import datetime
from design_elements.color_palette import get_histogram_colors
from utility.directories import get_results_dir, get_latex_template_dir
from utility.plot_style import setup_pgf_plots
from utility.time_helper import convert_timestamp_to_readable_format


def generate_binary_distribution_tree(
        gt_json_path: Path,
        template_path: Path,
        out_path: Path,
        equal_bar_width: bool,
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
                )

    # Load all level files with ground-truth results
    gt_level_files = sorted(gt_json_path.glob("level_*.json"), key=lambda p: int(p.stem.split("_")[1]))
    print(f"Found {len(gt_level_files)} level files with GT results: {[f.name for f in gt_level_files]}")

    # Build node and children mappings
    tree = build_tree(gt_level_files=gt_level_files)

    # Emit TikZ
    tikz_tree = emit_node(tree=tree, out_path=out_path, equal_bar_width=equal_bar_width, node_id=0, root=True) + ";"

    # Additional information: bins
    if not equal_bar_width:
        additional = ""
    else:
        additional = r"""% Bins
\node[
    font=\fontsize{6}{7}\selectfont,
    anchor=north west,
    draw=black!65,
    rounded corners=2pt,
    line width=1.2pt,
    fill=white,
    inner xsep=6pt,
    inner ysep=6pt,
] at (-6.95, -4.33) {
\begin{tabular}{@{}c@{\hspace{2.4mm}}l@{}}
\textbf{Bins} & \textbf{Range} \\[1mm]
"""

        # Sample colors from a colormap
        bin_edges = tree[0]["ground_truth"]["bin_edges"]
        gt_colors = get_histogram_colors(llm=False, num_colors=len(bin_edges[:-1]))

        for idx, bin_edge in enumerate(bin_edges[:-1]):
            label = chr(ord("A") + idx)
            left_edge = convert_edge_to_str(bin_edge)
            right_edge = convert_edge_to_str(bin_edges[idx + 1])

            # Assuming colors are RGB triples in [0, 1]
            gt_rgb = tuple(int(255 * c) for c in gt_colors[idx][:3])
            gt_name = f"gtbin{idx}"

            additional += (
                rf"\definecolor{{{gt_name}}}{{RGB}}{{{gt_rgb[0]},{gt_rgb[1]},{gt_rgb[2]}}}" "\n"
                rf"\tikz[baseline=-0.5ex]{{" "\n"
                rf"\draw[fill={gt_name}, draw=none, rounded corners=0mm] (0,-0.06) rectangle (0.10,0.14);" "\n"
                rf"}}~\textbf{{{label}}}" "\n"
                rf" & $[{left_edge}, {right_edge})$ USD \\[0.8mm]"
                "\n"
            )

        # Remove last line break and add final string
        additional = additional[:-17] + r"]$ USD"
        additional += r"""\end{tabular}
                };"""

    # Fill template
    template = Path(template_path).read_text()
    tex = template.replace("{{SUBTITLE}}", subtitle)
    tex = tex.replace("{{TREE}}", tikz_tree)
    tex = tex.replace("{{ADDITIONAL}}", additional)

    # Write output
    out_path.parent.mkdir(exist_ok=True)
    Path(out_path).write_text(tex)
    print(f"Generated {out_path}")


def convert_edge_to_str(edge: float | int) -> str:
    if edge % 1000 == 0:
        return f"{edge / 1000:.0f}" + r"\mathrm{k}"
    if edge % 100 == 0:
        return f"{edge / 1000:.1f}" + r"\mathrm{k}"
    else:
        return f"{edge:.0f}"


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
        equal_bar_width: bool,
        indent: int = 0,
        root: bool = False
):
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

    # Description
    if equal_bar_width:
        description = ""
    else:
        description = (f"{pad + single_pad}GT probability = {100 * prob_mass:.0f}" + r"\% \\[1mm]" + "\n"
                       + f"{pad + single_pad}Prior = {100 * prior:.2f}" + r"\% \\[1mm]" + "\n"
                       + f"{pad + single_pad}Standard dev = ${std / 1000:.1f}" + r"\mathrm{e}3$\\[1mm]" + "\n")

    # Create histogram
    plt_path = create_few_bin_histogram(
        bin_edges=np.array(node["ground_truth"]["bin_edges"]),
        counts=np.array(node["ground_truth"]["counts"]),
        mean=mean,
        out_path=out_path.parent,
        node_id=node_id,
        equal_bar_width=equal_bar_width,
    )
    figure = r"""\begingroup
\tikzset{
  every path/.style={},
  every node/.style={},
  every picture/.style={},
}%
\pgfsetcornersarced{\pgfpointorigin}%
\resizebox{\linewidth}{!}{%
\input{""" + str(plt_path) + r"""}
}
\endgroup
"""

    # Root node needs leading backslash
    if root:
        node_tex = ("\\node[treenode] {\n" + description
                    + f"{pad + single_pad}{figure}" + "}")
        edge_tex = ""
    else:
        short_label = node["ground_truth"]["short_label"].replace("_", "-").replace("PARTITIONED", "PART")
        node_tex = ("node[treenode] {\n" + description
                    + f"{pad + single_pad}{figure}" + f"{pad}}}")
        edge_tex = "edge from parent node[edgelabel, pos=0.3] " + r"{\textbf{" + short_label + "}}"

    # Recurse through children
    for child_id in sorted(node.get("children", [])):
        child = emit_node(tree=tree, node_id=child_id, out_path=out_path, equal_bar_width=equal_bar_width, indent=indent + 2)
        node_tex += (f"\n{pad + single_pad}child {{"
                     f"\n{pad + 2 * single_pad}{child}" + f"\n{pad + single_pad}" + "}")

    # Add edge to parent
    node_tex += "\n" + pad + edge_tex

    return node_tex


def create_few_bin_histogram(
        bin_edges: np.ndarray,
        counts: np.ndarray,
        mean: float,
        out_path: Path,
        node_id: int,
        equal_bar_width: bool,
) -> Path:
    # Setup
    setup_pgf_plots(font_size=8, line_scaling=0.6)

    # Create figure
    plt.figure(figsize=(1.3, 0.8))

    # Ground-truth
    if equal_bar_width:
        # x-axis
        x = np.arange(len(counts))
        labels = [chr(ord("A") + i) for i in range(len(counts))]

        # Normalize counts to one
        counts = counts / sum(counts)

        # Sample colors from a colormap
        colors = get_histogram_colors(llm=False, num_colors=len(counts))

        plt.bar(x, counts, align="center", alpha=0.85, color=colors,
                edgecolor="none", linewidth=0.0)
        plt.xticks(x, labels)
        plt.xlabel("Income bin", labelpad=1.75)
        plt.ylabel("Probability", labelpad=3)
    else:
        bin_widths = np.diff(bin_edges)
        plt.bar(bin_edges[:-1], counts, width=bin_widths, align="edge", alpha=0.6, label="GT", color="blue",
                edgecolor="none", linewidth=0.0)
        plt.axvline(mean, color="blue", linestyle="--", linewidth=2, label="GT mean")

        # Format
        plt.legend()

    # Save
    plot_name = f"binned_hist_{node_id}.pgf"
    plot_path = out_path / plot_name
    plt.savefig(plot_path, bbox_inches="tight", pad_inches=0.008)

    return plot_path


if __name__ == "__main__":
    gt_experiment_name = "ground_truth_few_bins"
    gt_datetime = "2026-04-17__15-30-00"
    gt_cluster = True

    # Construct paths
    gt_base_path = get_results_dir(experiment_name=gt_experiment_name, cluster=gt_cluster, use_timestamp=True,
                                   timestamp=gt_datetime)
    gt_json_path = gt_base_path / "tree"

    combined_name = "ground_truth_few_bins"
    datetime = datetime.now().strftime("%Y-%m-%d__%H-%M-%S")
    base_path = get_results_dir(
        experiment_name=combined_name,
        cluster=False,
        use_timestamp=True,
        create_dir=True,
    )
    template_dir = get_latex_template_dir()
    template_path = template_dir / "gt_histogram_tree" / "template_hist_tree_three.tex"
    out_path = base_path / "main.tex"

    # Generate TikZ tree from JSON and template
    generate_binary_distribution_tree(
        gt_json_path=gt_json_path,
        template_path=template_path,
        out_path=out_path,
        equal_bar_width=True,
    )
