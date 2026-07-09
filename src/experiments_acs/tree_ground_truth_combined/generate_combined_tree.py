import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from pathlib import Path
from datetime import datetime
from design_elements.color_palette import get_histogram_colors
from experiments_acs.tree_income_distribution.latex.generate_tex_distribution_tree import subsample_bins
from utility.directories import get_results_dir, get_latex_template_dir
from utility.plot_style import setup_pgf_plots
from utility.time_helper import convert_timestamp_to_readable_format


def generate_combined_tree(
        distr_json_path: Path,
        binned_json_path: Path,
        template_path: Path,
        out_path: Path,
        subsampling_factor: int,
        lower_limit: int,
        upper_limit: int,
):
    # Load metadata
    distr_metadata_path = distr_json_path / "meta.json"
    distr_metadata = json.loads(distr_metadata_path.read_text())

    # Extract timestamp
    timestamp = convert_timestamp_to_readable_format(distr_metadata["timestamp"])

    # Create subtitle from metadata
    subtitle = (f"Experiment timestamp: {timestamp}"
                r"\\ "
                f"Survey: {distr_metadata['survey_name']} ({distr_metadata['survey_year']})"
                r"\\ "
                f"Target: {distr_metadata['target_description']} ({distr_metadata['target_column']})"
                r"\\ "
                f"Bin subsampling factor: {subsampling_factor}"
                )

    # Load all level files with distribution results
    distr_level_files = sorted(distr_json_path.glob("level_*.json"), key=lambda p: int(p.stem.split("_")[1]))
    print(f"Found {len(distr_level_files)} level files with distribution results: {[f.name for f in distr_level_files]}")

    # Load all level files with binned results
    binned_level_files = sorted(binned_json_path.glob("level_*.json"), key=lambda p: int(p.stem.split("_")[1]))
    print(f"Found {len(binned_level_files)} level files with binned results: {[f.name for f in binned_level_files]}")

    # Build node and children mappings
    tree = build_tree(distr_level_files=distr_level_files, binned_level_files=binned_level_files)

    # Emit TikZ
    tikz_tree = emit_node(tree=tree, out_path=out_path, lower_limit=lower_limit,
                          upper_limit=upper_limit, subsampling_factor=subsampling_factor, node_id=0, root=True) + ";"

    # Additional information: bins
    additional = r"""% Bins
    \node[
        anchor=north west,
        draw=black!65,
        rounded corners=2pt,
        line width=3pt,
        fill=white,
        inner xsep=15pt,
        inner ysep=15pt,
    ] at (-31.15, -2.62) {
    \parbox{100mm}{
    \LARGE\textbf{Bins}\\[1mm]
    """

    bin_edges = tree[0]["binned"]["bin_edges"]
    for idx, bin_edge in enumerate(bin_edges[:-1]):
        label = chr(ord("A") + idx)
        additional += f"Answer {label}: $[{bin_edge:.0f}, {bin_edges[idx + 1]:.0f})$ USD" + r"\\[1mm]" + "\n"

    # Remove last line break and add final string
    additional = additional[:-14] + r"]$ USD}};"

    # Fill template
    template = Path(template_path).read_text()
    tex = template.replace("{{SUBTITLE}}", subtitle)
    tex = tex.replace("{{TREE}}", tikz_tree)
    tex = tex.replace("{{ADDITIONAL}}", additional)
    tex = tex.replace("{{SUBTITLE_AGG_DISTRIBUTION}}", "")
    tex = tex.replace("{{TREE_AGG_DISTRIBUTION}}", "")
    tex = tex.replace("{{SUBTITLE_AGG_ERRORS}}", "")
    tex = tex.replace("{{TREE_AGG_ERRORS}}", "")

    # Write output
    out_path.parent.mkdir(exist_ok=True)
    Path(out_path).write_text(tex)
    print(f"Generated {out_path}")


def build_tree(distr_level_files: list, binned_level_files: list) -> dict:
    """
    Build a single dict mapping node_id → node_data including children list.
    Assumes levels are ordered: root first, then level 1, level 2, ...
    """
    tree = {}

    for level_file in distr_level_files:
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

    for level_file in binned_level_files:
        data = json.loads(Path(level_file).read_text())
        for node in data:
            node_id = node["id"]
            assert tree[node_id]["ground_truth"]["label"] == node["label"], ("Node labels from ground-truth"
                                                                             " and LLM results do not match!")
            tree[node_id]["binned"] = dict(node)

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

    # Subsampling income distribution
    bin_edges, counts, sub_prob = subsample_bins(
        bin_edges=np.array(node["ground_truth"]["bin_edges"]),
        counts=np.array(node["ground_truth"]["counts"]),
        lower_limit=lower_limit,
        upper_limit=upper_limit,
        subsampling_factor=subsampling_factor)

    # Create histogram
    distr_plt_path = create_income_distr_histogram(
        bin_edges=bin_edges,
        counts=counts,
        mean=node["ground_truth"]["target_mean"],
        out_path=out_path.parent,
        node_id=node_id)

    # Create binned histogram
    binned_plt_path = create_few_bin_histogram(
        bin_edges=np.array(node["binned"]["bin_edges"]),
        counts=np.array(node["binned"]["counts"]),
        mean=node["binned"]["target_mean"],
        out_path=out_path.parent,
        node_id=node_id,
        equal_bar_width=True,
        llm_color_scheme=False,
    )

    # TeX code for figure
    figure = r"""\begin{minipage}{0.49\linewidth}
\input{"""
    figure += str(distr_plt_path)
    figure += r"""}
\end{minipage}
\hfill
\begin{minipage}{0.49\linewidth}
\input{"""
    figure += str(binned_plt_path)
    figure += r"""}
\end{minipage}"""

    # Root node needs leading backslash
    if root:
        node_tex = ("\\node[treenode] " + "{\n"
                    + f"{pad + single_pad}{figure}" + "\n}")
        edge_tex = ""
    else:
        short_label = node["ground_truth"]["short_label"].replace("_", "-").replace("PARTITIONED", "PART")
        node_tex = ("node[treenode] " + "{\n"
                    + f"{pad + single_pad}{figure}" + f"\n{pad}}}")
        edge_tex = "edge from parent node[edgelabel, pos=0.3] " + r"{\textbf{" + short_label + "}}"

    # Recurse through children
    for child_id in sorted(node.get("children", [])):
        child = emit_node(tree=tree, node_id=child_id, out_path=out_path,
                          lower_limit=lower_limit, upper_limit=upper_limit, subsampling_factor=subsampling_factor,
                          indent=indent + 2)
        node_tex += (f"\n{pad + single_pad}child {{"
                     f"\n{pad + 2 * single_pad}{child}" + f"\n{pad + single_pad}" + "}")

    # Add edge to parent
    node_tex += "\n" + pad + edge_tex

    return node_tex


def create_income_distr_histogram(
        bin_edges: np.ndarray,
        counts: np.ndarray,
        mean: float,
        out_path: Path,
        node_id: int) -> Path:
    # Setup
    setup_pgf_plots(font_size=14)

    # Create figure
    plt.figure(figsize=(2.5, 1.9))

    # Ground-truth
    width = 0.9 * (bin_edges[1] - bin_edges[0])
    gt_bin_centers = (bin_edges[1:] + bin_edges[:-1]) / 2
    plt.bar(gt_bin_centers, counts, width=width, align="edge", alpha=0.6, label="GT", color="blue")
    # plt.axvline(mean, color="blue", linestyle="--", linewidth=2, label="GT mean")

    # Format
    plt.ylabel("Density", labelpad=10)
    plt.tick_params(axis="x", pad=4)
    plt.tick_params(axis="y", pad=4)
    plt.tight_layout()
    ax = plt.gca()
    ax.xaxis.set_major_formatter(
        mticker.FuncFormatter(lambda x, _: f"{x / 1_000:.0f}k")
    )
    # Hacky way of placing x-label
    # plt.xlabel("Inc.", labelpad=0, loc="right")
    # ax.xaxis.set_label_coords(1.12, -0.08)
    ax.text(
        0.97,
        -0.15,
        "Inc.",
        transform=ax.transAxes,
        ha="left",
        va="center",
        clip_on=False,
    )

    # Save
    plot_name = f"hist_{node_id}.pgf"
    plot_path = out_path / plot_name
    plt.savefig(plot_path, bbox_inches="tight", pad_inches=0.008)

    return plot_path


def create_few_bin_histogram(
        bin_edges: np.ndarray,
        counts: np.ndarray,
        mean: float,
        out_path: Path,
        node_id: int,
        equal_bar_width: bool,
        llm_color_scheme: bool,
        prefix: str = "",
) -> Path:
    # Setup
    setup_pgf_plots(font_size=15)

    # Create figure
    plt.figure(figsize=(2.5, 1.9))

    # Ground-truth
    if equal_bar_width:
        # x-axis
        x = np.arange(len(counts))
        labels = [chr(ord("A") + i) for i in range(len(counts))]

        # Normalize counts to one
        counts = counts / sum(counts)

        # Sample colors from a colormap
        colors = get_histogram_colors(llm=llm_color_scheme, num_colors=len(counts))

        # Plot
        plt.bar(x, counts, align="center", alpha=0.85, color=colors, edgecolor="none", linewidth=0)
        plt.xticks(x, labels)
        plt.tick_params(axis="x", pad=4)
        plt.ylabel("Probability", labelpad=4)
    else:
        bin_widths = np.diff(bin_edges)
        plt.bar(bin_edges[:-1], counts, width=bin_widths, align="edge", alpha=0.6, label="GT", color="blue",
                edgecolor="none", linewidth=0)
        plt.axvline(mean, color="blue", linestyle="--", linewidth=2, label="GT mean")

        # Format
        plt.legend()

    # Save
    plot_name = f"{prefix}binned_hist_{node_id}.pgf"
    plot_path = out_path / plot_name
    plt.savefig(plot_path, bbox_inches="tight", pad_inches=0.008)

    return plot_path


if __name__ == "__main__":
    distr_experiment_name = "ground_truth_income"
    distr_datetime = "2026-04-16__18-20-00"
    distr_cluster = True
    binned_experiment_name = "ground_truth_few_bins"
    binned_datetime = "2026-04-17__15-30-00"
    binned_cluster = True

    # Construct paths
    distr_base_path = get_results_dir(experiment_name=distr_experiment_name, cluster=distr_cluster, use_timestamp=True,
                                      timestamp=distr_datetime)
    distr_json_path = distr_base_path / "tree"
    binned_base_path = get_results_dir(experiment_name=binned_experiment_name, cluster=binned_cluster,
                                       use_timestamp=True,
                                       timestamp=binned_datetime)
    binned_json_path = binned_base_path / "tree"

    combined_name = "gt_combined_tree"
    datetime = datetime.now().strftime("%Y-%m-%d__%H-%M-%S")
    base_path = get_results_dir(
        experiment_name=combined_name,
        cluster=False,
        use_timestamp=True,
        create_dir=True,
    )
    template_dir = get_latex_template_dir()
    template_path = template_dir / "combined_income_tree/template_comb_tree_two.tex"
    out_path = base_path / "main.tex"

    # Generate TikZ tree from JSON and template
    generate_combined_tree(
        distr_json_path=distr_json_path,
        binned_json_path=binned_json_path,
        template_path=template_path,
        out_path=out_path,
        lower_limit=-20_000,
        upper_limit=300_000,
        subsampling_factor=5,
    )
