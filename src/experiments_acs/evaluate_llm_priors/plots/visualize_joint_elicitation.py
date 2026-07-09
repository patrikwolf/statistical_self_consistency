import json
import numpy as np
import matplotlib.pyplot as plt

from utility.directories import get_plot_dir, get_results_dir
from utility.plot_style import set_plot_style
from experiments_acs.tree_income_threshold.latex.aggregation_helper import build_aggregated_threshold_tree
from experiments_acs.tree_income_threshold.latex.convert_shards_to_json_tree import convert_shards_to_json_tree


set_plot_style(sans_serif=True)


def main(
        plot_prefix: str,
        plot_name: str,
        plot_dir_name: str,
        experiment_name: str,
        timestamp: str,
        cluster: bool,
):
    # Convert shards to JSON tree
    _, _, num_levels = convert_shards_to_json_tree(
        experiment_name=experiment_name,
        datetime=timestamp,
        cluster=cluster,
    )

    # Construct paths
    base_path = get_results_dir(experiment_name=experiment_name, cluster=cluster, use_timestamp=True, timestamp=timestamp)
    json_path = base_path / "tree"

    # Load all level files
    level_files = sorted(json_path.glob("level_*.json"), key=lambda p: int(p.stem.split("_")[1]))
    print(f"Found {len(level_files)} level files: {[f.name for f in level_files]}")

    # Build node and children mappings
    load_joint_llm_priors = True
    tree = build_aggregated_threshold_tree(level_files=level_files, load_joint_llm_priors=load_joint_llm_priors)

    # Load metadata
    meta_path = json_path / "meta.json"
    metadata = json.loads(meta_path.read_text())

    # Extract plot results
    plot_results = {}
    for level, data in tree.items():
        if level == 0:
            continue

        # Extract ground-truth data
        x = np.arange(len(data["ground_truth_prior_list"]))
        gt_priors = data["ground_truth_prior_list"]
        normalized_llm_priors = data["priors"]

        # Sanity checks
        assert len(gt_priors) == len(normalized_llm_priors)
        assert np.isclose(sum(gt_priors), 1.0)
        assert np.isclose(sum(normalized_llm_priors), 1.0)

        # Compute TV distance
        tv_distance = 0.5 * np.sum(np.abs(np.array(gt_priors) - np.array(normalized_llm_priors)))

        # Store results for plotting
        plot_results[level] = {
            "x": x.tolist(),
            "ground_truth_prior_list": gt_priors,
            "normalized_llm_priors": normalized_llm_priors,
            "tv_distance": float(tv_distance),
        }

    # Plot
    plot_prior_comparison(
        plot_results=plot_results,
        tree=tree,
        metadata=metadata,
        plot_name=plot_name,
        plot_dir_name=plot_dir_name,
        plot_prefix=plot_prefix,
        timestamp=timestamp,
    )


def plot_prior_comparison(
        plot_results: dict,
        tree: dict,
        metadata: dict,
        plot_name: str,
        plot_dir_name: str,
        plot_prefix: str,
        timestamp: str,
):
    # Plot directory
    plot_base_dir = get_plot_dir()
    plot_dir = plot_base_dir / plot_dir_name
    plot_dir.mkdir(parents=True, exist_ok=True)

    title = (r"\huge\bfseries Jointly elicited LLM-estimated Priors versus Ground-Truth Priors at Different Levels of the Tree"
             "\n"
             r"\normalsize "
             f"Timestamp: {timestamp}"
             "\n"
             r"\normalsize "
             f"Model: {metadata['model']}"
             f", reasoning effort: {metadata['reasoning_effort']}"
             "\n"
             r"\normalsize "
             f"Prompting scheme: '{metadata['prompting_scheme']}'"
             "\n"
             r"\normalsize "
             f"Split: '{metadata['decomposition_attribute_list']}'"
             "\n"
             r"\normalsize "
             f"Income threshold: {metadata['income_threshold']} USD"
             "\n"
             )

    num_levels = len(tree) - 1

    # Plot
    fig, axs = plt.subplots(1, num_levels, figsize=(7 * num_levels, 8))
    fig.suptitle(title)

    for level, data in tree.items():
        if level == 0:
            continue

        # Extract data
        x = np.array(plot_results[level]["x"])
        gt_priors = plot_results[level]["ground_truth_prior_list"]
        normalized_llm_priors = plot_results[level]["normalized_llm_priors"]

        # Bar plots
        bar_width = 0.4
        axs[level - 1].bar(x - bar_width / 2, normalized_llm_priors, width=bar_width, label="LLM priors")
        axs[level - 1].bar(x + bar_width / 2, gt_priors, width=bar_width, label="GT priors")

        # Formatting
        axs[level - 1].set_title(f"Level {level}, " + r"$d_\mathrm{TV} = " + f"{plot_results[level]['tv_distance']:.3f}$")
        axs[level - 1].set_xticks(range(len(x)), [(integer + 1) for integer in x])
        axs[level - 1].grid()

    # Formatting
    axs[0].legend()
    plt.tight_layout()

    # Prefix
    if len(plot_prefix) > 0:
        plot_prefix = f"{plot_prefix}_"

    # Save plot
    plt.savefig(plot_dir / f"{plot_prefix}{timestamp}_{plot_name}.png", dpi=300)
    plt.savefig(plot_dir / f"{plot_prefix}{timestamp}_{plot_name}.pdf")

    # Show
    plt.show()


if __name__ == "__main__":
    plot_prefix = ""
    plot_dir_name = "llm_prior_tree"
    plot_name = "joint_vs_gt_prior_comparison"

    # Experiment
    experiment_name = "aggregation_refinement"
    timestamp = "2026-04-27__20-26-26"
    cluster = True

    # Run main scripts
    main(
        plot_prefix=plot_prefix,
        plot_dir_name=plot_dir_name,
        plot_name=plot_name,
        experiment_name=experiment_name,
        timestamp=timestamp,
        cluster=cluster,
    )
