import yaml
import matplotlib.pyplot as plt

from experiments_acs.aggregation_refinement.plot.plot_refinement_thresholds import process_logs, \
    plot_aggregation_refinement
from file_logging.read_and_write_json import save_as_json
from utility.plot_style import set_plot_style
from utility.directories import get_plot_dir


if __name__ == "__main__":
    # Plot directory
    plot_base_dir = get_plot_dir()
    plot_dir = plot_base_dir / "aggregation_refinement_ablation_split_crit"
    plot_dir.mkdir(parents=True, exist_ok=True)

    # Set style
    set_plot_style(sans_serif=False, factor=1.225)

    # Line width
    plt.rcParams.update({
        "axes.linewidth": 2,
        "grid.linewidth": 1.75,
    })

    # Load experiment logs
    with open("experiments_refinement_threshold.yaml", "r") as f:
        logs = yaml.safe_load(f)

    # Toggle to use new jointly elicited priors
    load_joint_llm_priors = True

    # Process results
    results, experiment_name = process_logs(logs=logs,
                                            overwrite_cluster_to_false=False,
                                            load_joint_llm_priors=load_joint_llm_priors)

    # Filename
    filename = f"thresholds_{"llm_prior" if load_joint_llm_priors else "gt_prior"}.json"

    # Save results to JSON file
    results_path = save_as_json(
        data=results,
        experiment="figure_aggregation_gain_improved_split_criteria",
        filename=filename,
        timestamp=False)
    print(f"\nResults saved to {results_path}")

    # Plot
    plot_aggregation_refinement(results=results,
                                plot_dir=plot_dir,
                                experiment_name=experiment_name,
                                load_joint_llm_priors=load_joint_llm_priors)
