import yaml
import numpy as np
import matplotlib.pyplot as plt

from matplotlib.colors import ListedColormap
from experiments_acs.micro_macro_multi.plot_implicit.plot_error_gain import compute_mtm_error_list
from experiments_acs.micro_macro_multi.plot_implicit.plot_helper import load_all_results
from file_logging.read_and_write_json import save_as_json
from utility.plot_style import set_plot_style
from utility.directories import get_plot_dir


# Set style
set_plot_style(sans_serif=False, factor=1.225)

# Line width
plt.rcParams.update({
    "axes.linewidth": 2,
    "grid.linewidth": 1.75,
})


def main_processing(
        models: dict,
        selected_models_win_rate: list | None,
        selected_threshold_win_rate: list | None,
) -> tuple[dict, dict, dict]:

    # Load experiment results
    exp_results = load_experiment_results(models=models)

    # Process results
    detailed_results = process_results(exp_results=exp_results)

    # Extract plot results
    plot_results = extract_plot_results(detailed_results=detailed_results)

    # Assertions
    first_model = list(plot_results.keys())[0]
    first_threshold_list = plot_results[first_model]["threshold_list"]
    for model, results in plot_results.items():
        assert results["threshold_list"] == first_threshold_list, (f"Threshold lists do not match between models. "
                                                                   f"Model {model} has a different threshold "
                                                                   f"list than model {first_model}.")

    # Save results
    save_as_json(
        data=plot_results,
        experiment="micro_macro_aggregated",
        filename="ACS_thresholded_income.json"
    )

    if selected_models_win_rate is None:
        models = list(plot_results.keys())
    else:
        models = selected_models_win_rate

    if selected_threshold_win_rate is None:
        thresholds = plot_results[models[0]]["threshold_list"]
    else:
        thresholds = selected_threshold_win_rate

    # Compute win rate results
    win_rate_results = compute_win_rate_results(
        detailed_results=detailed_results,
        selected_models=models,
        selected_thresholds=thresholds,
    )

    # Save results
    save_as_json(
        data=win_rate_results,
        experiment="micro_macro_aggregated",
        filename="ACS_thresholded_income_win_rate.json"
    )

    return detailed_results, plot_results, win_rate_results


def load_experiment_results(
        models: dict,
) -> dict:
    # Load micro-to-macro experiment logs
    with open("../plot_implicit/exp_micro_macro.yaml", "r") as f:
        full_logs = yaml.safe_load(f)

    mtm_logs = full_logs["experiment_logs"]

    # Load experiment data
    exp_results = {}
    for model_info in models.values():
        # Find log entry
        mtm_log_entry = next((item for item in mtm_logs["experiments"] if item["model"] == model_info["model"]), None)

        # Load results
        mtm_results, baseline_results = load_all_results(
            mtm_experiment_name=mtm_logs["experiment_name"],
            mtm_timestamp=mtm_log_entry["timestamp"],
            mtm_cluster=mtm_log_entry["cluster"],
            baseline_experiment_name="micro_macro_aggregation_baseline",
            baseline_experiment_log_file=f"../plot_implicit/{model_info['baseline_yaml']}.yaml",
            load_joint_llm_priors=True,
        )

        # Add to results
        exp_results[model_info["model"]] = {
            "model": model_info["model"],
            "baseline_results": baseline_results,
            "mtm_results": mtm_results,
        }

    return exp_results


def process_results(
        exp_results: dict
) -> dict:
    # Iterate over models
    detailed_results = {}
    for model, results in exp_results.items():
        print(f"\nModel: {model}")
        detailed_results[model] = {}

        # Extract results
        mtm_results = results["mtm_results"]
        baseline_results = results["baseline_results"]

        # Iterate over thresholds and compute error gain
        for mtm_res in mtm_results:
            income_threshold = mtm_res["income_threshold"]
            print(f"Income threshold: {income_threshold}")

            # Find corresponding baseline result
            baseline_res = next((b for b in baseline_results if b["metadata"]["income_threshold"] == income_threshold),
                                None)

            # Get error lists
            direct_error_list = baseline_res["aggregated_tree"][0]["aggregated_estimation_error_list"]
            mtm_error_list = compute_mtm_error_list(mtm_result=mtm_res)

            # Assertions
            assert np.isclose(mtm_res["ground_truth_probability"],
                              baseline_res["aggregated_tree"][0]["aggregated_ground_truth_probability"]), (
                "Ground truth probabilities do not match between MtM and baseline results.")

            # Average error lists
            avg_direct_error = np.average(direct_error_list)
            avg_mtm_error = np.average(mtm_error_list)

            # Compute error gain
            error_gain = (np.average(direct_error_list) - np.average(mtm_error_list)) / np.average(direct_error_list)

            # Store results
            detailed_results[model][income_threshold] = {
                "avg_direct_error": float(avg_direct_error),
                "avg_mtm_error": float(avg_mtm_error),
                "direct_error_lists": direct_error_list,
                "mtm_error_lists": mtm_error_list,
                "error_gain": float(error_gain),
            }

    return detailed_results


def extract_plot_results(
        detailed_results: dict
) -> dict:
    plot_results = {}
    for model, results in detailed_results.items():
        avg_error_gain = np.average([item["error_gain"] for item in results.values()])
        threshold_list = list(results.keys())

        # Store results
        plot_results[model] = {
            "model": model,
            "average_error_gain": float(avg_error_gain),
            "threshold_list": threshold_list,
        }

    return plot_results


def compute_win_rate_results(
        detailed_results: dict,
        selected_models: list,
        selected_thresholds: list,
) -> dict:
    # Build win-event matrix
    win_events = np.full((len(selected_models), len(selected_thresholds)), np.nan)

    for i, model in enumerate(selected_models):
        model_res = detailed_results[model]
        for j, threshold in enumerate(selected_thresholds):
            threshold_res = model_res[threshold]
            win_events[i, j] = threshold_res["avg_mtm_error"] < threshold_res["avg_direct_error"]

    # Convert to list
    win_events = win_events.tolist()

    # Collect results
    win_rate_results = {
        "win_events": win_events,
        "models": selected_models,
        "thresholds": selected_thresholds,
    }

    return win_rate_results


def plot_avg_error_gains(
        plot_results: dict,
):
    # Figure
    plt.figure(figsize=(10, 6))
    plt.title("Average Error Gain Across Models")

    # Plot
    models = []
    y_list = []
    count = 0
    for item in reversed(list(plot_results.values())):
        models.append(item["model"])
        y_list.append(count)

        # Horizontal lines for GT results
        plt.hlines(
            y=count,
            xmin=0,
            xmax=item["average_error_gain"],
            color="green" if item["average_error_gain"] > 0 else "red",
            linewidth=2.5,
            zorder=2,
        )

        # Endpoint marker for GT
        plt.plot(
            item["average_error_gain"],
            count,
            marker="o",
            color="green" if item["average_error_gain"] > 0 else "red",
            markersize=7,
            zorder=2,
        )

        # Increment count for next model
        count += 1

    # Formatting
    plt.yticks(ticks=y_list, labels=models)
    plt.xlabel("Average Error Gain of MtM over Direct Prompting")
    plt.grid()
    plt.tight_layout()

    # Plot directory
    plot_base_dir = get_plot_dir()
    plot_dir = plot_base_dir / "micro_macro_aggregated"
    plot_dir.mkdir(parents=True, exist_ok=True)

    # Save plot
    plt.savefig(plot_dir / "ACS_thresholded_income_avg_error_gain.png", dpi=300, bbox_inches="tight")
    plt.savefig(plot_dir / "ACS_thresholded_income_avg_error_gain.pdf")

    # Show plot
    plt.show()


def plot_win_rates(
        win_rate_results: dict,
):
    # Extract axes lists
    models = win_rate_results["models"]
    threshold_list = win_rate_results["thresholds"]

    # Red = Direct wins, Green = MtM wins
    cmap = ListedColormap(["#d62728", "#2ca02c"])

    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(9, 7))

    # Plot as heatmap
    ax.imshow(
        win_rate_results["win_events"],
        aspect="auto",
        cmap=cmap,
        vmin=0,
        vmax=1,
    )

    # Title
    avg_win_rate = np.average(win_rate_results["win_events"])
    ax.set_title(rf"Win rate = {100 * avg_win_rate:.1f}\,\%")

    ax.set_xticks(np.arange(len(threshold_list)))
    ax.set_xticklabels([f"{t}" for t in threshold_list])

    ax.set_yticks(np.arange(len(models)))
    ax.set_yticklabels([model.split("/")[1] for model in models])

    # Grid lines between cells
    ax.set_xticks(np.arange(-0.5, len(threshold_list), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(models), 1), minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=0.8)

    # Hide minor tick marks
    ax.tick_params(which="minor", bottom=False, left=False)
    ax.set_xlabel("Thresholds")
    # ax.set_ylabel("Models")

    fig.tight_layout()

    # Plot directory
    plot_base_dir = get_plot_dir()
    plot_dir = plot_base_dir / "micro_macro_aggregated"
    plot_dir.mkdir(parents=True, exist_ok=True)

    # Save plot
    plt.savefig(plot_dir / "ACS_thresholded_income_win_rate.png", dpi=300, bbox_inches="tight")
    plt.savefig(plot_dir / "ACS_thresholded_income_win_rate.pdf")

    plt.show()


if __name__ == "__main__":
    # Specify experiments
    models = {
        "gpt54": {
            "model": "openai/gpt-5.4",
            "baseline_yaml": "exp_baseline_gpt54"
        },
        "gpt4o_mini": {
            "model": "openai/gpt-4o-mini",
            "baseline_yaml": "exp_baseline_gpt4o_mini"
        },
        "opus": {
            "model": "anthropic/claude-opus-4.7",
            "baseline_yaml": "exp_baseline_opus"
        },
        "sonnet": {
            "model": "anthropic/claude-sonnet-4.6",
            "baseline_yaml": "exp_baseline_sonnet"
        },
        "gemini_pro": {
            "model": "google/gemini-3.1-pro-preview",
            "baseline_yaml": "exp_baseline_gemini_pro"
        },
        "grok": {
            "model": "x-ai/grok-4.20",
            "baseline_yaml": "exp_baseline_grok"
        }
    }

    # Run main script
    detailed_results, plot_results, win_rate_results = main_processing(
        models=models,
        selected_models_win_rate=None,
        selected_threshold_win_rate=None,
    )

    # Plot 1: Averaged error gain
    plot_avg_error_gains(plot_results=plot_results)

    # Plot 2: Win rates
    plot_win_rates(win_rate_results=win_rate_results)
