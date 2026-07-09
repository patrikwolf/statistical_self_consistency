import math
import yaml
import numpy as np
import matplotlib.pyplot as plt

from matplotlib.colors import ListedColormap
from utility.plot_style import set_plot_style
from utility.directories import get_results_dir, get_plot_dir
from file_logging.read_and_write_json import read_json, save_as_json


# Set style
set_plot_style(sans_serif=False, factor=1.225)

# Line width
plt.rcParams.update({
    "axes.linewidth": 2,
    "grid.linewidth": 1.75,
})


def main_processing(
        experiment_name: str,
        logs: str,
        models_for_avg_err: list | None,
        question_for_avg_err: list,
        countries_for_avg_err: list,
        models_win_rate: list | None,
        questions_win_rate: list,
        countries_win_rate: list,
) -> tuple[dict, dict, dict]:

    # Load sharded results
    exp_results = load_sharded_results(
        experiment_name=experiment_name,
        logs_more_countries=logs,
    )

    # Shared processing pipeline
    return shared_processing(
        exp_results=exp_results,
        models_for_avg_err=models_for_avg_err,
        question_for_avg_err=question_for_avg_err,
        countries_for_avg_err=countries_for_avg_err,
        models_win_rate=models_win_rate,
        questions_win_rate=questions_win_rate,
        countries_win_rate=countries_win_rate,
    )


def main_processing_with_merging(
        experiment_name: str,
        log_files: list,
        logs_more_countries: str,
        models_for_avg_err: list | None,
        question_for_avg_err: list,
        countries_for_avg_err: list,
        models_win_rate: list | None,
        questions_win_rate: list,
        countries_win_rate: list,
) -> tuple[dict, dict, dict]:

    # Load individual results
    exp_results = load_individual_results(
        experiment_name=experiment_name,
        log_files=log_files,
    )

    # Load sharded results
    raw_results = load_sharded_results(
        experiment_name=experiment_name,
        logs_more_countries=logs_more_countries,
    )

    # Merge results
    merge_results(
        exp_results=exp_results,
        raw_results=raw_results,
    )

    # Shared processing pipeline
    return shared_processing(
        exp_results=exp_results,
        models_for_avg_err=models_for_avg_err,
        question_for_avg_err=question_for_avg_err,
        countries_for_avg_err=countries_for_avg_err,
        models_win_rate=models_win_rate,
        questions_win_rate=questions_win_rate,
        countries_win_rate=countries_win_rate,
    )


def shared_processing(
        exp_results: dict,
        models_for_avg_err: list | None,
        question_for_avg_err: list,
        countries_for_avg_err: list,
        models_win_rate: list | None,
        questions_win_rate: list,
        countries_win_rate: list,
) -> tuple[dict, dict, dict]:
    # Process results
    detailed_results = process_results(exp_results=exp_results)

    # Assert that we average over the same set of questions and same countries for all models
    first_model = list(detailed_results.keys())[0]
    first_question = list(detailed_results[first_model].keys())[0]
    first_question_list = list(detailed_results[first_model].keys())
    first_country_list = list(detailed_results[first_model][first_question].keys())
    for model, model_res in detailed_results.items():
        assert set(model_res.keys()) == set(first_question_list), (
            f"Model {model} has different set of questions than first model {first_model}."
            f"Got: {set(model_res.keys())}, expected: {set(first_question_list)}"
        )

        for question, mtm_res in model_res.items():
            assert set(mtm_res.keys()) == set(
                first_country_list), (f"Model {model} question {question} has different set of "
                                      f"countries than first model {first_model} question {first_question}")

    # Extract plot results
    plot_results = extract_plot_results(
        detailed_results=detailed_results,
        selected_models=models_for_avg_err,
        selected_questions=question_for_avg_err,
        selected_countries=countries_for_avg_err,
    )

    # Save results
    save_as_json(
        data=plot_results,
        experiment="micro_macro_aggregated",
        filename="GOQA_thresholded.json"
    )

    # Compute overall win rate
    win_array = [country_res["mtm_error"] < country_res["direct_error"]
                 for model_res in detailed_results.values()
                 for question_res in model_res.values()
                 for country_res in question_res.values()
                 ]
    win_rate_mtm = np.mean(win_array)
    print(f"Length: {len(win_array)}")
    print(f"Win rate of micro-to-macro prompting compared to direct prompting: {100 * win_rate_mtm:.2f}%")

    # Compute win rate results
    win_rate_results = compute_win_rate_results(
        detailed_results=detailed_results,
        selected_models=models_win_rate,
        selected_questions=questions_win_rate,
        selected_countries=countries_win_rate,
    )

    # Save results
    save_as_json(
        data=win_rate_results,
        experiment="micro_macro_aggregated",
        filename="GOQA_thresholded_all_win_rates.json"
    )

    return plot_results, detailed_results, win_rate_results


def load_individual_results(
        experiment_name: str,
        log_files: list,
) -> dict:

    exp_results = {}
    for log_file in log_files:
        with open(log_file, "r") as f:
            model_logs = yaml.safe_load(f)

        model = model_logs["log"]["model"]
        exp_results[model] = {}

        for experiment_log in model_logs["log"]["experiments"]:
            # Load experiment results
            mtm_results = read_mtm_results(
                experiment_name=experiment_name,
                datetime=experiment_log["timestamp"],
                cluster=experiment_log["cluster"],
                question_identifier=int(experiment_log["question"]),
            )

            assert mtm_results["question_identifier"] == f"Q{experiment_log['question']}", "Question mismatch"

            # Add to dictionary
            exp_results[model][experiment_log["question"]] = mtm_results

    return exp_results


def read_mtm_results(experiment_name: str, datetime: str, cluster: bool, question_identifier: int) -> dict:
    # Directory
    results_dir = get_results_dir(
        experiment_name=experiment_name,
        cluster=cluster,
        use_timestamp=True,
        timestamp=datetime,
        create_dir=False)

    # Check if results_dir exists
    if not results_dir.is_dir():
        raise NotADirectoryError(f"Specified results directory is not a valid directory: {results_dir}")

    data = read_json(
        experiment_name=f"{experiment_name}/{datetime}",
        filename="summary_results.json",
        cluster=cluster,
    )

    assert data["question_identifier"] == f"Q{question_identifier}", (f"Expected question {question_identifier}, "
                                                                      f"got {data['question_identifier']}")

    return data


def load_sharded_results(
        experiment_name: str,
        logs_more_countries: str,
) -> dict:
    # Load logs
    with open(logs_more_countries, "r") as f:
        logs_more = yaml.safe_load(f)

    # Iterate over result logs
    raw_results = {}
    for model, timestamp in logs_more["log"]["experiments"].items():
        print(f"\nLoading sharded results for model: {model} with timestamp: {timestamp}")
        raw_results[model] = {}

        # Directory
        results_dir = get_results_dir(
            experiment_name=experiment_name,
            cluster=True,
            use_timestamp=True,
            timestamp=timestamp,
            create_dir=False)

        # Check if directory exists
        if not results_dir.exists() or not results_dir.is_dir():
            raise FileNotFoundError(f"Directory does not exist: {results_dir}")

        # Load shards
        shards = sorted(results_dir.glob("shard_*_summary_results.json"))

        # Process each shard
        for idx, shard in enumerate(shards):
            # Load results
            data = read_json(
                experiment_name=f"{experiment_name}/{timestamp}",
                filename=shard.name,
                cluster=True,
            )

            question_identifier = data["question_identifier"]
            question_id = question_identifier.removeprefix("Q")

            # Assertion
            assert data["model"] == model, f"Expected model {model}, got {data['model']}"

            # Add to list
            raw_results[model][question_id] = data

    return raw_results


def merge_results(
        exp_results: dict,
        raw_results: dict
):
    for model, model_res in raw_results.items():
        for question, question_res in raw_results[model].items():

            # Find corresponding results in exp_results
            exp_res = exp_results[model][question]

            # Assertions
            assert question_res["model"] == exp_res["model"]
            assert question_res["reasoning_effort"] == exp_res["reasoning_effort"]
            assert question_res["prompting_scheme"] == exp_res["prompting_scheme"]
            assert question_res["num_of_samples"] == exp_res["num_of_samples"]
            assert question_res["question_identifier"] == exp_res["question_identifier"]
            assert question_res["question"] == exp_res["question"]
            assert question_res["options"] == exp_res["options"]
            assert question_res["direct_template"] == exp_res["direct_template"]
            assert question_res["micro_to_macro_template"] == exp_res["micro_to_macro_template"]

            assert question_res["reasoning_effort"] == exp_res["reasoning_effort"]
            assert question_res["reasoning_effort"] == exp_res["reasoning_effort"]
            assert question_res["reasoning_effort"] == exp_res["reasoning_effort"]
            assert question_res["reasoning_effort"] == exp_res["reasoning_effort"]
            assert question_res["reasoning_effort"] == exp_res["reasoning_effort"]

            # Merge selections
            exp_res["selections"] |= question_res["selections"]

            # Merge evaluation results
            exp_res["evaluation_results"].extend(question_res["evaluation_results"])

    return


def process_results(
        exp_results: dict,
) -> dict:
    detailed_results = {}
    for model, model_res in exp_results.items():
        print(f"\nProcessing results for model: {model}")
        detailed_results[model] = {}

        # Iterate over all questions
        for question, mtm_res in model_res.items():
            print(f"  --> Processing question: {question}")
            detailed_results[model][question] = {}

            # Iterate over all countries
            for item in mtm_res["evaluation_results"]:
                country = item["country"]
                print(f"      --> Country: {country}")

                # Ground-truth distribution
                idx_yes = 0
                survey_distribution = item["ground_truth"][idx_yes]

                # Compute error for direct prompting
                direct_avg_llm_prediction = item["direct_prompting"]["avg_llm_prediction"]
                direct_error = abs(survey_distribution - direct_avg_llm_prediction)

                # Compute error for micro-to-macro prompting
                mtm_avg_llm_prediction = item["micro_to_macro"]["avg_llm_prediction"]
                mtm_error = abs(survey_distribution - mtm_avg_llm_prediction)

                # Store results
                detailed_results[model][question][country] = {
                    "direct_error": direct_error,
                    "mtm_error": mtm_error,
                }

    return detailed_results


def extract_plot_results(
        detailed_results: dict,
        selected_models: list | None,
        selected_questions: list,
        selected_countries: list,
) -> dict:
    plot_results = {}
    for model, model_res in detailed_results.items():
        if selected_models is not None and model not in selected_models:
            print(f"  --> Skipping model '{model}' since it is not in the list of selected models")
            continue
        direct_error_list = []
        mtm_error_list = []
        for question_id, question_res in detailed_results[model].items():
            if question_id not in selected_questions:
                print(f"  --> Skipping question '{question_id}' since it is not in the list of selected questions")
                continue
            for country, country_res in question_res.items():
                if country not in selected_countries:
                    print(f"  --> Skipping country '{country}' since it is not in the list of selected countries")
                    continue

                # Add to list
                direct_error_list.append(country_res["direct_error"])
                mtm_error_list.append(country_res["mtm_error"])

        # Compute average error across questions and countries for each prompting method
        average_direct_error = np.mean(direct_error_list)
        average_mtm_error = np.mean(mtm_error_list)
        relative_gain = (average_direct_error - average_mtm_error) / average_direct_error

        # Store results
        plot_results[model] = {
            "model": model,
            "average_direct_error": average_direct_error,
            "average_mtm_error": average_mtm_error,
            "relative_gain": relative_gain,
        }

    return plot_results


def plot_average_error(
        plot_results: dict,
):
    # Figure
    plt.figure(figsize=(10, 10))
    ax = plt.gca()
    plt.title("Average Error Across Models\n\n")

    # Extract results
    x = np.arange(len(plot_results))
    models = []
    direct_errors = []
    mtm_errors = []
    for item in plot_results.values():
        models.append(item["model"])
        direct_errors.append(item["average_direct_error"])
        mtm_errors.append(item["average_mtm_error"])

    direct_errors = np.array(direct_errors)
    mtm_errors = np.array(mtm_errors)

    # Relative reduction in average error: positive means MtM improves
    relative_reductions = (direct_errors - mtm_errors) / direct_errors

    # Plot
    bar_width = 0.4
    plt.bar(x - bar_width / 2, direct_errors, width=bar_width, label="Direct prompting")
    plt.bar(x + bar_width / 2, mtm_errors, width=bar_width, label="Micro to macro")

    # Add relative reduction annotations
    for i, rel_red in enumerate(relative_reductions):
        y = max(direct_errors[i], mtm_errors[i])
        ax.text(
            x[i],
            y + 0.003,
            rf"${100 * rel_red:.1f}\,\%$",
            color="green" if rel_red > 0 else "red",
            ha="center",
            va="bottom",
            fontsize=17,
            bbox=dict(
                facecolor="white",
                edgecolor="lightgray",
                linewidth=0.5,
                alpha=0.9,
                boxstyle="round, pad=0.3",
            )
        )

    # Formatting
    plt.xticks(ticks=x, labels=models, rotation=90)
    bottom, top = plt.ylim()
    plt.ylim(bottom, top * 1.1)
    plt.grid()
    plt.tight_layout()

    # Legend
    handles, labels = ax.get_legend_handles_labels()
    plt.legend(handles,
               labels,
               loc="upper center",
               bbox_to_anchor=(0.5, 1.08),
               ncol=5,
               prop={"size": 11.5},
               columnspacing=0.8,
               handletextpad=0.4,
               handlelength=1.72,
               )

    # Plot directory
    plot_base_dir = get_plot_dir()
    plot_dir = plot_base_dir / "micro_macro_aggregated"
    plot_dir.mkdir(parents=True, exist_ok=True)

    # Save plot
    plt.savefig(plot_dir / "GOQA_thresholded.png", dpi=300, bbox_inches="tight")
    plt.savefig(plot_dir / "GOQA_thresholded.pdf")

    # Show plot
    plt.show()


def compute_win_rate_results(
        detailed_results: dict,
        selected_models: list | None,
        selected_questions: list,
        selected_countries: list,
) -> dict:

    # Compute win events
    win_rate_results = {}
    for model in detailed_results.keys():
        if selected_models is not None and model not in selected_models:
            print(f"  --> Skipping model '{model}' since it is not in the list of selected models")
            continue

        # Build win-event matrix
        win_events = np.full((len(selected_questions), len(selected_countries)), np.nan)

        for i, question in enumerate(selected_questions):
            question_res = detailed_results[model][question]
            for j, country in enumerate(selected_countries):
                if country not in question_res:
                    continue
                country_res = question_res[country]
                win_events[i, j] = country_res["mtm_error"] < country_res["direct_error"]

        # Convert to list
        win_events = win_events.tolist()

        # Add to dict
        win_rate_results[model] = {
            "win_events": win_events,
            "questions": selected_questions,
            "countries": selected_countries,
        }

    return win_rate_results


def plot_heatmaps(win_rate_results: dict):
    # Red = Direct wins, Green = MtM wins
    cmap = ListedColormap(["#d62728", "#2ca02c"])

    models = list(win_rate_results.keys())
    questions = win_rate_results[models[0]]["questions"]
    countries = win_rate_results[models[0]]["countries"]

    n_models = len(models)
    if n_models > 3:
        n_rows = 2
    else:
        n_rows = 1
    n_cols = math.ceil(n_models / n_rows)

    # Create figure
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(2 + 4 * n_cols, 2 + 4 * n_rows),
        sharex=True,
        sharey=True,
        squeeze=False,
    )

    # Iterate over all models
    for model_idx, model in enumerate(models):
        row_idx = model_idx // n_cols
        col_idx = model_idx % n_cols

        ax = axes[row_idx, col_idx]
        win_events = win_rate_results[model]["win_events"]
        model_win_rate = np.average(win_events)

        # Plot as heatmap
        ax.imshow(
            win_events,
            aspect="auto",
            cmap=cmap,
            vmin=0,
            vmax=1,
        )

        ax.set_title(rf"{model.split("/")[1]}, win rate = {100 * model_win_rate:.1f}\,\%")

        ax.set_xticks(np.arange(len(countries)))
        ax.set_xticklabels(countries, rotation=45, ha="right")

        ax.set_yticks(np.arange(len(questions)))
        ax.set_yticklabels(questions)

        # Grid lines between cells
        ax.set_xticks(np.arange(-0.5, len(countries), 1), minor=True)
        ax.set_yticks(np.arange(-0.5, len(questions), 1), minor=True)
        ax.grid(which="minor", color="white", linestyle="-", linewidth=0.8)

        # Hide minor tick marks
        ax.tick_params(which="minor", bottom=False, left=False)

        if col_idx == 0:
            ax.set_ylabel("Question")

        if row_idx == n_rows - 1:
            ax.set_xlabel("Country")

    # Hide unused axes
    for model_idx in range(n_models, n_rows * n_cols):
        row_idx = model_idx // n_cols
        col_idx = model_idx % n_cols
        axes[row_idx, col_idx].axis("off")

    fig.tight_layout()

    # Plot directory
    plot_base_dir = get_plot_dir()
    plot_dir = plot_base_dir / "micro_macro_aggregated"
    plot_dir.mkdir(parents=True, exist_ok=True)

    # Save plot
    plt.savefig(plot_dir / "GOQA_thresholded_heatmap.png", dpi=300, bbox_inches="tight")
    plt.savefig(plot_dir / "GOQA_thresholded_heatmap.pdf")

    plt.show()


if __name__ == "__main__":
    # Parameters
    experiment_name = "goqa_micro_macro_thresholded"

    # Questions: ["55", "136", "573", "839", "1108"]
    # Countries: ["Germany", "Japan", "France", "United States", "Australia", "Italy", "Spain"]

    # None means "all models"
    models_for_avg_err = None
    question_for_avg_err = ["55", "136", "573", "839", "1108"]
    countries_for_avg_err = ["Germany", "Japan"]

    models_win_rate = ["openai/gpt-5.4"]
    questions_win_rate = ["55", "136", "573", "839", "1108"]
    countries_win_rate = ["Germany", "Japan", "France", "United States", "Australia", "Italy", "Spain"]

    # Logs for individual experiments
    log_files = [
        "experiments_gpt54.yaml",
        "experiments_gpt4o_mini.yaml",
        "experiments_opus.yaml",
        "experiments_sonnet.yaml",
        "experiments_gemini_pro.yaml",
        # "experiments_gemini_flash.yaml",
        "experiments_grok.yaml",
    ]

    logs_more_countries = "experiments_new_countries.yaml"

    # Run main function
    plot_results, detailed_results, win_rate_results = main_processing_with_merging(
        experiment_name=experiment_name,
        log_files=log_files,
        logs_more_countries=logs_more_countries,
        models_for_avg_err=models_for_avg_err,
        question_for_avg_err=question_for_avg_err,
        countries_for_avg_err=countries_for_avg_err,
        models_win_rate=models_win_rate,
        questions_win_rate=questions_win_rate,
        countries_win_rate=countries_win_rate,
    )

    # Plot 1: average errors
    plot_average_error(plot_results=plot_results)

    # Plot 2: win rate heatmaps
    plot_heatmaps(win_rate_results=win_rate_results)
