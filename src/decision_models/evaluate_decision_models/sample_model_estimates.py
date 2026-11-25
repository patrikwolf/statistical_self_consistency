from decision_models.utility.prompt_decision_models import multi_prompting
from decision_models.utility.read_questions import read_filtered_questions
from file_logging.read_and_write_json import save_as_json
from utility.time_helper import get_experiment_timestamp


def sample_estimates_on_question(
        model: str,
        question_set: dict,
        num_runs: int,
        verbose: bool,
) -> dict:
    # Extract nodes
    nodes = {
        "root": question_set["root"],
        "left_child": question_set["children"][0],
        "right_child": question_set["children"][1],
        "prior": question_set["split"],
    }

    results = {}
    for name, node in nodes.items():
        print(f"\nSampling node '{name}'...")
        estimates, average = multi_prompting(
            model=model,
            context=node["context"],
            question=node["question"],
            options=node["options"],
            num_runs=num_runs,
            verbose=verbose,
        )

        results[name] = {
            "estimates": estimates,
            "average": average,
            "split_option": node.get("split_option", None),
        }

    # Print
    print(f"\nAverage estimates over {num_runs} runs per component:")
    for name, result in results.items():
        print(f"{name}: {result['average']}")

    return results


if __name__ == "__main__":
    # Model selection
    # model = "typesafe/jev-1.13"
    # model = "typesafe/jev-1.14"
    # model = "liquid/d1"
    # model = "liquid/d1-ultra"
    # model = "openai/gpt-6-luna-decisions"
    # model = "microsoft/microsoft-decision-1"
    # model = "nace-ai/drex-v1.5"
    # model = "nace-ai/drex-v2.0"
    # model = "inception/mercury-decide"
    model = "convaiinnovations/laya"

    # Number of runs (to average over)
    num_runs = 2

    # Experiment folder
    date, timestamp, experiment_folder = get_experiment_timestamp(experiment_name="decision_model_evaluation")

    # Read filtered questions
    questions = read_filtered_questions()
    questions = questions[:3]  # Limit to first 3 questions for testing

    # Evaluate the model on each question set
    samples = []
    for idx, question_set in enumerate(questions):
        print(f"\nSampling question {idx + 1}/{len(questions)}...")

        # Get model estimates
        estimates = sample_estimates_on_question(
            model=model,
            question_set=question_set,
            num_runs=num_runs,
            verbose=True,
        )

        # Combine the results with the question set for later analysis
        question_set["model_estimates"] = estimates
        samples.append(question_set)

    # Collect data
    results = {
        "model": model,
        "num_runs": num_runs,
        "samples": samples,
    }

    # Save results
    filename = save_as_json(
        data=results,
        experiment=experiment_folder,
        filename="sampling_results.json",
        timestamp=False,
        cluster=False,
    )

    print("\n" + "*" * 80)
    print(f"Model: {model}")
    print(f"Sampling results saved to {filename}")
    print("*" * 80)
