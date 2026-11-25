from decision_models.utility.prompt_decision_models import prompt_decision_model


def decision_model_multi_prompting(
        model_name: str,
        question: str,
        context: str,
        options: dict[str, str],
        num_of_samples: int,
) -> dict:
    # Decision model prediction
    log_direct = {}
    for idx in range(num_of_samples):
        print(f"Sample {idx + 1}/{num_of_samples}")

        # Prompt mode
        distribution, output = prompt_decision_model(
            model=model_name,
            context=context,
            question=question,
            options=options,
            verbose=False,
        )

        # Store log
        log_direct[f"run_{idx}"] = {
            "prediction": distribution,
            "output": output,
            "question": question,
            "context": context,
            "options": options,
        }

    return log_direct
