from typing import Literal
from prompting.formatting_helper import format_template, make_example_json


def assemble_goqa_distribution_prompt(
        question: str,
        options: list[str],
        country: str,
        prompting_scheme: Literal["sociodemographic", "persona", "unspecified"],
        template_filename: str,
) -> tuple[str, list[str]]:
    if prompting_scheme != "sociodemographic":
        raise ValueError(f"Prompting scheme '{prompting_scheme}' is not supported.")

    # Format options
    formatted_options, answer_option_keys = format_answer_options(options=options)
    example_output = make_example_json(num_groups=len(options), group_descriptor="Answer option", uniform_distribution=True)

    # Format prompt
    prompt = format_template(template_filename=template_filename,
                             country=country,
                             question=question,
                             answer_options=formatted_options,
                             answer_option_keys=", ".join(answer_option_keys),
                             example_output=example_output,
                             )

    return prompt, answer_option_keys


def assemble_goqa_probability_prompt(
        country: str,
        prompting_scheme: Literal["sociodemographic", "persona", "unspecified"],
        template_filename: str,
) -> str:
    if prompting_scheme != "sociodemographic":
        raise ValueError(f"Prompting scheme '{prompting_scheme}' is not supported.")

    # Format prompt
    prompt = format_template(template_filename=template_filename,
                             country=country,
                             )

    return prompt


def format_answer_options(options: list[str]) -> tuple[str, list[str]]:
    formatted_options = ""
    answer_option_keys = []
    for idx, option in enumerate(options):
        label = f"Answer option {idx + 1}"
        formatted_options += f"- {label}: {option}\n"
        answer_option_keys.append(f'"{label}"')

    return formatted_options[:-1], answer_option_keys


if __name__ == "__main__":
    # Parameters
    question = ("Which statement comes closer to your own views, even if neither is exactly right? Using overwhelming "
                "military force is the best way to defeat terrorism around the world, Relying too much on military "
                "force to defeat terrorism creates hatred that leads to more terrorism")
    options = [
        "Using overwhelming military force is the best way to defeat terrorism around the world",
        "If (survey country) had cooperated more with other countries, the number of coronavirus cases would have been "
        + "lower in this country",
        "Many of the problems facing our country can be solved by working with other countries",
        "Relying too much on military force to defeat terrorism creates hatred that leads to more terrorism",
        "No amount of cooperation between (survey country) and other countries would have reduced the number of "
        + "coronavirus cases in this country",
        "Few of the problems facing our country can be solved by working with other countries",
        "DK/Refused",
        "When dealing with major international issues, our country should take into account the interests of other "
        + "countries, even if it means making compromises with them",
        "When dealing with major international issues, our country should follow its own interests, even when other "
        + "countries strongly disagree"
    ]
    country = "Belgium"
    prompting_scheme = "sociodemographic"

    # Direct prompting
    direct_prompt, _ = assemble_goqa_distribution_prompt(
        question=question,
        options=options,
        country=country,
        prompting_scheme=prompting_scheme,
        template_filename="GOQA_direct_prompt_v1.txt",
    )

    # Micro to macro prompting
    mtm_prompt, _ = assemble_goqa_distribution_prompt(
        question=question,
        options=options,
        country=country,
        prompting_scheme=prompting_scheme,
        template_filename="GOQA_implicit_micro_macro.txt",
    )

    # MtM probability prompt
    mtm_probability = assemble_goqa_probability_prompt(
        country=country,
        prompting_scheme=prompting_scheme,
        template_filename="GOQA_Q55_micro_macro.txt",
    )

    print(direct_prompt)
    print("\n" + "*" * 80 + "\n")
    print(mtm_prompt)
    print("\n" + "*" * 80 + "\n")
    print(mtm_probability)
