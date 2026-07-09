from prompting.formatting_helper import make_example_json, format_template, \
    _generate_attribute_list_from_combination


def assemble_wvs_answer_distribution_prompt(
        question: str,
        answer_options: list[str],
        attributes: list[dict],
        country: str,
        template_filename: str = "WVS_answer_distribution_estimation_v2.txt",
) -> tuple[str, dict]:
    # Generate attributes list
    attribute_list = f"- country: {country}\n"
    for attribute in attributes:
        attribute_list += f"- {attribute["attribute_description"]}: {attribute["value_description"]}\n"

    # Group descriptor
    group_descriptor = "Answer option"

    # Generate answer option to label dict
    answer_option_to_label_dict = {
        option: f"{group_descriptor} {idx + 1}" for idx, option in enumerate(answer_options)
    }

    # Generate answer option list
    answer_option_list = "\n".join([f"- {label}: {option}" for option, label in answer_option_to_label_dict.items()])

    # Generate group names
    answer_option_keys = ", ".join([f'"{group_descriptor} {idx + 1}"' for idx in range(len(answer_options))])

    # Generate example output
    example_output = make_example_json(
        num_groups=len(answer_options),
        group_descriptor=group_descriptor,
        uniform_distribution=True
    )

    # Assemble prompt
    prompt = format_template(template_filename=template_filename,
                             attribute_list=attribute_list[:-1],
                             question=question,
                             answer_option_list=answer_option_list,
                             answer_option_keys=answer_option_keys,
                             example_output=example_output)

    return prompt, answer_option_to_label_dict


def assemble_tennis_prompt(
        attributes: list[dict],
        template_filename: str = "tennis_distribution_estimation_v2.txt",
) -> str:
    # Generate attributes list
    attribute_list = ""
    for attribute in attributes:
        attribute_list += f"- {attribute["attribute_description"]}: {attribute["value_description"]}\n"

    # Assemble prompt
    prompt = format_template(template_filename=template_filename,
                             attribute_list=attribute_list[:-1])

    return prompt


def assemble_fantasy_prompt(
        attributes: list[dict],
        template_filename: str = "fantasy_distribution_estimation_v2.txt",
) -> str:
    # Generate attributes list
    if len(attributes) == 0:
        attribute_list = "- no additional context available "
    else:
        attribute_list = ""
        for attribute in attributes:
            attribute_list += f"- {attribute["attribute_description"]}: {attribute["value_description"]}\n"

    # Assemble prompt
    prompt = format_template(template_filename=template_filename,
                             attribute_list=attribute_list[:-1])

    return prompt


def assemble_wvs_prior_distribution_prompt(
        country: str,
        combinations: list[list[dict]],
        template_filename: str = "WVS_prior_distribution_estimation_v2.txt",
) -> tuple[str, list[dict]]:
    # Generate all group descriptions
    group_wise_attribute_lists = []
    for idx, combination in enumerate(combinations):
        group_wise_attribute_lists.append(
            {
                "group_name": f"Group {idx + 1}",
                "attribute_list_string": _generate_attribute_list_from_combination(combination),
                "combination": combination,
            }
        )

    # Concatenate
    groups = ""
    for group in group_wise_attribute_lists:
        groups += group["group_name"] + "\n" + group["attribute_list_string"] + "\n\n"

    # Truncate last two \n
    groups = groups[:-2]

    # Generate group names
    group_names = ", ".join([f'"{group["group_name"]}"' for group in group_wise_attribute_lists])

    # Generate example output
    example_output = make_example_json(num_groups=len(combinations), uniform_distribution=True)

    # Assemble prompt
    prompt = format_template(template_filename=template_filename,
                             country=country,
                             groups=groups,
                             group_names=group_names,
                             example_output=example_output
                             )

    return prompt, group_wise_attribute_lists


if __name__ == "__main__":
    # Question and answer options
    question = ("For each of the following aspects, indicate how important it is in your life. Would you say it is very"
                " important, rather important, not very important or not important at all? - Family")
    answer_options = ["Very important", "Rather important", "Not very important", "Not at all important"]

    # Attributes
    attributes = [{"attribute_description": "sex", "value_description": "female"}]

    # Country
    country = "Canada"

    # Generate prompt
    prompt, answer_option_to_label_dict = assemble_wvs_answer_distribution_prompt(
        question=question,
        answer_options=answer_options,
        attributes=attributes,
        country=country,
    )
    print(f"Prompt: {prompt}")
    print(f"Answer option to label dict: {answer_option_to_label_dict}")

    # Generate prompt for prior elicitation
    combinations = [
        [{"attribute_description": "sex", "value_description": "female"}],
        [{"attribute_description": "sex", "value_description": "male"}],
    ]
    prompt, _ = assemble_wvs_prior_distribution_prompt(
        country=country,
        combinations=combinations,
    )
    print(f"Prompt: {prompt}")

    # Attributes
    attributes = [
        {
            "attribute_description": "court surface",
            "value_description": "clay"
        },
        {
            "attribute_description": "first set",
            "value_description": "Federer wins the first set"
        }
    ]

    # Prompt
    tennis_prompt = assemble_tennis_prompt(attributes=attributes)
    print("*" * 80)
    print(f"Prompt: {tennis_prompt}")
