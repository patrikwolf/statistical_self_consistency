REPLACEMENTS = {
    "liquid/d1": "Liquid D1",
    "liquid/d1-ultra": "Liquid D1 Ultra",
    "typesafe/jev-1.13": "Jev 1.13",
    "typesafe/jev-1.14": "Jev 1.14",
    "openai/gpt-6-luna-decisions": "GPT-6 Luna Decisions",
    "microsoft/microsoft-decision-1": "Microsoft Decision 1",
    "nace-ai/drex-v1.5": "Drex 1.5",
    "nace-ai/drex-v2.0": "Drex 2.0",
    "inception/mercury-decide": "Mercury Decide",
    "convaiinnovations/laya": "Laya",
    "openai/gpt-4o-mini": "GPT-4o mini",
    "openai/gpt-6-luna": "GPT-6 Luna",
    "inception/mercury-2.5": "Mercury 2.5",
    "inception/mercury-3": "Mercury 3",
}


def replace_model_name(model_name: str) -> str:
    return REPLACEMENTS.get(model_name, model_name)


if __name__ == "__main__":
    print(replace_model_name("liquid/d1"))
