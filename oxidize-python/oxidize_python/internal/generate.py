from dataclasses import dataclass


@dataclass
class PlaceholderSpec:
    response_format: object | None = None
    guided_json: bytes | None = None
    json_schema: bytes | None = None
    guided_regex: str = ""
    guided_choice: list[str] | None = None


def placeholder_text(spec: PlaceholderSpec) -> str:
    if spec.guided_choice:
        return spec.guided_choice[0]
    if spec.guided_json or spec.json_schema:
        return "{}"
    if spec.guided_regex:
        return spec.guided_regex
    if spec.response_format is not None and hasattr(spec.response_format, "output_text"):
        return spec.response_format.output_text()
    return ""


def cli_text(prompt: str) -> str:
    return f"oxidize-cli: {prompt}"


def cli_transcript(prompt: str) -> str:
    return (
        "generation progress: 1/2 tokens\n"
        "generation progress: 2/2 tokens\n"
        f"{cli_text(prompt)}\n"
        "generation stats: tokens=2 speed=4.00 tok/s\n"
    )
