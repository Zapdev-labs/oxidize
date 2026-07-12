from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType


def load_builder() -> ModuleType:
    path = Path(__file__).with_name("build_seed_dataset.py")
    spec = importlib.util.spec_from_file_location("build_seed_dataset", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_role_aliases_and_alternate_content_fields() -> None:
    # Given
    builder = load_builder()
    messages = [{"from": "human", "value": "hello"}, {"from": "gpt", "text": "world"}]

    # When
    text = builder.messages_to_text(messages)

    # Then
    assert text == (
        "<|im_start|>user\nhello<|im_end|>\n"
        "<|im_start|>assistant\nworld<|im_end|>\n"
    )


def test_invalid_and_empty_messages_are_ignored() -> None:
    # Given
    builder = load_builder()

    # When
    text = builder.messages_to_text(["invalid", {"role": "unknown", "content": "x"}, {}])

    # Then
    assert text == ""


def test_chat_delimiters_are_stripped_but_ordinary_text_is_preserved() -> None:
    # Given
    builder = load_builder()
    messages = [{"role": "user", "content": "もしも <|im_start|> nested <|im_end|>"}]

    # When
    text = builder.messages_to_text(messages)

    # Then
    assert "もしも" in text
    assert text.count("<|im_start|>") == 1
    assert text.count("<|im_end|>") == 1


def test_trajectory_action_and_observation_fields_normalize() -> None:
    # Given
    builder = load_builder()
    row = {"trajectory": [{"action": "inspect"}, {"observation": "done"}]}

    # When
    record = builder.normalize(row)

    # Then
    assert record is not None
    assert "inspect" in record["text"]
    assert "done" in record["text"]
