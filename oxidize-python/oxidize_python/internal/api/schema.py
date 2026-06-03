"""OpenAI API request schema mirroring oxidize-golang/internal/api/schema.go."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class ResponseFormatType(StrEnum):
    TEXT = "text"
    JSON_OBJECT = "json_object"
    JSON_SCHEMA = "json_schema"


@dataclass
class ResponseFormat:
    type: ResponseFormatType = ResponseFormatType.TEXT
    schema: dict[str, Any] | None = None

    def output_text(self) -> str:
        if self.type == ResponseFormatType.JSON_OBJECT:
            return "{}"
        return ""


@dataclass
class ContentImageURL:
    url: str
    detail: str = ""


@dataclass
class ContentPart:
    type: str
    text: str = ""
    image_url: ContentImageURL | None = None


@dataclass
class MessageContent:
    text: str = ""
    parts: list[ContentPart] = field(default_factory=list)
    mode: str = "text"

    @staticmethod
    def from_text(text: str) -> MessageContent:
        return MessageContent(text=text, mode="text")

    @staticmethod
    def from_parts(parts: list[ContentPart]) -> MessageContent:
        return MessageContent(parts=list(parts), mode="parts")

    def text_value(self) -> str:
        if self.mode != "parts":
            return self.text
        return "".join(p.text for p in self.parts if p.type == "text")


@dataclass
class ChatMessage:
    role: str
    content: MessageContent
    images: list[str] = field(default_factory=list)


@dataclass
class ChatCompletionRequest:
    model: str = ""
    messages: list[ChatMessage] = field(default_factory=list)
    response_format: ResponseFormat | None = None
    guided_json: bytes | None = None
    json_schema: bytes | None = None
    guided_regex: str = ""
    guided_choice: list[str] = field(default_factory=list)
    stream: bool = False
    max_tokens: int | None = None
    max_completion_tokens: int | None = None
    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    min_p: float | None = None
    typical_p: float | None = None
    tail_free_z: float | None = None
    stop: list[str] | list[int] | None = None
    seed: int | None = None
    n: int | None = None
    best_of: int | None = None

    @classmethod
    def from_json(cls, raw: bytes | str) -> ChatCompletionRequest:
        data = json.loads(raw) if isinstance(raw, (bytes, str)) else raw
        messages: list[ChatMessage] = []
        for m in data.get("messages", []):
            content = m.get("content", "")
            if isinstance(content, str):
                mc = MessageContent.from_text(content)
            else:
                parts = [
                    ContentPart(
                        type=p.get("type", "text"),
                        text=p.get("text", ""),
                        image_url=ContentImageURL(**p["image_url"])
                        if "image_url" in p
                        else None,
                    )
                    for p in content
                ]
                mc = MessageContent.from_parts(parts)
            messages.append(ChatMessage(role=m.get("role", "user"), content=mc))
        return cls(model=data.get("model", ""), messages=messages, stream=data.get("stream", False))
