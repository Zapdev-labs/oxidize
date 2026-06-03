"""OpenAI API responses mirroring oxidize-golang/internal/api/responses.go."""

from __future__ import annotations

from dataclasses import dataclass, field

from oxidize_python.internal.api.schema import ChatMessage, MessageContent


@dataclass
class APIError:
    message: str
    type: str = "invalid_request_error"
    code: str | None = None


@dataclass
class ErrorResponse:
    error: APIError
    status_code: int = 400


@dataclass
class Usage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


@dataclass
class ModelData:
    id: str
    object: str = "model"
    owned_by: str = "oxidize"


@dataclass
class ModelsResponse:
    object: str = "list"
    data: list[ModelData] = field(default_factory=list)


@dataclass
class ChatDelta:
    role: str = ""
    content: str = ""


@dataclass
class ChatChoice:
    index: int = 0
    message: ChatMessage | None = None
    delta: ChatDelta | None = None
    finish_reason: str | None = None


@dataclass
class ChatCompletionResponse:
    id: str = "chatcmpl-placeholder"
    object: str = "chat.completion"
    created: int = 0
    model: str = ""
    choices: list[ChatChoice] = field(default_factory=list)
    usage: Usage = field(default_factory=Usage)


def build_chat_completion(model: str, content: str) -> ChatCompletionResponse:
    return ChatCompletionResponse(
        model=model,
        choices=[
            ChatChoice(
                index=0,
                message=ChatMessage(
                    role="assistant",
                    content=MessageContent.from_text(content),
                ),
                finish_reason="stop",
            )
        ],
    )


def build_chat_chunk(model: str, content: str, finished: bool) -> ChatCompletionResponse:
    choice = ChatChoice(index=0, delta=ChatDelta(content=content))
    if finished:
        choice.finish_reason = "stop"
    return ChatCompletionResponse(id="chatcmpl-chunk", object="chat.completion.chunk", model=model, choices=[choice])
