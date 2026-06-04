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


def api_error(
    message: str,
    *,
    type_: str = "invalid_request_error",
    code: str | None = None,
) -> APIError:
    return APIError(message=message, type=type_, code=code)


def model_not_found(model: str) -> ErrorResponse:
    return ErrorResponse(
        error=api_error(f"The model `{model}` does not exist", type_="invalid_request_error"),
        status_code=404,
    )


def malformed_json() -> ErrorResponse:
    return ErrorResponse(
        error=api_error("malformed JSON body"),
        status_code=400,
    )


def validate_candidate_count(n: int | None, best_of: int | None) -> ErrorResponse | None:
    if n is not None and n < 1:
        return ErrorResponse(error=api_error("n must be at least 1"), status_code=400)
    if best_of is not None and best_of < 1:
        return ErrorResponse(error=api_error("best_of must be at least 1"), status_code=400)
    return None


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
    return ChatCompletionResponse(
        id="chatcmpl-chunk",
        object="chat.completion.chunk",
        model=model,
        choices=[choice],
    )


@dataclass
class TextChoice:
    index: int = 0
    text: str = ""
    finish_reason: str | None = None


@dataclass
class TextCompletionResponse:
    id: str = "cmpl-placeholder"
    object: str = "text_completion"
    created: int = 0
    model: str = ""
    choices: list[TextChoice] = field(default_factory=list)
    usage: Usage = field(default_factory=Usage)


def build_models_response(*model_ids: str) -> ModelsResponse:
    return ModelsResponse(
        data=[ModelData(id=mid) for mid in model_ids],
    )


def build_text_completion(model: str, text: str) -> TextCompletionResponse:
    return TextCompletionResponse(
        model=model,
        choices=[TextChoice(index=0, text=text, finish_reason="stop")],
    )


def build_text_chunk(model: str, text: str, finished: bool) -> TextCompletionResponse:
    choice = TextChoice(index=0, text=text)
    if finished:
        choice.finish_reason = "stop"
    return TextCompletionResponse(
        id="cmpl-chunk",
        object="text_completion.chunk",
        model=model,
        choices=[choice],
    )


@dataclass
class EmbeddingData:
    object: str = "embedding"
    embedding: list[float] = field(default_factory=list)
    index: int = 0


@dataclass
class EmbeddingsResponse:
    object: str = "list"
    data: list[EmbeddingData] = field(default_factory=list)
    model: str = ""


def build_embeddings_response(model: str, dim: int = 8) -> EmbeddingsResponse:
    return EmbeddingsResponse(
        model=model,
        data=[EmbeddingData(embedding=[0.0] * dim)],
    )


def _message_to_dict(msg: ChatMessage) -> dict:
    return {"role": msg.role, "content": msg.content.text_value()}


def chat_response_to_dict(resp: ChatCompletionResponse) -> dict:
    choices = []
    for c in resp.choices:
        entry: dict = {"index": c.index}
        if c.message is not None:
            entry["message"] = _message_to_dict(c.message)
        if c.delta is not None:
            entry["delta"] = {"content": c.delta.content, "role": c.delta.role or None}
            entry["delta"] = {k: v for k, v in entry["delta"].items() if v}
        if c.finish_reason:
            entry["finish_reason"] = c.finish_reason
        choices.append(entry)
    return {
        "id": resp.id,
        "object": resp.object,
        "created": resp.created,
        "model": resp.model,
        "choices": choices,
        "usage": {
            "prompt_tokens": resp.usage.prompt_tokens,
            "completion_tokens": resp.usage.completion_tokens,
            "total_tokens": resp.usage.total_tokens,
        },
    }


def text_response_to_dict(resp: TextCompletionResponse) -> dict:
    return {
        "id": resp.id,
        "object": resp.object,
        "created": resp.created,
        "model": resp.model,
        "choices": [
            {"index": c.index, "text": c.text, "finish_reason": c.finish_reason}
            for c in resp.choices
        ],
        "usage": {
            "prompt_tokens": resp.usage.prompt_tokens,
            "completion_tokens": resp.usage.completion_tokens,
            "total_tokens": resp.usage.total_tokens,
        },
    }


def error_response_to_dict(resp: ErrorResponse) -> dict:
    out: dict = {"error": {"message": resp.error.message, "type": resp.error.type}}
    if resp.error.code:
        out["error"]["code"] = resp.error.code
    return out
