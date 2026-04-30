from __future__ import annotations

from typing import Literal, Protocol, Sequence, TypedDict, overload


TensorOutput = Literal["list", "numpy", "torch"]


class _TensorLike(Protocol):
    def tolist(self) -> Sequence[int]: ...


class ChatCompletionMessage(TypedDict):
    role: Literal["assistant"]
    content: str


class ChatCompletionChoice(TypedDict):
    index: int
    message: ChatCompletionMessage
    finish_reason: Literal["length"]


class ChatCompletionResponse(TypedDict):
    id: str
    object: Literal["chat.completion"]
    choices: list[ChatCompletionChoice]

class CompletionChoice(TypedDict):
    index: int
    text: str
    finish_reason: Literal["length"]


class CompletionResponse(TypedDict):
    id: str
    object: Literal["text_completion"]
    choices: list[CompletionChoice]


class Llama:
    def __init__(
        self,
        model_path: str,
        vocab_size: int = 32000,
        context_size: int = 4096,
        layer_count: int = 32,
    ) -> None: ...

    def generate(self, prompt: str, max_tokens: int = 16) -> str: ...
    async def generate_async(self, prompt: str, max_tokens: int = 16) -> str: ...
    def create_chat_completion(
        self,
        messages: Sequence[str] | Sequence[dict[str, str]],
        max_tokens: int = 16,
    ) -> ChatCompletionResponse: ...
    async def create_chat_completion_async(
        self,
        messages: Sequence[str] | Sequence[dict[str, str]],
        max_tokens: int = 16,
    ) -> ChatCompletionResponse: ...
    def create_completion(
        self,
        prompt: str,
        max_tokens: int = 16,
    ) -> CompletionResponse: ...
    def __call__(
        self,
        prompt: str,
        max_tokens: int = 16,
    ) -> CompletionResponse: ...
    def embed(self, text: str) -> list[float]: ...
    def tokenize(self, text: str, add_bos: bool = True) -> list[int]: ...
    def detokenize(self, tokens: Sequence[int] | _TensorLike) -> bytes: ...
    @overload
    def generate_from_tokens(
        self,
        prompt_tokens: Sequence[int] | _TensorLike,
        max_tokens: int = 16,
        output_tensor: Literal["list"] | None = None,
    ) -> list[int]: ...
    @overload
    def generate_from_tokens(
        self,
        prompt_tokens: Sequence[int] | _TensorLike,
        max_tokens: int = 16,
        output_tensor: Literal["numpy"],
    ) -> object: ...
    @overload
    def generate_from_tokens(
        self,
        prompt_tokens: Sequence[int] | _TensorLike,
        max_tokens: int = 16,
        output_tensor: Literal["torch"],
    ) -> object: ...
    def generate_from_tokens(
        self,
        prompt_tokens: Sequence[int] | _TensorLike,
        max_tokens: int = 16,
        output_tensor: TensorOutput | None = None,
    ) -> object: ...
    @overload
    def embed_tensor(
        self,
        text: str,
        output_tensor: Literal["list"] | None = None,
    ) -> list[float]: ...
    @overload
    def embed_tensor(
        self,
        text: str,
        output_tensor: Literal["numpy"],
    ) -> object: ...
    @overload
    def embed_tensor(
        self,
        text: str,
        output_tensor: Literal["torch"],
    ) -> object: ...
    def embed_tensor(
        self,
        text: str,
        output_tensor: TensorOutput | None = None,
    ) -> object: ...


def workspace_health() -> str: ...
def version() -> str: ...


__version__: str
__all__: list[str]
