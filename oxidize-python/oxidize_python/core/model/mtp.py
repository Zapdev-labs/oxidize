"""MTP generation mirroring oxidize-golang/core/model/mtp.go."""

from __future__ import annotations

from oxidize_python.core.ggufcore import gguf as ggufcore
from oxidize_python.core.model.generation import (
    ERR_GENERATION_FINISHED,
    GenerationConfig,
    GenerationError,
)
from oxidize_python.core.model.model import Model, Session, Token
from oxidize_python.core.model.sampling import sample


def has_mtp_weights(path: str) -> bool:
    try:
        mapped = ggufcore.load_mapped(path)
    except OSError:
        return False
    for tensor in mapped.parsed.tensor_infos:
        name = tensor.name.lower()
        if "nextn" in name or "mtp" in name:
            return True
    return False


class MtpGenerationStream:
    def __init__(self, model: Model, session: Session, config: GenerationConfig) -> None:
        self.model = model
        self.session = session
        self.config = config
        self.done = False
        self.prompt: list[Token] = []

    def seed(self, prompt: list[Token]) -> None:
        self.prompt = list(prompt)

    def next(self) -> tuple[Token, bool, GenerationError | None]:
        if self.done:
            return 0, True, ERR_GENERATION_FINISHED
        context_tokens = list(self.prompt)
        logits = self.model.forward(context_tokens, self.session)
        token = sample(logits, self.config.sampling, None)
        if token == self.config.stop_token:
            self.done = True
            return token, True, None
        self.prompt.append(token)
        if len(self.prompt) >= self.config.max_new_tokens:
            self.done = True
        return token, self.done, None
