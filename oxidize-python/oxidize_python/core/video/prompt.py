"""Multimodal video prompt mirroring oxidize-golang/core/video/prompt.go."""

from __future__ import annotations

from dataclasses import dataclass, field

from oxidize_python.core.video.errors import VideoError


@dataclass
class VideoSegment:
    """Per-frame embeddings flattened row-major."""

    embeddings: list[float]
    num_frames: int
    llm_hidden_size: int


@dataclass
class PromptSegment:
    """One block of a multimodal video prompt."""

    text_tokens: list[int] = field(default_factory=list)
    video: VideoSegment | None = None


class VideoPrompt:
    """Builds a flattened embedding sequence for video + text inputs."""

    def __init__(self) -> None:
        self.segments: list[PromptSegment] = []
        self.video_start_embedding: list[float] = []
        self.video_end_embedding: list[float] = []

    def add_text(self, tokens: list[int]) -> None:
        """Append a text token block."""
        self.segments.append(PromptSegment(text_tokens=list(tokens)))

    def add_video(self, embeddings: list[float], num_frames: int, hidden: int) -> None:
        """Append a video embedding block."""
        self.segments.append(
            PromptSegment(
                video=VideoSegment(
                    embeddings=list(embeddings),
                    num_frames=num_frames,
                    llm_hidden_size=hidden,
                )
            )
        )

    def build_sequence(
        self,
        table: list[float],
        vocab_size: int,
        hidden_size: int,
    ) -> list[float]:
        """Flatten segments using the token embedding table for text rows."""
        llm_hidden = self._infer_hidden_size(hidden_size)
        total_rows = self._count_rows(llm_hidden)
        out = [0.0] * (total_rows * llm_hidden)
        cursor = 0

        def write_row(row: list[float]) -> None:
            nonlocal cursor
            if len(row) != llm_hidden:
                raise VideoError(f"row width {len(row)} != {llm_hidden}")
            out[cursor : cursor + llm_hidden] = row
            cursor += llm_hidden

        for seg in self.segments:
            if seg.video is not None:
                if len(self.video_start_embedding) == llm_hidden:
                    write_row(self.video_start_embedding)
                v = seg.video
                if v.num_frames * v.llm_hidden_size != len(v.embeddings):
                    raise VideoError("video embedding length mismatch")
                for f in range(v.num_frames):
                    start = f * v.llm_hidden_size
                    write_row(v.embeddings[start : start + v.llm_hidden_size])
                if len(self.video_end_embedding) == llm_hidden:
                    write_row(self.video_end_embedding)
                continue
            for tok in seg.text_tokens:
                if tok >= vocab_size:
                    raise VideoError(f"token {tok} >= vocab {vocab_size}")
                start = tok * hidden_size
                if start + hidden_size > len(table):
                    raise VideoError("embedding table too small")
                row = table[start : start + hidden_size]
                if hidden_size == llm_hidden:
                    write_row(row)
                    continue
                padded = [0.0] * llm_hidden
                padded[: len(row)] = row[:llm_hidden]
                write_row(padded)
        return out

    def _infer_hidden_size(self, fallback: int) -> int:
        for seg in self.segments:
            if seg.video is not None and seg.video.llm_hidden_size > 0:
                return seg.video.llm_hidden_size
        if fallback <= 0:
            raise VideoError("cannot infer hidden size")
        return fallback

    def _count_rows(self, llm_hidden: int) -> int:
        rows = 0
        for seg in self.segments:
            if seg.video is not None:
                extra = 0
                if len(self.video_start_embedding) == llm_hidden:
                    extra += 1
                if len(self.video_end_embedding) == llm_hidden:
                    extra += 1
                rows += extra + seg.video.num_frames
                continue
            rows += len(seg.text_tokens)
        if rows == 0:
            raise VideoError("empty prompt")
        return rows
