"""Pydantic models for retrieved passages and grounded answers."""

from __future__ import annotations

from pydantic import BaseModel, Field


class RetrievedPassage(BaseModel):
    section_id: str
    title: str
    url: str = ""
    heading: str = ""
    text: str
    score: float
    strategy: str = ""

    def short_label(self) -> str:
        return self.section_id if not self.heading else f"{self.section_id} — {self.heading}"


class Citation(BaseModel):
    section_id: str
    title: str
    url: str = ""


class Answer(BaseModel):
    question: str
    answer: str
    citations: list[Citation] = Field(default_factory=list)
    passages: list[RetrievedPassage] = Field(default_factory=list)
    refused: bool = Field(
        default=False,
        description="True when retrieval was too weak to ground an answer and the system declined.",
    )
