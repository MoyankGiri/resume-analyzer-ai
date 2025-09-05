from __future__ import annotations
from typing import Literal, Dict, Any, List
from pydantic import BaseModel, Field
from langgraph.graph import MessagesState


class ResumeAnalysisState(MessagesState):
    """Graph state for the resume analyzer."""
    analysis_type: str = ""
    resume_text: str = ""
    initial_score: Dict[str, Any] = {}
    improvement_suggestions: List[str] = []


class ResumeScore(BaseModel):
    overall_score: int = Field(ge=0, le=100, description="Overall score out of 100")
    content_score: int = Field(ge=0, le=25, description="Content quality score out of 25")
    format_score: int = Field(ge=0, le=25, description="Format and structure score out of 25")
    keyword_score: int = Field(ge=0, le=25, description="Industry keywords score out of 25")
    impact_score: int = Field(ge=0, le=25, description="Impact and achievements score out of 25")
    feedback: str = Field(description="Brief feedback on strengths and weaknesses")


class AnalysisDecision(BaseModel):
    decision: Literal["score_and_compare", "general_advice"] = Field(
        description="Whether to do detailed analysis or give general advice"
    )