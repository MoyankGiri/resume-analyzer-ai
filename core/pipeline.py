from __future__ import annotations
import os
from dotenv import load_dotenv
from typing import Dict, Any, Literal, List

from langgraph.graph import StateGraph, START, END
from langchain_core.messages import ToolMessage

from langchain_google_genai import ChatGoogleGenerativeAI

from core.processing import read_resume_file, clean_resume_text
from core.scoring import ResumeAnalysisState, ResumeScore, AnalysisDecision
from core.knowledge_base import setup_resume_knowledge_base

# --------- Models & Config ---------
load_dotenv()
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
# GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    # Fail early with a clear message in server logs (UI will show error)
    print("[WARN] GOOGLE_API_KEY not set. Please configure it in Environment Variables.")

MODEL_NAME = os.environ.get("GENAI_MODEL", "gemini-2.0-flash-lite")
TEMPERATURE = float(os.environ.get("GENAI_TEMPERATURE", "0"))
# MODEL_NAME = os.getenv("GENAI_MODEL", "gemini-2.0-flash-lite")
# TEMPERATURE = float(os.getenv("GENAI_TEMPERATURE", "0"))

chat_model = ChatGoogleGenerativeAI(model=MODEL_NAME, temperature=TEMPERATURE)
_resume_knowledge_tool = None
def get_resume_knowledge_tool():
    global _resume_knowledge_tool
    if _resume_knowledge_tool is None:
        _resume_knowledge_tool = setup_resume_knowledge_base()
    return _resume_knowledge_tool

def decide_analysis_type(state: ResumeAnalysisState):
    question = state["messages"][-1].content
    decision_prompt = f"""
    Analyze this user request: "{question}"
    Decide if this requires:
    - "score_and_compare": user has a resume or wants detailed scoring/feedback
    - "general_advice": user wants general resume tips or strategy
    """
    decision_model = chat_model.with_structured_output(AnalysisDecision)
    response = decision_model.invoke([{"role": "user", "content": decision_prompt}])
    return {"analysis_type": response.decision}


def extract_resume_text(state: ResumeAnalysisState):
    question = state["messages"][-1].content.strip()
    # Heuristic: if user passed a file path or pasted long resume-like text
    if question.endswith((".pdf", ".txt")) and os.path.exists(question):
        try:
            return {"resume_text": read_resume_file(question)}
        except Exception as e:
            return {"messages": state["messages"] + [{"role": "assistant", "content": f"Error reading file: {e}"}]}

    if len(question) > 200 and any(k in question.lower() for k in ["experience", "education", "skills", "work", "university", "project"]):
        return {"resume_text": clean_resume_text(question)}

    # Ask for proper input if we cannot detect resume text
    return {"messages": state["messages"] + [{"role": "assistant", "content": (
        "Please provide your resume for analysis as either (1) an uploaded PDF/TXT in the UI, or (2) paste the full text." )}]}


def score_resume(state: ResumeAnalysisState):
    resume_text = state.get("resume_text", "")
    if not resume_text:
        # No-op if text missing (UI will show the assistant message from previous node)
        return {}

    scoring_prompt = f"""
    Score this resume comprehensively on these criteria (0-25 each):\n
    CONTENT, FORMAT, KEYWORDS, IMPACT. Provide an overall score (0-100) and concise feedback.\n
    RESUME:\n{resume_text}
    """
    scoring_model = chat_model.with_structured_output(ResumeScore)
    r = scoring_model.invoke([{"role": "user", "content": scoring_prompt}])
    return {"initial_score": {
        "overall_score": r.overall_score,
        "content_score": r.content_score,
        "format_score": r.format_score,
        "keyword_score": r.keyword_score,
        "impact_score": r.impact_score,
        "feedback": r.feedback,
    }}


def retrieve_best_practices(state: ResumeAnalysisState):
    resume_knowledge_tool = get_resume_knowledge_tool()
    scores = state.get("initial_score", {})
    resume_text = state.get("resume_text", "")

    # Identify weak areas
    improvement_areas: List[str] = []
    if scores.get("content_score", 25) < 22:
        improvement_areas.append("content professional summary career progression")
    if scores.get("format_score", 25) < 22:
        improvement_areas.append("format structure ATS layout")
    if scores.get("keyword_score", 25) < 22:
        improvement_areas.append("industry keywords technical skills ATS")
    if scores.get("impact_score", 25) < 22:
        improvement_areas.append("quantified achievements action verbs measurable results")

    # Simple industry detection for AI/DS/Backend, else general
    resume_lower = resume_text.lower()
    detected = "general"
    for ind, keys in {
        "ai_ml": ["pytorch", "tensorflow", "hugging face", "llm", "nlp", "computer vision"],
        "data_science": ["pandas", "numpy", "statistics", "a/b", "bigquery", "tableau"],
        "backend": ["fastapi", "django", "spring", "microservices", "postgres", "redis"],
    }.items():
        if sum(1 for k in keys if k in resume_lower) >= 2:
            detected = ind
            break

    queries: List[str] = []
    if improvement_areas:
        queries.append(f"{detected} {' '.join(improvement_areas[:2])}")
    queries.append("resume best practices examples high scoring")

    all_chunks: List[str] = []
    for q in queries:
        try:
            chunk = resume_knowledge_tool.invoke({"query": q})
            all_chunks.append(f"=== QUERY: {q} ===\n{chunk}\n")
        except Exception:
            continue

    tool_message = ToolMessage(
        content="\n".join(all_chunks) if all_chunks else "",
        tool_call_id="best_practices",
        additional_kwargs={"detected_industry": detected, "improvement_focus": improvement_areas},
    )
    return {"messages": state["messages"] + [tool_message]}


def generate_improvement_suggestions(state: ResumeAnalysisState):
    scores = state.get("initial_score", {})
    resume_text = state.get("resume_text", "")
    tool_msgs = [m for m in state["messages"] if isinstance(m, ToolMessage)]
    rag_knowledge = tool_msgs[-1].content if tool_msgs else ""
    detected = tool_msgs[-1].additional_kwargs.get("detected_industry", "general") if tool_msgs else "general"

    overall = scores.get("overall_score", 0)
    if overall >= 85:
        level = "EXCELLENT"
        msg = "Your resume is strong—let's push it to perfection."
    elif overall >= 70:
        level = "GOOD"
        msg = "Solid resume with room for optimization."
    elif overall >= 55:
        level = "AVERAGE"
        msg = "Meaningful improvements can lift your score."
    else:
        level = "NEEDS WORK"
        msg = "Significant improvements required—here's the plan."

    prompt = f"""
    Using the RAG knowledge and best practices below, generate a detailed improvement plan with: 
    - 3–5 top priority changes (with expected point gains),
    - before/after bullet rewrites using strong action verbs and metrics,
    - missing industry-specific keywords for {detected},
    - formatting/layout refinements for ATS friendliness,
    - and a projected improved score.

    CURRENT SCORE: {overall}/100 ({level})\n
    RAG KNOWLEDGE:\n{rag_knowledge}\n
    RESUME:\n{resume_text}
    """

    response = chat_model.invoke([{"role": "user", "content": prompt}], max_token=1024)

    suggestions_list = [
        "Priority improvements",
        "Bullet rewrites",
        "Industry keywords",
        "Format refinements",
        "Projected score",
    ]

    return {
        "messages": state["messages"] + [response],
        "improvement_suggestions": suggestions_list,
    }


def generate_general_resume_advice(state: ResumeAnalysisState):
    q = state["messages"][-1].content
    advice_prompt = f"""
    Provide actionable resume advice for: "{q}"
    Cover structure, ATS tips, writing measurable bullets, and common mistakes.
    """
    response = chat_model.invoke([{"role": "user", "content": advice_prompt}])
    return {"messages": state["messages"] + [response]}


# --------- Graph Wiring ---------
workflow = StateGraph(ResumeAnalysisState)
workflow.add_node("decide_analysis_type", decide_analysis_type)
workflow.add_node("extract_resume_text", extract_resume_text)
workflow.add_node("score_resume", score_resume)
workflow.add_node("retrieve_best_practices", retrieve_best_practices)
workflow.add_node("generate_improvement_suggestions", generate_improvement_suggestions)
workflow.add_node("generate_general_resume_advice", generate_general_resume_advice)

workflow.add_edge(START, "decide_analysis_type")


def _route(state: ResumeAnalysisState) -> Literal["extract_resume_text", "generate_general_resume_advice"]:
    return "extract_resume_text" if state.get("analysis_type") == "score_and_compare" else "generate_general_resume_advice"

workflow.add_conditional_edges(
    "decide_analysis_type",
    _route,
    {
        "extract_resume_text": "extract_resume_text",
        "generate_general_resume_advice": "generate_general_resume_advice",
    },
)

workflow.add_edge("extract_resume_text", "score_resume")
workflow.add_edge("score_resume", "retrieve_best_practices")
workflow.add_edge("retrieve_best_practices", "generate_improvement_suggestions")
workflow.add_edge("generate_improvement_suggestions", END)
workflow.add_edge("generate_general_resume_advice", END)

resume_checker = workflow.compile()


# --------- Public API ---------

def analyze_resume(input_text: str, verbose: bool = False) -> Dict[str, Any]:
    """Run the graph and collate results into a dict for the UI."""
    try:
        final_result: Dict[str, Any] = {"analysis": None, "scores": None, "suggestions": []}
        for chunk in resume_checker.stream({"messages": [{"role": "user", "content": input_text}]}):
            for _, update in chunk.items():
                if verbose:
                    print({"node": _, "keys": list(update.keys())})
                if "initial_score" in update:
                    final_result["scores"] = update["initial_score"]
                if "improvement_suggestions" in update:
                    final_result["suggestions"] = update["improvement_suggestions"]
                if "messages" in update and update["messages"]:
                    final_result["analysis"] = update["messages"][-1].content
        return final_result
    except Exception as e:
        return {"error": f"Pipeline error: {e}"}


def analyze_resume_from_file(file_path: str, verbose: bool = False) -> Dict[str, Any]:
    return analyze_resume(file_path, verbose)