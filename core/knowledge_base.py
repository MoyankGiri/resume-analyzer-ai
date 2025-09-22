from __future__ import annotations
import os
import re
from typing import List
from bs4 import BeautifulSoup
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.vectorstores import InMemoryVectorStore
from langchain.tools.retriever import create_retriever_tool
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv

import asyncio
import sys

# Ensure an event loop exists for gRPC async clients
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

try:
    asyncio.get_running_loop()
except RuntimeError:
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

# Embedding model (requires GOOGLE_API_KEY in env)
load_dotenv()
_embeddings_model = os.environ.get("EMBEDDINGS_MODEL", "models/embedding-001")
topk = os.environ.get("EMBEDDINGS_TOPK", "5")
# _embeddings_model = os.getenv("EMBEDDINGS_MODEL", "models/embedding-001")


def _clean_web_content(text: str) -> str:
    text = BeautifulSoup(text, "html.parser").get_text()
    text = re.sub(r"(Subscribe|Newsletter|Cookie Policy|Privacy Policy).*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"Advertisement.*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\n\s*\n+", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def _best_practices_blob() -> str:
    return (
        """
        RESUME BEST PRACTICES:\n\n
        • Keep to 1 page (early career), 2 max (senior).\n
        • Use strong action verbs and quantify impact.\n
        • Sections: Header, Summary, Experience, Education, Skills, Projects/Certs.\n
        • Mirror job description keywords for ATS.\n
        • Ensure consistent formatting and zero typos.\n
        • AI/ML keywords: PyTorch, TensorFlow, Hugging Face, XGBoost, RAG, LLM fine-tuning, MLflow, W&B, MLOps.\n        """
    ).strip()


def setup_resume_knowledge_base():
    """Build an in-memory retriever tool with fallback content if web fails."""
    knowledge_urls: List[str] = [
        "https://www.indeed.com/career-advice/resumes-cover-letters/resume-examples",
        "https://resumegenius.com/resume-samples",
        "https://www.monster.com/career-advice/article/good-resume-examples",
        "https://www.glassdoor.com/blog/guide/how-to-write-a-resume/",
    ]

    docs = [Document(page_content=_best_practices_blob(), metadata={"source": "best_practices"})]

    loaded_any = False
    for url in knowledge_urls:
        try:
            web_docs = WebBaseLoader(url).load()
            for d in web_docs:
                d.page_content = _clean_web_content(d.page_content)
            docs.extend(web_docs)
            loaded_any = True
        except Exception:
            # Some sites may block scrapers in PaaS environments; silently continue
            continue

    # Split
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(chunk_size=800, chunk_overlap=100)
    splits = splitter.split_documents(docs)

    # Vector store
    embeddings = GoogleGenerativeAIEmbeddings(model=_embeddings_model)
    vstore = InMemoryVectorStore.from_documents(documents=splits, embedding=embeddings)
    retriever = vstore.as_retriever(search_kwargs={"k": topk})

    # Turn into a tool usable by LangChain/LangGraph
    return create_retriever_tool(
        retriever,
        "retrieve_resume_knowledge",
        "Search resume best practices and strong real-world examples.",
    )