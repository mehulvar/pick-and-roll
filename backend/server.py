"""FastAPI chat endpoint over the same planner and executor used by backend/rag.py.

usage:
    uvicorn backend.server:app --reload
"""
import sqlalchemy as sa
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from backend.chat import answer_free_text
from backend.config import DB_DSN

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:4200"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
eng = sa.create_engine(DB_DSN)


class Q(BaseModel):
    question: str


@app.post("/api/chat")
def answer(q: Q) -> dict:
    """Answer a free-text question with the planner and executor from rag.py.

    :param q: request body with a single "question" field
    :return: {"answer": str, "evidence": [{"table", "id", "doc"}]}
    """
    with eng.begin() as cx:
        return answer_free_text(cx, q.question)