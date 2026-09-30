"""Free-text answering for /api/chat, built on the same planner and executor as rag.py."""
import re

from sqlalchemy import text

from backend.data import load_frames
from backend.execute import execute
from backend.plan import plan
from backend.rag import load_text_tables
from backend.stretch import answer_text

# Free text has no return block, so the demo picks a text handler from the wording.
# Graded answers in rag.py route by the declared evidence tables instead.
TEXT_KINDS = [
    ("go_ahead", r"go[- ]ahead|game[- ]winn"),
    ("deficit", r"\b(deficit|comeback|overcame)\b"),
    ("injury", r"\b(why|miss|missed)\b"),
]
KEYS = {"game_details": "game_id", "game_recaps": "recap_id", "injury_notes": "note_id"}
_CTX = None


def get_ctx(cx):
    global _CTX
    if _CTX is None:
        recaps, notes = load_text_tables(cx)
        _CTX = {"frames": load_frames(cx), "recaps": recaps, "notes": notes}
    return _CTX


def evidence_docs(cx, evidence):
    out = []
    for e in evidence:
        if e["table"] == "player_box_scores":
            row = cx.execute(
                text("SELECT doc FROM player_box_scores WHERE game_id = :g AND person_id = :p"),
                {"g": e["game_id"], "p": e["person_id"]},
            ).first()
            out.append({"table": e["table"], "id": e["game_id"], "game_id": e["game_id"],
                        "person_id": e["person_id"], "doc": row[0] if row else ""})
        else:
            key = KEYS[e["table"]]
            row = cx.execute(text(f"SELECT doc FROM {e['table']} WHERE {key} = :i"), {"i": e["id"]}).first()
            out.append({"table": e["table"], "id": e["id"], "doc": row[0] if row else ""})
    return out


def render(facts):
    return ", ".join(f"{k.replace('_', ' ')}: {v}" for k, v in facts.items())


def answer_free_text(cx, question):
    ctx = get_ctx(cx)
    kind = next((k for k, pat in TEXT_KINDS if re.search(pat, question, re.I)), None)
    if kind:
        out = answer_text(cx, {"question": question}, kind, ctx["frames"], ctx["recaps"], ctx["notes"])
    else:
        spec = plan(cx, question)
        out = None if spec is None else execute(ctx["frames"], spec)
    if out is None:
        return {"answer": "The data cannot answer this question.", "evidence": []}
    facts, evidence = out
    return {"answer": render(facts), "evidence": evidence_docs(cx, evidence)}