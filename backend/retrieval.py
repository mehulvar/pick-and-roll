"""Exact-scan retrieval over the embedded text tables, scoped to candidate games."""
import os

from sqlalchemy import text

from backend.config import EMBED_MODEL
from backend.utils import ollama_embed

QUERY_PREFIX = "search_query: "
TABLES = {"game_recaps": "recap_id", "injury_notes": "note_id"}
MARGIN = 0.02


def retrieve_from(cx, table, qvec, game_ids, k=2):
    key = TABLES[table]
    sql = (
        f"SELECT {key} AS id, game_id, 1 - (embedding <=> (:q)::vector) AS score "
        f"FROM {table} WHERE embedding IS NOT NULL AND game_id = ANY(CAST(:ids AS BIGINT[])) "
        f"ORDER BY embedding <=> (:q)::vector, {key} LIMIT :k"
    )
    params = {"q": qvec, "ids": list(game_ids), "k": k}
    return [dict(r) for r in cx.execute(text(sql), params).mappings().all()]


def pick_game(cx, question, game_ids, table):
    """Return the best matching candidate game_id, or None if the top two are too close."""
    qvec = ollama_embed(EMBED_MODEL, QUERY_PREFIX + question)
    rows = retrieve_from(cx, table, qvec, game_ids)
    if os.getenv("RETR_DEBUG"):
        print("retrieval:", [(r["game_id"], round(float(r["score"]), 4)) for r in rows])
    if not rows:
        return None
    if len(rows) > 1 and float(rows[0]["score"]) - float(rows[1]["score"]) < MARGIN:
        return None
    return int(rows[0]["game_id"])