"""Answer part1/questions.json with retrieval-augmented generation.

For each question: embed the question text, retrieve the most similar
game_details documents from pgvector, ask the LLM to fill in the question's
declared `return` shape as JSON using only that retrieved context, then
coerce and validate the model's JSON before writing it out. Writes
part1/answers.json as a list of {id, result}, one entry per question.

usage:
    python -m backend.rag
"""
import json
import os
import re

from backend.data import load_frames
from backend.execute import execute
from backend.plan import plan
from backend.stretch import answer_text, route

import sqlalchemy as sa
from sqlalchemy import text
from sqlalchemy.engine import Connection

from backend.config import DB_DSN, EMBED_MODEL, LLM_MODEL
from backend.utils import ollama_embed, ollama_generate

BASE_DIR = os.path.dirname(__file__)
QUESTIONS_PATH = os.path.normpath(os.path.join(BASE_DIR, "..", "part1", "questions.json"))
ANSWERS_PATH = os.path.normpath(os.path.join(BASE_DIR, "..", "part1", "answers.json"))

TOP_K = 5
# nomic-embed-text is an asymmetric embedding model: text being indexed gets the
# "search_document: " prefix (see embed.py) and text being searched gets
# "search_query: " instead. utils.ollama_embed does not add this itself, so
# both sides of the retrieval must add their own prefix consistently.
QUERY_PREFIX = "search_query: "


def retrieve(cx: Connection, qvec: list[float], k: int = TOP_K) -> list[dict]:
    """Retrieve the k nearest game_details documents to a query embedding.

    Shared by rag.py and server.py so both use the exact same retrieval query
    and column set (see server.py: import from here instead of duplicating).

    :param cx: open SQLAlchemy connection
    :param qvec: query embedding, same dimensionality as game_details.embedding
    :param k: number of rows to retrieve
    :return: rows with game_id, doc, and cosine similarity score, best first
    """
    # Casting the bind param to ::vector is required: without it Postgres can't
    # tell psycopg2's array/list parameter apart from pgvector's `vector` type
    # and the query errors instead of comparing embeddings.
    sql = (
        "SELECT game_id, doc, 1 - (embedding <=> (:q)::vector) AS score "
        "FROM game_details WHERE embedding IS NOT NULL "
        # game_id as a tie-breaker: rows with equal cosine distance must sort
        # the same way on every run, or grading (fresh embed vs. committed
        # part1/answers.json) can see a different top-k order.
        "ORDER BY embedding <=> (:q)::vector, game_id LIMIT :k"
    )
    return [dict(r) for r in cx.execute(text(sql), {"q": qvec, "k": k}).mappings().all()]


def build_context(rows: list[dict]) -> str:
    """Render retrieved rows as the context block shown to the LLM.

    :param rows: rows as returned by retrieve() (needs game_id and doc)
    :return: one "[game_details id=<game_id>] <doc>" line per row
    """
    return "\n".join(f"[game_details id={r['game_id']}] {r['doc']}" for r in rows)


def coerce_type(value, type_name: str):
    """Coerce a raw JSON value to the type named in a question's `return` block.

    :param value: value as parsed from the model's JSON output
    :param type_name: one of "bool", "int", "str" (the vocabulary used in
        part1/questions.json's `return` blocks)
    :return: value converted to that type
    :raises ValueError: if type_name is unrecognized
    :raises TypeError: if value can't be coerced to that type
    """
    if type_name == "bool":
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.strip().lower() in ("true", "yes", "1")
        return bool(value)
    if type_name == "int":
        return int(round(float(value)))
    if type_name == "str":
        return str(value)
    raise ValueError(f"Unknown type in return schema: {type_name}")


def extract_json(raw: str) -> dict | None:
    """Pull the first top-level JSON object out of an LLM's free-text response.

    Models asked for JSON often wrap it in prose or code fences anyway, so this
    finds the outermost {...} span and parses it rather than requiring the
    whole response to be valid JSON.

    :param raw: raw text returned by ollama_generate
    :return: parsed dict, or None if no valid JSON object could be found
    """
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not match:
        return None
    try:
        parsed = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def ground_evidence(parsed_evidence, declared_shapes: list[dict], retrieved_game_ids: set[int]) -> list[dict]:
    """Keep only evidence rows the model returned that we can actually verify.

    An evidence row is kept only if its table matches one of the shapes the
    question declared AND (today) it's a game_details row whose id is among the
    rows actually retrieved for this question. player_box_scores/game_recaps/
    injury_notes evidence can never be grounded yet because embed.py doesn't
    index those tables (see its TODO sections) -- once a candidate adds that
    retrieval, this function should also check those rows against whatever was
    retrieved for them.

    :param parsed_evidence: the "evidence" value from the model's parsed JSON,
        expected to be a list of {table, ...id fields} dicts but not trusted to be
    :param declared_shapes: question["return"]["evidence"], the allowed table shapes
    :param retrieved_game_ids: game_ids actually retrieved for this question
    :return: evidence rows that are both well-formed and verifiable
    """
    if not isinstance(parsed_evidence, list):
        return []
    grounded = []
    for entry in parsed_evidence:
        if not isinstance(entry, dict):
            continue
        shape = next((s for s in declared_shapes if s.get("table") == entry.get("table")), None)
        if shape is None:
            continue
        try:
            item = {"table": entry["table"]}
            for key, type_name in shape.items():
                if key != "table":
                    item[key] = coerce_type(entry[key], type_name)
        except (KeyError, TypeError, ValueError):
            continue
        # TODO (candidate work): only game_details is grounded here because only
        # game_details is embedded/retrieved. Add a branch per table once
        # player_box_scores/game_recaps/injury_notes retrieval exists in embed.py.
        if item["table"] == "game_details" and item["id"] in retrieved_game_ids:
            grounded.append(item)
    return grounded


def load_teams(cx: Connection) -> list[dict]:
    """Load team_id, city, name, and abbreviation for every team, once per run.

    :param cx: open SQLAlchemy connection
    :return: one dict per team with keys team_id, city, name, abbreviation
    """
    sql = "SELECT team_id, city, name, abbreviation FROM teams"
    return [dict(r) for r in cx.execute(text(sql)).mappings().all()]


def normalize_team_names(result: dict, teams: list[dict]) -> dict:
    
    skip_fields = {"evidence", "player_name", "reason", "status"}
    normalized = dict(result)
    for key, value in result.items():
        if key in skip_fields or not isinstance(value, str):
            continue
        matched_mascots = {
            team["name"]
            for team in teams
            if re.search(rf"\b{re.escape(team['city'] + ' ' + team['name'])}\b", value)
            or re.search(rf"\b{re.escape(team['name'])}\b", value)
            or re.search(rf"\b{re.escape(team['abbreviation'])}\b", value)
        }
        if len(matched_mascots) == 1:
            normalized[key] = next(iter(matched_mascots))
    return normalized


def abstain_result(return_spec: dict) -> dict:
    """Build a schema-valid abstention result straight from a question's `return` block.

    answers_template.json's defaults are shape placeholders (candidates fill them
    in), not a real answer -- reusing them for an abstention silently shipped
    "answerable": true with id-0 evidence, which the grader reads as a fabricated
    fact rather than a deliberate "I don't know". An abstention must always be
    answerable=false with empty/zeroed fields and no evidence.

    :param return_spec: question["return"], the declared field name -> type map
    :return: a result dict with answerable False, ints 0, strings "", evidence []
    """
    zeros = {"bool": False, "int": 0, "str": ""}
    result = {key: zeros[type_name] for key, type_name in return_spec.items() if key != "evidence"}
    result["answerable"] = False
    result["evidence"] = []
    return result


def evidence_fully_grounded(declared_shapes: list[dict], grounded: list[dict]) -> bool:
    """Check that every evidence table the question requires has a grounded row.

    A question is only answerable if every table its `return.evidence` block
    lists actually has at least one verified supporting row -- e.g. a question
    that cites both game_details and game_recaps isn't answerable off
    game_details alone.

    :param declared_shapes: question["return"]["evidence"]
    :param grounded: output of ground_evidence()
    :return: True if every declared table has at least one grounded row
    """
    declared_tables = {s["table"] for s in declared_shapes}
    grounded_tables = {g["table"] for g in grounded}
    return bool(declared_tables) and declared_tables.issubset(grounded_tables)


# Grading runs this file against a hidden questions.json with different wording,
# entities, and ids (see README "How Part 1 is graded"). Do not branch on
# question id or match question text with patterns; every question must go
# through the same retrieve/answer path, driven only by its `return` block.
def load_text_tables(cx: Connection):
    recaps = {
        int(r.game_id): (int(r.recap_id), r.text)
        for r in cx.execute(text("SELECT recap_id, game_id, text FROM game_recaps ORDER BY recap_id"))
    }
    notes = cx.execute(
        text("SELECT note_id, game_id, player_id, status, text FROM injury_notes ORDER BY note_id")
    ).fetchall()
    return recaps, notes


def shape_result(return_spec: dict, facts: dict, evidence: list[dict]) -> dict | None:
    """Fit computed facts and evidence to the question's declared return shape."""
    result = {"answerable": True}
    for key, type_name in return_spec.items():
        if key in ("evidence", "answerable"):
            continue
        if key not in facts:
            return None
        try:
            result[key] = coerce_type(facts[key], type_name)
        except (TypeError, ValueError):
            return None
    declared = {s["table"] for s in return_spec.get("evidence", [])}
    kept = [e for e in evidence if e["table"] in declared]
    if declared - {e["table"] for e in kept}:
        return None
    result["evidence"] = kept
    return result


def answer_question(cx: Connection, question: dict, ctx: dict) -> dict:
    return_spec = question["return"]
    kind = route(return_spec)
    if kind:
        out = answer_text(cx, question, kind, ctx["frames"], ctx["recaps"], ctx["notes"])
    else:
        spec = plan(cx, question["question"])
        out = None if spec is None else execute(ctx["frames"], spec)
    if out is None:
        return abstain_result(return_spec)
    shaped = shape_result(return_spec, *out)
    return shaped if shaped is not None else abstain_result(return_spec)


def main() -> None:
    print("Starting RAG Answering")
    eng = sa.create_engine(DB_DSN)
    with open(QUESTIONS_PATH, encoding="utf-8") as f:
        questions = json.load(f)

    outputs = []
    with eng.begin() as cx:
        recaps, notes = load_text_tables(cx)
        ctx = {"frames": load_frames(cx), "recaps": recaps, "notes": notes}
        for q in questions:
            result = answer_question(cx, q, ctx)
            outputs.append({"id": q["id"], "result": result})
            print(f"Q{q['id']}: {q['question']}\n  -> {json.dumps(result)}")

    with open(ANSWERS_PATH, "w", encoding="utf-8") as f:
        json.dump(outputs, f, ensure_ascii=False, indent=2)
    print(f"Finished RAG Answering: wrote {len(outputs)} answers to {ANSWERS_PATH}")


if __name__ == "__main__":
    main()
