import argparse

import pandas as pd
import sqlalchemy as sa
from sqlalchemy import text

from backend.config import DB_DSN, EMBED_MODEL
from backend.utils import ollama_embed_batch

# nomic-embed-text is an asymmetric model: text being indexed gets the
# "search_document: " prefix and text being searched gets "search_query: ".
# Ollama does not add these itself, so each side of retrieval adds its own.
DOCUMENT_PREFIX = "search_document: "


def build_game_doc(row: pd.Series) -> str:
    """Build a natural-language document for one game_details row.

    Joins in home/away team city+name+abbreviation (from the teams table, already
    merged onto `row` by the caller) so the document reads like a sentence a
    human would write, rather than raw foreign keys. This is what game retrieval
    in rag.py and server.py searches against.

    :param row: one row of game_details left-joined with teams for home and away
        (expects home_team_label, away_team_label, home_points, away_points,
        winning_team_id, home_team_id, game_timestamp, game_id)
    :return: one-paragraph natural-language description of the game
    """
    date = pd.to_datetime(row.game_timestamp, utc=True).strftime("%Y-%m-%d")
    home_points = int(row.home_points)
    away_points = int(row.away_points)
    if row.winning_team_id == row.home_team_id:
        winner_label, winner_points = row.home_team_label, home_points
        loser_label, loser_points = row.away_team_label, away_points
    else:
        winner_label, winner_points = row.away_team_label, away_points
        loser_label, loser_points = row.home_team_label, home_points

    return (
        f"On {date}, {row.home_team_label} hosted {row.away_team_label}. "
        f"Final score: {winner_label} {winner_points}, {loser_label} {loser_points}. "
        f"Winner: {winner_label}. (game_id={int(row.game_id)})"
    )

def embed_box_scores(cx, limit):
    cx.execute(text("ALTER TABLE player_box_scores ADD COLUMN IF NOT EXISTS doc text"))
    cx.execute(text("ALTER TABLE player_box_scores ADD COLUMN IF NOT EXISTS embedding vector(768)"))
    query = (
        "SELECT b.*, g.game_timestamp, "
        "p.first_name || ' ' || p.last_name AS player_name, "
        "t.city || ' ' || t.name || ' (' || t.abbreviation || ')' AS team_label, "
        "o.city || ' ' || o.name || ' (' || o.abbreviation || ')' AS opp_label "
        "FROM player_box_scores b "
        "JOIN game_details g ON g.game_id = b.game_id "
        "JOIN players p ON p.player_id = b.person_id "
        "JOIN teams t ON t.team_id = b.team_id "
        "JOIN teams o ON o.team_id = CASE WHEN g.home_team_id = b.team_id "
        "THEN g.away_team_id ELSE g.home_team_id END "
        "ORDER BY g.game_timestamp DESC, b.game_id DESC, b.person_id DESC"
    )
    params = {}
    if limit is not None:
        query += " LIMIT :limit"
        params["limit"] = limit
    df = pd.read_sql(text(query), cx, params=params)

    docs = []
    for _, r in df.iterrows():
        date = pd.to_datetime(r.game_timestamp, utc=True).strftime("%Y-%m-%d")
        docs.append(
            f"On {date}, {r.player_name} of {r.team_label} played against {r.opp_label} "
            f"({'starter' if r.starter else 'bench'}, {r.seconds / 60:.1f} minutes). "
            f"Stat line: {int(r.points)} points, "
            f"{int(r.offensive_reb + r.defensive_reb)} rebounds, {int(r.assists)} assists, "
            f"{int(r.steals)} steals, {int(r.blocks)} blocks, {int(r.turnovers)} turnovers, "
            f"{int(r.fg2_made + r.fg3_made)}/{int(r.fg2_attempted + r.fg3_attempted)} field goals, "
            f"{int(r.fg3_made)}/{int(r.fg3_attempted)} threes, "
            f"{int(r.ft_made)}/{int(r.ft_attempted)} free throws. "
            f"(game_id={int(r.game_id)}, person_id={int(r.person_id)})"
        )
    vectors = ollama_embed_batch(EMBED_MODEL, [DOCUMENT_PREFIX + d for d in docs])
    cx.execute(
        text("UPDATE player_box_scores SET doc = :doc, embedding = :v "
             "WHERE game_id = :gid AND person_id = :pid"),
        [{"doc": d, "v": v, "gid": int(g), "pid": int(p)}
         for d, v, g, p in zip(docs, vectors, df.game_id, df.person_id)],
    )
    print(f"Embedded {len(df)} box score rows")

def embed_recaps(cx, limit):
    cx.execute(text("ALTER TABLE game_recaps ADD COLUMN IF NOT EXISTS doc text"))
    cx.execute(text("ALTER TABLE game_recaps ADD COLUMN IF NOT EXISTS embedding vector(768)"))
    query = (
        "SELECT r.recap_id, r.game_id, r.text, g.game_timestamp, "
        "home.city || ' ' || home.name || ' (' || home.abbreviation || ')' AS home_label, "
        "away.city || ' ' || away.name || ' (' || away.abbreviation || ')' AS away_label "
        "FROM game_recaps r "
        "JOIN game_details g ON g.game_id = r.game_id "
        "JOIN teams home ON home.team_id = g.home_team_id "
        "JOIN teams away ON away.team_id = g.away_team_id "
        "ORDER BY g.game_timestamp DESC, r.recap_id DESC"
    )
    params = {}
    if limit is not None:
        query += " LIMIT :limit"
        params["limit"] = limit
    df = pd.read_sql(text(query), cx, params=params)

    docs = []
    for _, r in df.iterrows():
        date = pd.to_datetime(r.game_timestamp, utc=True).strftime("%Y-%m-%d")
        docs.append(
            f"Game recap for {r.home_label} vs {r.away_label} on {date}. {r.text} "
            f"(recap_id={int(r.recap_id)}, game_id={int(r.game_id)})"
        )
    vectors = ollama_embed_batch(EMBED_MODEL, [DOCUMENT_PREFIX + d for d in docs])
    cx.execute(
        text("UPDATE game_recaps SET doc = :doc, embedding = :v WHERE recap_id = :rid"),
        [{"doc": d, "v": v, "rid": int(i)} for d, v, i in zip(docs, vectors, df.recap_id)],
    )
    print(f"Embedded {len(df)} recaps")


def embed_injury_notes(cx, limit):
    cx.execute(text("ALTER TABLE injury_notes ADD COLUMN IF NOT EXISTS doc text"))
    cx.execute(text("ALTER TABLE injury_notes ADD COLUMN IF NOT EXISTS embedding vector(768)"))
    query = (
        "SELECT n.note_id, n.game_id, n.player_id, n.note_date, n.status, n.text, "
        "p.first_name || ' ' || p.last_name AS player_name, "
        "t.city || ' ' || t.name || ' (' || t.abbreviation || ')' AS team_label "
        "FROM injury_notes n "
        "JOIN players p ON p.player_id = n.player_id "
        "JOIN teams t ON t.team_id = n.team_id "
        "ORDER BY n.note_date DESC, n.note_id DESC"
    )
    params = {}
    if limit is not None:
        query += " LIMIT :limit"
        params["limit"] = limit
    df = pd.read_sql(text(query), cx, params=params)

    docs = [
        f"Injury report for {r.player_name} of {r.team_label} on {r.note_date}. "
        f"Status: {r.status}. {r.text} "
        f"(note_id={int(r.note_id)}, game_id={int(r.game_id)}, player_id={int(r.player_id)})"
        for _, r in df.iterrows()
    ]
    vectors = ollama_embed_batch(EMBED_MODEL, [DOCUMENT_PREFIX + d for d in docs])
    cx.execute(
        text("UPDATE injury_notes SET doc = :doc, embedding = :v WHERE note_id = :nid"),
        [{"doc": d, "v": v, "nid": int(i)} for d, v, i in zip(docs, vectors, df.note_id)],
    )
    print(f"Embedded {len(df)} injury notes")


def main(limit: int | None) -> None:
    """Embed game_details documents and store them with their vectors.

    :param limit: if given, only embed this many games (most recent first),
        for a quick smoke test instead of a full run.
    """
    print("Starting Embedding Process")
    eng = sa.create_engine(DB_DSN)
    with eng.begin() as cx:
        cx.execute(text("ALTER DATABASE nba REFRESH COLLATION VERSION"))
        # `doc` holds the human-readable text so it can be inspected/debugged
        # directly in SQL; `embedding` is the vector derived from it.
        cx.execute(text("ALTER TABLE IF EXISTS game_details ADD COLUMN IF NOT EXISTS doc text;"))
        cx.execute(text("ALTER TABLE IF EXISTS game_details ADD COLUMN IF NOT EXISTS embedding vector(768);"))
        # No vector index here on purpose. game_details has under 1k rows and
        # even player_box_scores is only ~18k, so an exact cosine scan (what
        # rag.py's ORDER BY does without an index) costs milliseconds -- there's
        # no performance reason to add one. An approximate index (hnsw/ivfflat) would be actively
        # harmful for grading: pgvector's hnsw assigns node levels randomly
        # during construction, so a fresh embed builds a differently-shaped
        # graph every time, which can flip which rows tie for the last
        # top-k slot and make retrieve() disagree with the committed
        # part1/answers.json. If you add an index anyway, make sure
        # retrieval still returns results in the same order on every build
        # (e.g. keep an explicit tie-breaker in ORDER BY, as rag.py does).

        query = (
            "SELECT g.game_id, g.game_timestamp, g.home_team_id, g.away_team_id, "
            "g.home_points, g.away_points, g.winning_team_id, "
            "home.city || ' ' || home.name || ' (' || home.abbreviation || ')' AS home_team_label, "
            "away.city || ' ' || away.name || ' (' || away.abbreviation || ')' AS away_team_label "
            "FROM game_details g "
            "JOIN teams home ON home.team_id = g.home_team_id "
            "JOIN teams away ON away.team_id = g.away_team_id "
            "ORDER BY g.game_timestamp DESC, g.game_id DESC"
        )
        if limit is not None:
            query += " LIMIT :limit"
            df = pd.read_sql(text(query), cx, params={"limit": limit})
        else:
            df = pd.read_sql(text(query), cx)

        docs = [build_game_doc(r) for _, r in df.iterrows()]
        prefixed = [DOCUMENT_PREFIX + d for d in docs]
        vectors = ollama_embed_batch(EMBED_MODEL, prefixed)

        for game_id, doc, vec in zip(df.game_id, docs, vectors):
            cx.execute(
                text("UPDATE game_details SET doc = :doc, embedding = :v WHERE game_id = :gid"),
                {"doc": doc, "v": vec, "gid": int(game_id)},
            )
        # Also embed player box scores (one document per player-game)
        embed_box_scores(cx, limit)
        # Embed full game recaps and injury notes as well
        embed_recaps(cx, limit)
        embed_injury_notes(cx, limit)
    print(f"Finished Embeddings: {len(df)} Rows Updated")


# ---------------------------------------------------------------------------
# TODO (candidate work): player_box_scores documents.
#
# Build one document per player_box_scores row (or per player-game) that reads
# naturally: player name (join players), their team and the opponent (join
# teams via game_details), the game date, and their stat line (points,
# rebounds, assists, etc.). Embed and store the same way as game_details
# (a `doc` + `embedding vector(768)` column, batched with ollama_embed_batch
# and the "search_document: " prefix; no vector index needed, see the note
# above main()). This is what lets
# rag.py answer player-specific questions like "how many points did X score
# in game Y" instead of only whole-game summaries.
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# TODO (candidate work): game_recaps chunks.
#
# game_recaps.text holds a full recap per game. Decide how to chunk it (whole
# recap as one document is probably fine given the length here, but consider
# sentence/paragraph chunking if recaps get longer), embed each chunk, and
# store doc + embedding (+ a chunk index if you split). Recaps carry narrative
# detail (who hit the go-ahead shot, largest deficit overcome, etc.) that
# never shows up in the box score, so several of the harder questions in
# part1/questions.json are only answerable once this retrieval path exists.
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# TODO (candidate work): injury_notes rows.
#
# Each injury_notes row is already a short natural-language note; embed
# note.text directly (optionally prefixed with player/team/date context pulled
# via joins so retrieval isn't relying on the note text alone) and store
# doc + embedding. This is the retrieval path questions about why a player
# missed a game depend on.
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Embed game_details documents into pgvector.")
    parser.add_argument("--limit", type=int, default=None, help="Only embed the first N games, for a quick smoke run.")
    args = parser.parse_args()
    main(args.limit)
