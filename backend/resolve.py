"""Resolve teams, players, and dates mentioned in a question against the tables."""
import calendar
import json
import re
from datetime import date

import sqlalchemy as sa
from sqlalchemy import text

from backend.config import DB_DSN

MONTHS = {name.lower(): i for i, name in enumerate(calendar.month_name) if name}
MONTH_ALT = "|".join(MONTHS)
FULL_DATE = re.compile(rf"\b({MONTH_ALT})\s+(\d{{1,2}})(?:st|nd|rd|th)?,?\s+(\d{{4}})\b", re.I)
MONTH_YEAR = re.compile(rf"\b({MONTH_ALT})\s+(\d{{4}})\b", re.I)


def _word(s: str) -> re.Pattern:
    return re.compile(rf"\b{re.escape(s)}\b", re.I)


def find_teams(q: str, teams: list[dict]) -> list[dict]:
    city_count = {}
    for t in teams:
        city_count[t["city"]] = city_count.get(t["city"], 0) + 1
    hits = {}
    for t in teams:
        m = _word(t["name"]).search(q)
        if not m and city_count[t["city"]] == 1:
            m = _word(t["city"]).search(q)
        if not m:
            m = re.search(rf"\b{re.escape(t['abbreviation'])}\b", q)
        if m:
            hits[t["team_id"]] = (m.start(), t)
    return [t for _, t in sorted(hits.values(), key=lambda h: h[0])]


def find_players(q: str, players: list[dict]) -> list[dict]:
    hits = []
    for p in players:
        m = _word(f"{p['first_name']} {p['last_name']}").search(q)
        if m:
            hits.append((m.start(), p))
    return [p for _, p in sorted(hits, key=lambda h: h[0])]


def find_dates(q: str) -> list[dict]:
    """Return date ranges as {start, end}. None means open ended."""
    ranges = []
    for m in FULL_DATE.finditer(q):
        d = date(int(m.group(3)), MONTHS[m.group(1).lower()], int(m.group(2)))
        ranges.append({"start": d.isoformat(), "end": d.isoformat()})
    rest = FULL_DATE.sub(" ", q)
    for m in MONTH_YEAR.finditer(rest):
        y, mo = int(m.group(2)), MONTHS[m.group(1).lower()]
        last = calendar.monthrange(y, mo)[1]
        ranges.append({"start": date(y, mo, 1).isoformat(), "end": date(y, mo, last).isoformat()})
    return ranges


def resolve(cx, question: str) -> dict:
    teams = [dict(r) for r in cx.execute(text("SELECT team_id, city, name, abbreviation FROM teams")).mappings()]
    players = [dict(r) for r in cx.execute(text("SELECT player_id, team_id, first_name, last_name FROM players")).mappings()]
    return {
        "teams": find_teams(question, teams),
        "players": find_players(question, players),
        "dates": find_dates(question),
    }


if __name__ == "__main__":
    from backend.rag import QUESTIONS_PATH

    with open(QUESTIONS_PATH, encoding="utf-8") as f:
        questions = json.load(f)
    eng = sa.create_engine(DB_DSN)
    with eng.begin() as cx:
        for q in questions:
            r = resolve(cx, q["question"])
            print(q["id"], "teams:", [t["name"] for t in r["teams"]],
                  "players:", [f"{p['first_name']} {p['last_name']}" for p in r["players"]],
                  "dates:", r["dates"])
