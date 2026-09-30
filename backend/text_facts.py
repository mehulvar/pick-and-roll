"""Deterministic extraction from recap and injury note text.
Text is only used after it is tied to a game row from the tables."""
import re

DEFICIT_PATTERNS = [
    re.compile(r"(\d+)[- ]point (?:deficit|hole)", re.I),
    re.compile(r"(?:trailed|trailing|down|behind) by (?:as many as |up to )?(\d+) points?", re.I),
]
GO_AHEAD = re.compile(r"go[- ]ahead|game[- ]winn(?:er|ing)|winning (?:basket|shot|bucket)", re.I)
REASON_PATTERNS = [
    re.compile(r"^due to (?:an? )?([^,]+),", re.I),
    re.compile(r"\(([^)]+)\)"),
    re.compile(r"because of (?:an? )?([^.]+)", re.I),
    re.compile(r"cites (?:an? )?([^.]+)", re.I),
    re.compile(r"\bwith (?:an? )?([^.]+)", re.I),
]
TRAILING = re.compile(r"\s+and (?:still )?played.*$", re.I)

POINTS = re.compile(r"(\d+) points")


def roster(frames, game_id, team_name):
    b = frames["box"]
    b = b[(b.game_id == game_id) & (b.team_name == team_name)]
    p = frames["players"][["player_id", "first_name", "last_name"]]
    b = b.merge(p, left_on="person_id", right_on="player_id")
    return b[["person_id", "first_name", "last_name", "points"]].to_dict("records")


def _named(sentence, players):
    full = [p for p in players
            if re.search(rf"\b{re.escape(p['first_name'] + ' ' + p['last_name'])}\b", sentence)]
    if len(full) == 1:
        return full
    return [p for p in players if re.search(rf"\b{re.escape(p['last_name'])}\b", sentence)]


def _by_stated_points(text, cands):
    hits = []
    for p in cands:
        for s in _sentences(text):
            if re.search(rf"\b{re.escape(p['last_name'])}\b", s):
                if any(int(n) == int(p["points"]) for n in POINTS.findall(s)):
                    hits.append(p)
                    break
    return hits


def go_ahead_player(text, players):
    for sentence in _sentences(text):
        if not GO_AHEAD.search(sentence):
            continue
        cands = []
        for clause in sentence.split(","):
            if GO_AHEAD.search(clause):
                cands = _named(clause, players)
                if cands:
                    break
        if not cands:
            cands = _named(sentence, players)
        if len(cands) > 1:
            cands = _by_stated_points(text, cands)
        return cands[0] if len(cands) == 1 else None
    return None

def injury_reason(text):
    for pat in REASON_PATTERNS:
        m = pat.search(text)
        if m:
            return TRAILING.sub("", m.group(1)).strip().rstrip(".")
    return None


def _sentences(text):
    return re.split(r"(?<=[.!?])\s+", text)


def deficits(text, opp_name):
    """Every deficit the winner faced, skipping ones attributed to the opponent."""
    out = []
    for sentence in _sentences(text):
        for pat in DEFICIT_PATTERNS:
            for m in pat.finditer(sentence):
                clause = sentence[:m.start()].split(",")[-1]
                if opp_name and re.search(rf"\b{re.escape(opp_name)}\b", clause):
                    continue
                out.append(int(m.group(1)))
    return out
