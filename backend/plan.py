"""Turn a question into a query spec. The model only picks operations.
Entities and dates come from code, so they can never be invented."""
import json
import re

from backend.config import LLM_MODEL
from backend.resolve import resolve
from backend.utils import ollama_generate

ENUMS = {
    "level": {"game", "player"},
    "agg": {"sum", "top_by_sum", "top_by_count", "top_single"},
    "result": {"win", "loss"},
    "ordinal": {"first", "last"},
    "extreme": {"min_margin", "max_margin"},
    "side": {"team", "opponent"},
}

INSTRUCTIONS = (
    "Convert the question into a JSON query spec. Output only the JSON object. Omit fields that do not apply.\n"
    'level: "game" if the question asks for a game score, winner, opponent, or margin. "player" if it asks about player points.\n'
    'agg (player level only): "sum" total points of a named player, "top_by_sum" player with the most total points, '
    '"top_by_count" player with the most games of at least min_points, "top_single" player with the most points in one game.\n'
    'result: "win" or "loss" for the team.\n'
    'ordinal: "first" or "last" game.\n'
    'extreme: "max_margin" for the largest win, "min_margin" for the largest loss. Set result too.\n'
    "min_margin: integer, only games decided by at least this many points.\n"
    "min_points: integer, only player games with at least this many points.\n"
    "starter_only: true if only starters count.\n"
    'side: "team" for the team\'s own players, "opponent" for players facing the team.\n'
    "team, opponent, player: copy exactly from the candidates. Never invent a name.\n"
)

EXAMPLES = [
    ("How many points did Ann Lee score in March 2026?", [], ["Ann Lee"],
     {"level": "player", "agg": "sum", "player": "Ann Lee"}),
    ("Which Falcons starter scored the most total points in games the Falcons won by 15 or more?", ["Falcons"], [],
     {"level": "player", "agg": "top_by_sum", "team": "Falcons", "side": "team",
      "result": "win", "min_margin": 15, "starter_only": True}),
    ("What was the Falcons' largest margin of victory and who did they beat?", ["Falcons"], [],
     {"level": "game", "team": "Falcons", "result": "win", "extreme": "max_margin"}),
    ("How many points did Ann Lee score in the Falcons' first game?", ["Falcons"], ["Ann Lee"],
     {"level": "player", "agg": "sum", "team": "Falcons", "player": "Ann Lee",
      "ordinal": "first", "side": "team"}),
    ("Which player had the most 20 point games against the Falcons?", ["Falcons"], [],
     {"level": "player", "agg": "top_by_count", "team": "Falcons", "side": "opponent", "min_points": 20}),
    ("How many points did Ann Lee score against the Kings on May 1, 2026?", ["Kings"], ["Ann Lee"],
     {"level": "player", "agg": "sum", "team": "Kings", "player": "Ann Lee"}),
    ("Who led the Falcons in scoring in their loss to the Kings on May 1, 2026?", ["Falcons", "Kings"], [],
     {"level": "player", "agg": "top_single", "team": "Falcons", "opponent": "Kings",
      "side": "team", "result": "loss"}),
    ("What was the score of the Falcons game against the Kings on May 1, 2026 and who won?",
     ["Falcons", "Kings"], [],
     {"level": "game", "team": "Falcons", "opponent": "Kings"}),
    ("What was the Falcons' largest margin of defeat and who beat them?", ["Falcons"], [],
     {"level": "game", "team": "Falcons", "result": "loss", "extreme": "min_margin"}),
    ("How many points did Ann Lee score in the Falcons' season opener?", ["Falcons"], ["Ann Lee"],
     {"level": "player", "agg": "sum", "team": "Falcons", "player": "Ann Lee",
      "ordinal": "first", "side": "team"}),
]


def build_prompt(question, teams, players):
    parts = [INSTRUCTIONS, "Examples:"]
    for q, t, p, spec in EXAMPLES:
        parts.append(f"Question: {q}\nteams: {json.dumps(t)}\nplayers: {json.dumps(p)}\nJSON: {json.dumps(spec)}")
    parts.append(f"Now the real question.\nQuestion: {question}\nteams: {json.dumps(teams)}\n"
                 f"players: {json.dumps(players)}\nJSON:")
    return "\n\n".join(parts)



CUES = {
    ("ordinal", "first"): r"\b(first|opener|opening game|debut)\b",
    ("ordinal", "last"): r"\b(last game|final game|finale|most recent game)\b",
    ("result", "loss"): r"\b(loss|losses|lose|defeat|defeats|defeated)\b|(?<!who )\blost\b",
    ("result", "win"): r"\b(win|wins|victory|victories)\b|(?<!who )\bwon\b",
}


def cue_fields(question: str) -> dict:
    q = question.lower()
    found = {}
    for (field, value), pattern in CUES.items():
        if re.search(pattern, q):
            found.setdefault(field, set()).add(value)
    out = {k: next(iter(v)) for k, v in found.items() if len(v) == 1}
    if "deficit" not in q:
        if re.search(r"\b(largest|biggest|greatest|worst)\b[^.?]*\b(defeat|loss)", q):
            out.update({"extreme": "min_margin", "result": "loss"})
        elif re.search(r"\b(largest|biggest|greatest)\b[^.?]*\b(victory|win)", q):
            out.update({"extreme": "max_margin", "result": "win"})
    return out


def plan(cx, question):
    ents = resolve(cx, question)
    teams = [t["name"] for t in ents["teams"]]
    players = [f"{p['first_name']} {p['last_name']}" for p in ents["players"]]

    raw = ollama_generate(LLM_MODEL, build_prompt(question, teams, players))
    m = re.search(r"\{.*\}", raw, re.DOTALL)
    if not m:
        return None
    try:
        parsed = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    if not isinstance(parsed, dict):
        return None

    spec = {}
    for key, allowed in ENUMS.items():
        v = parsed.get(key)
        if isinstance(v, str) and v in allowed:
            spec[key] = v
    if "level" not in spec:
        return None
    if spec["level"] == "player" and "agg" not in spec:
        return None
    for key in ("min_margin", "min_points"):
        v = parsed.get(key)
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            spec[key] = int(v)
    if parsed.get("starter_only") is True:
        spec["starter_only"] = True
    for key, cands in (("team", teams), ("opponent", teams), ("player", players)):
        v = parsed.get(key)
        if isinstance(v, str) and v in cands:
            spec[key] = v

    if len(players) == 1 and spec.get("agg", "sum") == "sum":
        spec["player"] = players[0]

    
    for key, val in cue_fields(question).items():
        spec.setdefault(key, val)

    if "team" not in spec and teams:
        spec["team"] = teams[0]
    if "opponent" not in spec:
        others = [t for t in teams if t != spec.get("team")]
        if others:
            spec["opponent"] = others[0]
    if spec.get("level") == "player" and spec.get("agg") == "sum" and "player" not in spec:
        return None
    if len(teams) == 2:
        spec["team"], spec["opponent"] = teams[0], teams[1]
        if "player" in spec:
            spec.pop("side", None)
        else:
            spec["side"] = "team"

    starts = [d["start"] for d in ents["dates"] if d["start"]]
    ends = [d["end"] for d in ents["dates"] if d["end"]]
    if starts:
        spec["date_start"] = min(starts)
    if ends:
        spec["date_end"] = max(ends)
    return spec