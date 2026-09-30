"""Answer questions whose evidence lives in recap or injury note text.
The game is pinned from the tables first, then only that game's text is read."""
from backend.execute import _games
from backend.plan import cue_fields
from backend.resolve import resolve
from backend.text_facts import deficits, go_ahead_player, injury_reason, roster
from backend.retrieval import pick_game


def route(return_spec):
    tables = {s["table"] for s in return_spec.get("evidence", [])}
    fields = set(return_spec) - {"evidence", "answerable"}
    if "injury_notes" in tables:
        return "injury"
    if "game_recaps" in tables:
        if "deficit" in fields:
            return "deficit"
        if "player_name" in fields:
            return "go_ahead"
    return None


def _spec(cx, question):
    ents = resolve(cx, question)
    teams = [t["name"] for t in ents["teams"]]
    spec = dict(cue_fields(question))
    spec.pop("extreme", None)
    if teams:
        spec["team"] = teams[0]
    if len(teams) > 1:
        spec["opponent"] = teams[1]
    starts = [d["start"] for d in ents["dates"] if d["start"]]
    ends = [d["end"] for d in ents["dates"] if d["end"]]
    if starts:
        spec["date_start"] = min(starts)
    if ends:
        spec["date_end"] = max(ends)
    if "team" not in spec:
        spec.pop("result", None)
    return spec, ents


def deficit_answer(f, spec, recaps):
    if "team" not in spec:
        return None
    best = None
    for g in _games(f["games"], {**spec, "result": "win"}).itertuples():
        rid, txt = recaps.get(int(g.game_id), (None, None))
        if txt is None:
            continue
        ds = deficits(txt, g.opp_name)
        if ds and (best is None or max(ds) > best[0]):
            best = (max(ds), g, rid)
    if best is None:
        return None
    d, g, rid = best
    hi, lo = max(g.team_pts, g.opp_pts), min(g.team_pts, g.opp_pts)
    facts = {"opponent": g.opp_name, "deficit": int(d), "score": f"{int(hi)}-{int(lo)}"}
    evidence = [{"table": "game_details", "id": int(g.game_id)},
                {"table": "game_recaps", "id": int(rid)}]
    return facts, evidence


def go_ahead_answer(cx, question, f, spec, recaps):
    games = _games(f["games"], spec)
    if games.empty:
        return None
    if len(games) > 1:
        if "team" not in spec:
            return None
        gid = pick_game(cx, question, [int(x) for x in games.game_id], "game_recaps")
        if gid is None:
            return None
        games = games[games.game_id == gid]
    g = games.iloc[0]
    rid, txt = recaps.get(int(g.game_id), (None, None))
    if txt is None:
        return None
    winner = g.team_name if g.won else g.opp_name
    p = go_ahead_player(txt, roster(f, int(g.game_id), winner))
    if p is None:
        return None
    facts = {"player_name": f"{p['first_name']} {p['last_name']}"}
    evidence = [
        {"table": "game_details", "id": int(g.game_id)},
        {"table": "game_recaps", "id": int(rid)},
        {"table": "player_box_scores", "game_id": int(g.game_id), "person_id": int(p["person_id"])},
    ]
    return facts, evidence


def injury_answer(cx, question, f, spec, ents, notes):
    if len(ents["players"]) != 1:
        return None
    pid = int(ents["players"][0]["player_id"])
    gids = {int(x) for x in _games(f["games"], spec).game_id}
    hits = [n for n in notes if int(n.player_id) == pid and int(n.game_id) in gids]
    if not hits:
        return None
    if len(hits) > 1:
        gid = pick_game(cx, question, [int(n.game_id) for n in hits], "injury_notes")
        if gid is None:
            return None
        hits = [n for n in hits if int(n.game_id) == gid]
    n = hits[0]
    box = f["box"]
    played = ((box.game_id == n.game_id) & (box.person_id == n.player_id)).any()
    if n.status != "out" or played:
        return None
    reason = injury_reason(n.text)
    if not reason:
        return None
    facts = {"status": n.status, "reason": reason}
    evidence = [{"table": "injury_notes", "id": int(n.note_id)},
                {"table": "game_details", "id": int(n.game_id)}]
    return facts, evidence


def answer_text(cx, question, kind, f, recaps, notes):
    q = question["question"]
    spec, ents = _spec(cx, q)
    if kind == "deficit":
        return deficit_answer(f, spec, recaps)
    if kind == "go_ahead":
        return go_ahead_answer(cx, q, f, spec, recaps)
    return injury_answer(cx, q, f, spec, ents, notes)