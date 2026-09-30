"""Run a query spec against the frames. Returns (facts, evidence) or None to abstain."""


def _games(tg, spec):
    g = tg[tg.team_name == spec["team"]] if spec.get("team") else tg[tg.is_home]
    if spec.get("opponent"):
        g = g[g.opp_name == spec["opponent"]]
    if spec.get("date_start"):
        g = g[g.date >= spec["date_start"]]
    if spec.get("date_end"):
        g = g[g.date <= spec["date_end"]]
    if spec.get("result") == "win":
        g = g[g.won]
    elif spec.get("result") == "loss":
        g = g[~g.won]
    if spec.get("min_margin"):
        g = g[g.margin.abs() >= spec["min_margin"]]
    if spec.get("extreme") == "min_margin":
        g = g.sort_values(["margin", "ts", "game_id"]).head(1)
    elif spec.get("extreme") == "max_margin":
        g = g.sort_values(["margin", "ts", "game_id"], ascending=[False, True, True]).head(1)
    g = g.sort_values(["ts", "game_id"])
    if spec.get("ordinal") == "first":
        g = g.head(1)
    elif spec.get("ordinal") == "last":
        g = g.tail(1)
    return g


def _player_rows(pg, game_ids, spec):
    r = pg[pg.game_id.isin(game_ids)]
    if spec.get("side") == "team":
        r = r[r.team_name == spec["team"]]
    elif spec.get("side") == "opponent":
        r = r[r.opp_name == spec["team"]]
    if spec.get("player"):
        r = r[r.player_name == spec["player"]]
    if spec.get("starter_only"):
        r = r[r.starter]
    if spec.get("min_points") is not None:
        r = r[r.points >= spec["min_points"]]
    return r


def execute(frames, spec):
    games = _games(frames["games"], spec)
    if games.empty:
        return None
    if spec["level"] == "game" and len(games) != 1:
        return None
    if spec["level"] == "game":
        g = games.iloc[0]
        hi, lo = max(g.team_pts, g.opp_pts), min(g.team_pts, g.opp_pts)
        facts = {
            "winner": g.team_name if g.won else g.opp_name,
            "opponent": g.opp_name,
            "margin": int(abs(g.margin)),
            "score": f"{int(hi)}-{int(lo)}",
        }
        return facts, [{"table": "game_details", "id": int(g.game_id)}]

    rows = _player_rows(frames["box"], set(games.game_id), spec)
    if rows.empty:
        return None
    agg = spec["agg"]
    if agg != "sum":
        how = {"top_by_sum": "sum", "top_by_count": "size", "top_single": "max"}[agg]
        s = rows.groupby(["player_name", "person_id"]).points.agg(how).reset_index(name="v")
        best = s.sort_values(["v", "player_name"], ascending=[False, True]).iloc[0]
        rows = rows[rows.person_id == best.person_id]
        if agg == "top_single":
            rows = rows.sort_values(["points", "ts", "game_id"], ascending=[False, True, True]).head(1)
    facts = {"points": int(rows.points.sum()), "games": int(len(rows))}
    if spec.get("player") or agg != "sum":
        facts["player_name"] = rows.iloc[0].player_name
    evidence = [{"table": "game_details", "id": int(i)} for i in sorted(rows.game_id.unique())]
    evidence += [{"table": "player_box_scores", "game_id": int(r.game_id), "person_id": int(r.person_id)}
                 for r in rows.sort_values(["game_id", "person_id"]).itertuples()]
    return facts, evidence