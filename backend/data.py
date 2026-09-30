"""Load the structured tables into pandas frames with derived columns."""
import pandas as pd
from sqlalchemy import text


def load_frames(cx):
    games = pd.read_sql(text(
        "SELECT game_id, game_timestamp AS ts, home_team_id, away_team_id, "
        "home_points, away_points, winning_team_id FROM game_details"), cx)
    teams = pd.read_sql(text("SELECT team_id, name FROM teams"), cx)
    players = pd.read_sql(text("SELECT player_id, first_name, last_name FROM players"), cx)
    box = pd.read_sql(text(
        "SELECT game_id, person_id, team_id, starter, points FROM player_box_scores"), cx)

    games["date"] = pd.to_datetime(games["ts"]).dt.strftime("%Y-%m-%d")
    names = dict(zip(teams.team_id, teams.name))

    def side(own, opp, own_pts, opp_pts, home):
        d = games[["game_id", "ts", "date", "winning_team_id"]].copy()
        d["team_id"], d["opp_id"] = games[own], games[opp]
        d["team_pts"], d["opp_pts"] = games[own_pts], games[opp_pts]
        d["is_home"] = home
        return d

    tg = pd.concat([
        side("home_team_id", "away_team_id", "home_points", "away_points", True),
        side("away_team_id", "home_team_id", "away_points", "home_points", False),
    ], ignore_index=True)
    tg["team_name"] = tg.team_id.map(names)
    tg["opp_name"] = tg.opp_id.map(names)
    tg["margin"] = tg.team_pts - tg.opp_pts
    tg["won"] = tg.winning_team_id == tg.team_id
    tg = tg.drop(columns=["winning_team_id"])

    players["player_name"] = players.first_name + " " + players.last_name
    pg = box.merge(players[["player_id", "player_name"]], left_on="person_id", right_on="player_id")
    pg = pg.drop(columns=["player_id"]).merge(tg, on=["game_id", "team_id"])
    return {"games": tg, "box": pg, "teams": teams, "players": players}