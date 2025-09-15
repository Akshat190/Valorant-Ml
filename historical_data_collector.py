"""
Historical data collector using VLR (orlandomm) API for the last N pages
as a proxy for 2-3 years. Produces:

- data/processed/teams.csv              (aggregated team win/loss, win_rate)
- data/processed/previous_matches.csv   (flat match list)
- data/processed/map_win_rate.csv       (from map_stats_collector aggregation)

Notes:
- The VLR results endpoint supports pagination via `?page=`. Since exact
  date filters are not available, we approximate 2-3 years by fetching
  many pages (configurable via max_pages).
"""

from __future__ import annotations

import os
import time
from typing import Dict, List, Tuple

import requests
import pandas as pd

VLR_BASE = "https://vlr.orlandomm.net/api/v1"


def fetch_results_paginated(max_pages: int = 50, delay_s: float = 0.25) -> List[Dict]:
    session = requests.Session()
    session.headers.update({"User-Agent": "Valorant-ML/HistoricalCollector/1.0"})
    all_rows: List[Dict] = []
    for page in range(1, max_pages + 1):
        url = f"{VLR_BASE}/results?page={page}"
        try:
            resp = session.get(url, timeout=20)
            if resp.status_code != 200:
                break
            payload = resp.json()
            rows = payload.get("data", []) if isinstance(payload, dict) else []
            if not rows:
                break
            all_rows.extend(rows)
        except Exception:
            # continue on error
            pass
        time.sleep(delay_s)
    return all_rows


def build_previous_matches(results_rows: List[Dict]) -> pd.DataFrame:
    records: List[Dict] = []
    for r in results_rows:
        try:
            rid = r.get("id")
            status = r.get("status")
            tournament = r.get("tournament")
            event = r.get("event")
            teams = r.get("teams", [])
            team1 = teams[0] if len(teams) > 0 else {}
            team2 = teams[1] if len(teams) > 1 else {}
            t1_name = team1.get("name")
            t2_name = team2.get("name")
            t1_score = team1.get("score")
            t2_score = team2.get("score")
            winner = None
            if isinstance(t1_score, (int, float)) and isinstance(t2_score, (int, float)):
                if t1_score > t2_score:
                    winner = t1_name
                elif t2_score > t1_score:
                    winner = t2_name
            records.append({
                "match_id": rid,
                "team1": t1_name,
                "team2": t2_name,
                "team1_score": t1_score,
                "team2_score": t2_score,
                "winner": winner,
                "status": status,
                "event": event,
                "tournament": tournament,
            })
        except Exception:
            continue
    return pd.DataFrame(records)


def build_team_table(prev_matches_df: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict] = []
    for _, m in prev_matches_df.iterrows():
        team1 = m.get("team1")
        team2 = m.get("team2")
        winner = m.get("winner")
        if not team1 or not team2:
            continue
        for team_name, opponent in [(team1, team2), (team2, team1)]:
            win = 1 if winner == team_name else 0
            loss = 1 if (winner and winner != team_name) else 0
            rows.append({
                "team": team_name,
                "opponent": opponent,
                "played": 1,
                "wins": win,
                "losses": loss,
            })
    if not rows:
        return pd.DataFrame(columns=["team", "played", "wins", "losses", "win_rate"])
    df = pd.DataFrame(rows)
    agg = df.groupby(["team"], as_index=False).sum(numeric_only=True)
    agg["win_rate"] = agg.apply(lambda r: (r["wins"] / r["played"]) if r["played"] else 0.0, axis=1)
    agg = agg[["team", "played", "wins", "losses", "win_rate"]]
    return agg.sort_values(["played", "win_rate"], ascending=[False, False])


def save_csv(df: pd.DataFrame, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8")


def collect_historical(max_pages: int = 80) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Collect historical matches and produce three CSV files.

    Returns: teams_df, map_win_rate_df, previous_matches_df
    """
    results_rows = fetch_results_paginated(max_pages=max_pages)
    prev_matches = build_previous_matches(results_rows)
    teams_df = build_team_table(prev_matches)

    # Map win rate: reuse map_stats_collector aggregation if available
    try:
        from map_stats_collector import collect_map_stats
        map_stats, _ = collect_map_stats(limit=None)
        map_win_rate_df = map_stats.rename(columns={
            "team": "team",
            "map": "map",
            "played": "played",
            "wins": "wins",
            "losses": "losses",
            "win_rate": "win_rate",
        })
    except Exception:
        map_win_rate_df = pd.DataFrame(columns=["team", "map", "played", "wins", "losses", "win_rate"])

    # Save
    save_csv(teams_df, "data/processed/teams.csv")
    save_csv(map_win_rate_df, "data/processed/map_win_rate.csv")
    save_csv(prev_matches, "data/processed/previous_matches.csv")

    print("✓ Saved teams.csv, map_win_rate.csv, previous_matches.csv in data/processed/")
    return teams_df, map_win_rate_df, prev_matches


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Collect historical datasets")
    parser.add_argument("--max-pages", type=int, default=80, help="How many results pages to fetch")
    args = parser.parse_args()
    collect_historical(max_pages=args.max_pages)


if __name__ == "__main__":
    main()


