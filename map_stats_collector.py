"""
Collector for map-level statistics using unofficial VLR API endpoints.

Outputs:
- data/raw/map_details_<timestamp>.json (raw match detail payloads list)
- data/processed/team_map_stats.csv
- data/processed/team_map_vs_opponent.csv

Strategy:
- Seed match ids from latest files in data/raw/ (matches/results)
- Probe multiple match detail endpoint patterns; skip on errors
- Parse per-map winners/scores when present; aggregate per team and per opponent
"""

from __future__ import annotations

import os
import json
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import requests
import pandas as pd


VLR_BASE = "https://vlr.orlandomm.net/api/v1"
VLRGG_BASE = "https://vlrggapi.vercel.app"


def _list_latest_seed_files() -> Dict[str, Optional[str]]:
    os.makedirs("data/raw", exist_ok=True)
    seeds: Dict[str, Optional[str]] = {"events": None, "matches": None, "results": None}

    def _latest(glob_pattern: str) -> Optional[str]:
        import glob
        files = glob.glob(glob_pattern)
        if not files:
            return None
        return max(files, key=os.path.getctime)

    seeds["matches"] = _latest("data/raw/valorant_matches_*.csv")
    # results can be under two patterns
    seeds["results"] = _latest("data/raw/valorant_complete_data_results_*.csv")
    if seeds["results"] is None:
        seeds["results"] = _latest("data/raw/*results*.csv")
    return seeds


def _load_seed_match_ids() -> List[str]:
    seeds = _list_latest_seed_files()
    match_ids: List[str] = []

    # Matches CSVs produced by our collector have column 'match_id'
    if seeds["matches"] and os.path.exists(seeds["matches"]):
        try:
            df = pd.read_csv(seeds["matches"])
            if "match_id" in df.columns:
                match_ids.extend([str(x) for x in df["match_id"].dropna().astype(str).tolist()])
        except Exception:
            pass

    # Results CSV from VLR API typically has 'id'
    if seeds["results"] and os.path.exists(seeds["results"]):
        try:
            df = pd.read_csv(seeds["results"])
            id_col = "id" if "id" in df.columns else None
            if id_col:
                match_ids.extend([str(x) for x in df[id_col].dropna().astype(str).tolist()])
        except Exception:
            pass

    # Deduplicate while preserving order
    seen = set()
    unique_ids: List[str] = []
    for mid in match_ids:
        if mid and mid not in seen:
            seen.add(mid)
            unique_ids.append(mid)
    return unique_ids


def _fetch_match_detail(session: requests.Session, match_id: str) -> Optional[Dict]:
    candidates = [
        # orlandomm variants
        f"{VLR_BASE}/match/{match_id}",
        f"{VLR_BASE}/match?id={match_id}",
        f"{VLR_BASE}/matches/{match_id}",
        # vlrggapi (if it exposes match by id)
        f"{VLRGG_BASE}/match?id={match_id}",
    ]
    for url in candidates:
        try:
            resp = session.get(url, timeout=15)
            if resp.status_code == 200:
                data = resp.json()
                # Some endpoints wrap payload
                return data
        except Exception:
            pass
        time.sleep(0.25)
    return None


def _parse_maps_from_detail(detail: Dict) -> List[Dict]:
    """Return list of standardized map results.

    Output item keys:
    - map_name, team1_name, team2_name, team1_score, team2_score, map_winner
    """
    results: List[Dict] = []

    if not detail:
        return results

    payload = detail.get("data", detail)

    # Common patterns: payload may have 'maps' as list of maps
    maps = payload.get("maps") if isinstance(payload, dict) else None
    if isinstance(maps, list):
        for m in maps:
            try:
                map_name = m.get("name") or m.get("map") or "Unknown"
                teams = m.get("teams") or m.get("team") or []
                if isinstance(teams, list) and len(teams) >= 2:
                    t1, t2 = teams[0], teams[1]
                    team1_name = t1.get("name")
                    team2_name = t2.get("name")
                    team1_score = int(t1.get("score") or 0)
                    team2_score = int(t2.get("score") or 0)
                else:
                    # fallback to top-level teams
                    tlist = payload.get("teams", [])
                    if isinstance(tlist, list) and len(tlist) >= 2:
                        team1_name = tlist[0].get("name")
                        team2_name = tlist[1].get("name")
                    else:
                        team1_name = payload.get("team1") or "Team 1"
                        team2_name = payload.get("team2") or "Team 2"
                    team1_score = int(m.get("team1_score") or 0)
                    team2_score = int(m.get("team2_score") or 0)

                if team1_score > team2_score:
                    map_winner = team1_name
                elif team2_score > team1_score:
                    map_winner = team2_name
                else:
                    map_winner = "Tied"

                results.append({
                    "map_name": map_name,
                    "team1_name": team1_name,
                    "team2_name": team2_name,
                    "team1_score": team1_score,
                    "team2_score": team2_score,
                    "map_winner": map_winner,
                })
            except Exception:
                continue

    return results


def enrich_map_metadata(df_stats: pd.DataFrame) -> pd.DataFrame:
    """Add map metadata (uuid, displayName) from valorant-api.com if available."""
    try:
        resp = requests.get("https://valorant-api.com/v1/maps", timeout=20)
        if resp.status_code != 200:
            return df_stats
        data = resp.json().get("data", [])
        meta = []
        for m in data:
            name = (m.get("displayName") or "").strip()
            meta.append({
                "map": name,
                "map_uuid": m.get("uuid"),
                "map_display_name": name,
            })
        meta_df = pd.DataFrame(meta)
        if meta_df.empty:
            return df_stats
        # Left join on normalized map name
        left = df_stats.copy()
        left["map_norm"] = left["map"].str.strip().str.lower()
        meta_df["map_norm"] = meta_df["map"].str.strip().str.lower()
        out = left.merge(meta_df.drop_duplicates("map_norm"), on="map_norm", how="left", suffixes=("", "_meta"))
        out = out.drop(columns=[c for c in out.columns if c.endswith("_meta")])
        out = out.drop(columns=["map_norm"], errors="ignore")
        return out
    except Exception:
        return df_stats


def collect_map_stats(limit: Optional[int] = None, per_request_delay_s: float = 0.25) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Fetch map-level details and aggregate per-team map stats.

    Returns:
      team_map_stats_df, team_map_vs_opponent_df
    """
    session = requests.Session()
    session.headers.update({"User-Agent": "Valorant-ML/MapStats/1.0"})

    match_ids = _load_seed_match_ids()
    if limit is not None:
        match_ids = match_ids[:limit]

    raw_details: List[Dict] = []
    per_map_rows: List[Dict] = []

    for idx, mid in enumerate(match_ids, start=1):
        detail = _fetch_match_detail(session, mid)
        if detail:
            raw_details.append({"match_id": mid, "detail": detail})
            maps = _parse_maps_from_detail(detail)
            per_map_rows.extend([{**row, "match_id": mid} for row in maps])
        if per_request_delay_s:
            time.sleep(per_request_delay_s)

    # Save raw details snapshot
    os.makedirs("data/raw", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    with open(f"data/raw/map_details_{timestamp}.json", "w", encoding="utf-8") as f:
        json.dump(raw_details, f, ensure_ascii=False, indent=2)

    # Build DataFrame from per-map rows
    if per_map_rows:
        df_maps = pd.DataFrame(per_map_rows)
    else:
        df_maps = pd.DataFrame(columns=[
            "match_id", "map_name", "team1_name", "team2_name",
            "team1_score", "team2_score", "map_winner"
        ])

    # Aggregations
    stats_rows: List[Dict] = []
    vs_rows: List[Dict] = []

    for _, row in df_maps.iterrows():
        map_name = row.get("map_name", "Unknown")
        t1 = row.get("team1_name")
        t2 = row.get("team2_name")
        w = row.get("map_winner")
        if not t1 or not t2:
            continue

        # team1 perspective
        for team_name, opponent_name, team_score, opp_score in [
            (t1, t2, row.get("team1_score", 0), row.get("team2_score", 0)),
            (t2, t1, row.get("team2_score", 0), row.get("team1_score", 0)),
        ]:
            win = 1 if w == team_name else 0
            loss = 1 if (w != "Tied" and w == opponent_name) else 0
            stats_rows.append({
                "team": team_name,
                "map": map_name,
                "played": 1,
                "wins": win,
                "losses": loss,
            })
            vs_rows.append({
                "team": team_name,
                "opponent": opponent_name,
                "map": map_name,
                "played": 1,
                "wins": win,
                "losses": loss,
            })

    if stats_rows:
        df_stats = pd.DataFrame(stats_rows)
        df_stats = (
            df_stats.groupby(["team", "map"], as_index=False)
            .sum(numeric_only=True)
        )
        df_stats["win_rate"] = df_stats.apply(
            lambda r: (r["wins"] / r["played"]) if r["played"] else 0.0, axis=1
        )
    else:
        df_stats = pd.DataFrame(columns=["team", "map", "played", "wins", "losses", "win_rate"])

    if vs_rows:
        df_vs = pd.DataFrame(vs_rows)
        df_vs = (
            df_vs.groupby(["team", "opponent", "map"], as_index=False)
            .sum(numeric_only=True)
        )
        df_vs["win_rate"] = df_vs.apply(
            lambda r: (r["wins"] / r["played"]) if r["played"] else 0.0, axis=1
        )
    else:
        df_vs = pd.DataFrame(columns=["team", "opponent", "map", "played", "wins", "losses", "win_rate"])

    # Save outputs
    os.makedirs("data/processed", exist_ok=True)
    # Enrich with map metadata from valorant-api.com
    df_stats_enriched = enrich_map_metadata(df_stats)
    df_stats_enriched.to_csv("data/processed/team_map_stats.csv", index=False, encoding="utf-8")
    df_vs.to_csv("data/processed/team_map_vs_opponent.csv", index=False, encoding="utf-8")

    print("✓ Saved team_map_stats.csv and team_map_vs_opponent.csv in data/processed/")
    return df_stats, df_vs


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Collect map-level stats and aggregate")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of match ids to fetch")
    args = parser.parse_args()
    collect_map_stats(limit=args.limit)


if __name__ == "__main__":
    main()


