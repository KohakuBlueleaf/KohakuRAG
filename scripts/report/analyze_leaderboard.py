#!/usr/bin/env python3
"""Analyze leaderboard data for report.

Usage:
    python scripts/report/analyze_leaderboard.py
    python scripts/report/analyze_leaderboard.py --latex  # Output LaTeX table
"""

import argparse
import json
from pathlib import Path

LEADERBOARD_PATH = Path("leaderboard.json")

# Our team ID (KohakuRAG)
OUR_TEAM_ID = 14730116


def load_leaderboard() -> dict:
    """Load leaderboard JSON."""
    with open(LEADERBOARD_PATH) as f:
        return json.load(f)


def build_team_scores(data: dict) -> dict:
    """Build mapping of teamId -> {public_rank, public_score, private_rank, private_score}."""
    teams = {}

    # Process public leaderboard
    for entry in data["publicLeaderboard"]:
        team_id = entry["teamId"]
        if team_id not in teams:
            teams[team_id] = {}
        teams[team_id]["public_rank"] = entry["rank"]
        teams[team_id]["public_score"] = float(entry["displayScore"])

    # Process private leaderboard
    for entry in data["privateLeaderboard"]:
        team_id = entry["teamId"]
        if team_id not in teams:
            teams[team_id] = {}
        teams[team_id]["private_rank"] = entry["rank"]
        teams[team_id]["private_score"] = float(entry["displayScore"])

    return teams


def analyze_rank_changes(teams: dict) -> None:
    """Analyze rank changes between public and private leaderboards."""
    print("=" * 70)
    print("LEADERBOARD ANALYSIS")
    print("=" * 70)

    # Find teams with both public and private scores
    complete_teams = {
        tid: t for tid, t in teams.items() if "public_rank" in t and "private_rank" in t
    }

    # Our team
    our_team = teams.get(OUR_TEAM_ID, {})
    print(f"\nOUR TEAM (KohakuRAG, ID={OUR_TEAM_ID}):")
    print(
        f"  Public:  Rank #{our_team.get('public_rank', 'N/A')}, Score {our_team.get('public_score', 'N/A')}"
    )
    print(
        f"  Private: Rank #{our_team.get('private_rank', 'N/A')}, Score {our_team.get('private_score', 'N/A')}"
    )
    if "public_score" in our_team and "private_score" in our_team:
        delta = our_team["private_score"] - our_team["public_score"]
        print(f"  Delta:   {delta:+.3f}")

    # Top 10 in private leaderboard
    print("\n" + "-" * 70)
    print("TOP 10 PRIVATE LEADERBOARD (Final Rankings):")
    print("-" * 70)
    print(
        f"{'Rank':<6} {'TeamID':<12} {'Public':<10} {'Private':<10} {'Pub Rank':<10} {'Delta':<10}"
    )

    private_sorted = sorted(
        complete_teams.items(), key=lambda x: x[1].get("private_rank", 999)
    )[:10]

    for team_id, t in private_sorted:
        pub_score = t.get("public_score", 0)
        priv_score = t.get("private_score", 0)
        pub_rank = t.get("public_rank", "N/A")
        priv_rank = t.get("private_rank", "N/A")
        delta = priv_score - pub_score
        marker = " <-- OURS" if team_id == OUR_TEAM_ID else ""
        print(
            f"#{priv_rank:<5} {team_id:<12} {pub_score:<10.3f} {priv_score:<10.3f} #{pub_rank:<9} {delta:+.3f}{marker}"
        )

    # Top 10 in public leaderboard and their private performance
    print("\n" + "-" * 70)
    print("TOP 10 PUBLIC LEADERBOARD -> PRIVATE PERFORMANCE:")
    print("-" * 70)
    print(
        f"{'Pub Rank':<10} {'TeamID':<12} {'Public':<10} {'Private':<10} {'Priv Rank':<10} {'Delta':<10}"
    )

    public_sorted = sorted(
        complete_teams.items(), key=lambda x: x[1].get("public_rank", 999)
    )[:10]

    for team_id, t in public_sorted:
        pub_score = t.get("public_score", 0)
        priv_score = t.get("private_score", 0)
        pub_rank = t.get("public_rank", "N/A")
        priv_rank = t.get("private_rank", "N/A")
        delta = priv_score - pub_score
        marker = " <-- OURS" if team_id == OUR_TEAM_ID else ""
        print(
            f"#{pub_rank:<9} {team_id:<12} {pub_score:<10.3f} {priv_score:<10.3f} #{priv_rank:<9} {delta:+.3f}{marker}"
        )

    # Biggest gainers (improved most from public to private)
    print("\n" + "-" * 70)
    print("BIGGEST RANK GAINERS (Public -> Private):")
    print("-" * 70)

    gainers = [
        (tid, t, t.get("public_rank", 999) - t.get("private_rank", 999))
        for tid, t in complete_teams.items()
        if t.get("private_rank", 999) <= 20  # Only top 20 private
    ]
    gainers.sort(key=lambda x: -x[2])  # Sort by rank improvement

    for team_id, t, improvement in gainers[:5]:
        pub_rank = t.get("public_rank", "N/A")
        priv_rank = t.get("private_rank", "N/A")
        pub_score = t.get("public_score", 0)
        priv_score = t.get("private_score", 0)
        print(
            f"  #{pub_rank} -> #{priv_rank} (+{improvement} ranks): {pub_score:.3f} -> {priv_score:.3f}"
        )

    # Biggest losers (dropped most from public to private)
    print("\n" + "-" * 70)
    print("BIGGEST RANK DROPS (Public -> Private):")
    print("-" * 70)

    losers = [
        (tid, t, t.get("private_rank", 999) - t.get("public_rank", 999))
        for tid, t in complete_teams.items()
        if t.get("public_rank", 999) <= 20  # Only top 20 public
    ]
    losers.sort(key=lambda x: -x[2])  # Sort by rank drop

    for team_id, t, drop in losers[:5]:
        pub_rank = t.get("public_rank", "N/A")
        priv_rank = t.get("private_rank", "N/A")
        pub_score = t.get("public_score", 0)
        priv_score = t.get("private_score", 0)
        print(
            f"  #{pub_rank} -> #{priv_rank} (-{drop} ranks): {pub_score:.3f} -> {priv_score:.3f}"
        )


def generate_latex_table(teams: dict) -> str:
    """Generate LaTeX table for report."""
    lines = []
    lines.append("\\begin{table}[t]")
    lines.append("\\centering")
    lines.append(
        "\\caption{Final leaderboard comparison showing public vs.\\ private score dynamics. Only KohakuRAG maintained the top position across both partitions.}"
    )
    lines.append("\\label{tab:leaderboard_comparison}")
    lines.append("\\small")
    lines.append("\\begin{tabular}{lccc}")
    lines.append("\\toprule")
    lines.append(
        "\\textbf{Team} & \\textbf{Public} & \\textbf{Private} & \\textbf{$\\Delta$} \\\\"
    )
    lines.append("\\midrule")

    # Get complete teams
    complete_teams = {
        tid: t for tid, t in teams.items() if "public_rank" in t and "private_rank" in t
    }

    # Our team first
    our = teams[OUR_TEAM_ID]
    delta = our["private_score"] - our["public_score"]
    lines.append(
        f"KohakuRAG (Ours) & {our['public_score']:.3f} (\\#1) & {our['private_score']:.3f} (\\#1) & ${delta:+.3f}$ \\\\"
    )

    # Private #2 and #3
    private_sorted = sorted(
        complete_teams.items(), key=lambda x: x[1].get("private_rank", 999)
    )

    for team_id, t in private_sorted[1:3]:  # #2 and #3
        pub_score = t["public_score"]
        priv_score = t["private_score"]
        pub_rank = t["public_rank"]
        priv_rank = t["private_rank"]
        delta = priv_score - pub_score
        lines.append(
            f"Private \\#{priv_rank} & {pub_score:.3f} (\\#{pub_rank}) & {priv_score:.3f} (\\#{priv_rank}) & ${delta:+.3f}$ \\\\"
        )

    lines.append("\\midrule")

    # Public #2 and #3 (excluding us)
    public_sorted = sorted(
        [(tid, t) for tid, t in complete_teams.items() if tid != OUR_TEAM_ID],
        key=lambda x: x[1].get("public_rank", 999),
    )

    for team_id, t in public_sorted[:2]:  # Top 2 excluding us
        pub_score = t["public_score"]
        priv_score = t["private_score"]
        pub_rank = t["public_rank"]
        priv_rank = t["private_rank"]
        delta = priv_score - pub_score
        lines.append(
            f"Public \\#{pub_rank} & {pub_score:.3f} (\\#{pub_rank}) & {priv_score:.3f} (\\#{priv_rank}) & ${delta:+.3f}$ \\\\"
        )

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Analyze leaderboard data")
    parser.add_argument("--latex", action="store_true", help="Output LaTeX table")
    args = parser.parse_args()

    data = load_leaderboard()
    teams = build_team_scores(data)

    if args.latex:
        print(generate_latex_table(teams))
    else:
        analyze_rank_changes(teams)


if __name__ == "__main__":
    main()
