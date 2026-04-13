"""
Command line runner for the Music Recommender Simulation.

Demonstrates the Strategy pattern by running one user profile through
three scoring modes and printing an ASCII table for each.
"""

import os
from tabulate import tabulate

try:
    from recommender import load_songs, recommend_songs, BalancedMode, GenreFirstMode, MoodFirstMode
except ModuleNotFoundError:
    from src.recommender import load_songs, recommend_songs, BalancedMode, GenreFirstMode, MoodFirstMode

# ---------------------------------------------------------------------------
# Demo profile — "Chill Lofi" triggers the diversity penalty:
#   LoRoom has two songs in the catalog (Midnight Coding, Focus Flow).
#   Both rank inside the top-5 by raw score, so Focus Flow earns the
#   repeat-artist penalty once Midnight Coding is already selected.
#
#   Balanced  : penalty visible in reasons; Focus Flow holds #3 (margin 0.02)
#   GenreFirst: penalty visible but large genre bonus keeps Focus Flow at #3
#   MoodFirst : mood weight (+3) lifts Spacewalk Thoughts above the penalised
#               Focus Flow — the penalty actually reshuffles the ranking here
# ---------------------------------------------------------------------------
DEMO_PROFILE = {
    "name": "Chill Lofi",
    "prefs": {"genre": "lofi", "mood": "chill", "energy": 0.35},
}

MODES = [BalancedMode(), GenreFirstMode(), MoodFirstMode()]

K = 5  # recommendations per mode


def run_mode(profile: dict, songs: list, strategy) -> None:
    prefs = profile["prefs"]
    results = recommend_songs(prefs, songs, k=K, strategy=strategy)

    # Header block
    profile_line = (
        f"  Profile : {profile['name']}  |  "
        f"genre={prefs['genre']}  mood={prefs['mood']}  energy={prefs['energy']}"
    )
    mode_line = f"  Scoring : {strategy.NAME}"
    width = max(len(profile_line), len(mode_line))
    border = "=" * width

    rows = []
    for rank, (song, score, explanation) in enumerate(results, start=1):
        rows.append([rank, song["title"], f"{score:.2f}", explanation])

    table = tabulate(
        rows,
        headers=["#", "Song Title", "Score", "Reasons"],
        tablefmt="simple_outline",
        colalign=("center", "left", "center", "left"),
    )

    print(f"\n{border}")
    print(profile_line)
    print(mode_line)
    print(border)
    print(table)


def main() -> None:
    data_path = os.path.join(os.path.dirname(__file__), "..", "data", "songs.csv")
    songs = load_songs(data_path)

    for mode in MODES:
        run_mode(DEMO_PROFILE, songs, mode)

    print()


if __name__ == "__main__":
    main()
