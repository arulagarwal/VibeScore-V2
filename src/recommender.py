import csv
from abc import ABC, abstractmethod
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass


@dataclass
class Song:
    """
    Represents a song and its attributes.
    Required by tests/test_recommender.py
    """
    id: int
    title: str
    artist: str
    genre: str
    mood: str
    energy: float
    tempo_bpm: float
    valence: float
    danceability: float
    acousticness: float


@dataclass
class UserProfile:
    """
    Represents a user's taste preferences.
    Required by tests/test_recommender.py
    """
    favorite_genre: str
    favorite_mood: str
    target_energy: float
    likes_acoustic: bool


# ---------------------------------------------------------------------------
# Strategy pattern: interchangeable scoring modes
# ---------------------------------------------------------------------------

class ScoringStrategy(ABC):
    """Abstract base for all scoring modes."""
    NAME: str = "Base Strategy"

    @abstractmethod
    def score(self, user_prefs: Dict, song: Dict) -> Tuple[float, List[str]]:
        """Score a song against user preferences.

        Returns:
            (score, reasons) where reasons is a list of human-readable strings.
        """


class BalancedMode(ScoringStrategy):
    """
    Balanced Mode — equal weight on genre, mood, and energy.

      +2.0  genre match
      +1.0  mood match
      +0–1  energy similarity: 1.0 - abs(song_energy - target_energy)
    """
    NAME = "Balanced Mode"

    def score(self, user_prefs: Dict, song: Dict) -> Tuple[float, List[str]]:
        score = 0.0
        reasons = []

        if song.get('genre') == user_prefs.get('genre'):
            score += 2.0
            reasons.append('genre match (+2.0)')

        if song.get('mood') == user_prefs.get('mood'):
            score += 1.0
            reasons.append('mood match (+1.0)')

        target_energy = float(user_prefs.get('energy', 0.5))
        energy_sim = 1.0 - abs(float(song.get('energy', 0.5)) - target_energy)
        score += energy_sim
        reasons.append(f'energy similarity ({energy_sim:+.2f})')

        return score, reasons


class GenreFirstMode(ScoringStrategy):
    """
    Genre-First Mode — heavily prioritises genre alignment.

      +4.0  genre match
      +1.0  mood match
      +0–1  energy similarity: 1.0 - abs(song_energy - target_energy)
    """
    NAME = "Genre-First Mode"

    def score(self, user_prefs: Dict, song: Dict) -> Tuple[float, List[str]]:
        score = 0.0
        reasons = []

        if song.get('genre') == user_prefs.get('genre'):
            score += 4.0
            reasons.append('genre match (+4.0)')

        if song.get('mood') == user_prefs.get('mood'):
            score += 1.0
            reasons.append('mood match (+1.0)')

        target_energy = float(user_prefs.get('energy', 0.5))
        energy_sim = 1.0 - abs(float(song.get('energy', 0.5)) - target_energy)
        score += energy_sim
        reasons.append(f'energy similarity ({energy_sim:+.2f})')

        return score, reasons


class MoodFirstMode(ScoringStrategy):
    """
    Mood-First Mode — heavily prioritises emotional match.

      +2.0  genre match
      +3.0  mood match
      +0–1  energy similarity: 1.0 - abs(song_energy - target_energy)
    """
    NAME = "Mood-First Mode"

    def score(self, user_prefs: Dict, song: Dict) -> Tuple[float, List[str]]:
        score = 0.0
        reasons = []

        if song.get('genre') == user_prefs.get('genre'):
            score += 2.0
            reasons.append('genre match (+2.0)')

        if song.get('mood') == user_prefs.get('mood'):
            score += 3.0
            reasons.append('mood match (+3.0)')

        target_energy = float(user_prefs.get('energy', 0.5))
        energy_sim = 1.0 - abs(float(song.get('energy', 0.5)) - target_energy)
        score += energy_sim
        reasons.append(f'energy similarity ({energy_sim:+.2f})')

        return score, reasons


# ---------------------------------------------------------------------------
# Diversity penalty
# ---------------------------------------------------------------------------

DIVERSITY_PENALTY = -1.0


def _select_with_diversity(
    scored: List[Tuple[Dict, float, List[str]]],
    k: int,
) -> List[Tuple[Dict, float, str]]:
    """
    Greedy re-ranking loop that enforces artist diversity.

    At each step the penalty is re-applied to every remaining candidate
    whose artist is already in the selected set, then candidates are
    re-sorted. This means the penalty can actually change the final order,
    not just annotate it.

    Args:
        scored: All songs pre-scored as (song_dict, base_score, base_reasons).
        k:      Number of results to return.

    Returns:
        List of (song_dict, adjusted_score, explanation_string) of length <= k.
    """
    selected: List[Tuple[Dict, float, str]] = []
    seen_artists: set = set()
    remaining = list(scored)           # [(song, base_score, base_reasons)]

    while remaining and len(selected) < k:
        # Build this round's candidates with penalties applied where needed
        candidates = []
        for song, base_score, base_reasons in remaining:
            adj_score = base_score
            adj_reasons = list(base_reasons)
            if song.get('artist') in seen_artists:
                adj_score += DIVERSITY_PENALTY
                adj_reasons.append(
                    f'repeat artist penalty ({DIVERSITY_PENALTY:+.1f})'
                )
            candidates.append((song, adj_score, adj_reasons))

        # Pick the highest scorer after adjustments
        candidates.sort(key=lambda x: x[1], reverse=True)
        best_song, best_score, best_reasons = candidates[0]

        selected.append((best_song, best_score, '; '.join(best_reasons)))
        seen_artists.add(best_song.get('artist'))

        # Drop the chosen song from remaining (match by id)
        best_id = best_song.get('id')
        remaining = [(s, sc, r) for s, sc, r in remaining if s.get('id') != best_id]

    return selected


# ---------------------------------------------------------------------------
# OOP interface (used by tests)
# ---------------------------------------------------------------------------

class Recommender:
    """
    OOP implementation of the recommendation logic.
    Required by tests/test_recommender.py
    """
    def __init__(self, songs: List[Song], strategy: ScoringStrategy = None):
        self.songs = songs
        self.strategy = strategy or BalancedMode()

    def recommend(self, user: UserProfile, k: int = 5) -> List[Song]:
        user_prefs = {
            'genre': user.favorite_genre,
            'mood': user.favorite_mood,
            'energy': user.target_energy,
        }
        # Convert Song objects to minimal dicts so we can reuse the shared pipeline
        song_dicts = [
            {'id': s.id, 'artist': s.artist, 'genre': s.genre,
             'mood': s.mood, 'energy': s.energy}
            for s in self.songs
        ]
        song_by_id = {s.id: s for s in self.songs}

        results = recommend_songs(user_prefs, song_dicts, k=k, strategy=self.strategy)
        return [song_by_id[song_dict['id']] for song_dict, _, _ in results]

    def explain_recommendation(self, user: UserProfile, song: Song) -> str:
        user_prefs = {
            'genre': user.favorite_genre,
            'mood': user.favorite_mood,
            'energy': user.target_energy,
        }
        _, reasons = self.strategy.score(user_prefs, {
            'genre': song.genre,
            'mood': song.mood,
            'energy': song.energy,
        })
        return '; '.join(reasons) if reasons else 'No matching features'


# ---------------------------------------------------------------------------
# Functional interface (used by main.py)
# ---------------------------------------------------------------------------

def load_songs(csv_path: str) -> List[Dict]:
    """
    Loads songs from a CSV file.
    Required by src/main.py
    """
    float_fields = {'energy', 'tempo_bpm', 'valence', 'danceability', 'acousticness', 'popularity'}
    songs = []
    with open(csv_path, newline='') as f:
        for row in csv.DictReader(f):
            for field in float_fields:
                if field in row and row[field] != '':
                    row[field] = float(row[field])
            if 'id' in row:
                row['id'] = int(row['id'])
            songs.append(dict(row))
    return songs


def score_song(
    user_prefs: Dict,
    song: Dict,
    strategy: ScoringStrategy = None,
) -> Tuple[float, List[str]]:
    """
    Scores a single song against user preferences using the given strategy.
    Defaults to BalancedMode when no strategy is provided.

    Returns:
        (score, reasons) where reasons is a list of human-readable strings.
    """
    return (strategy or BalancedMode()).score(user_prefs, song)


def recommend_songs(
    user_prefs: Dict,
    songs: List[Dict],
    k: int = 5,
    strategy: ScoringStrategy = None,
) -> List[Tuple[Dict, float, str]]:
    """
    Scores every song, then runs the greedy diversity re-ranking loop
    to select the final top-k results.

    Returns:
        List of (song_dict, adjusted_score, explanation) tuples,
        sorted by adjusted score descending, with at most k entries.
    """
    active = strategy or BalancedMode()

    # Score every song with the chosen strategy
    all_scored = []
    for song in songs:
        base_score, base_reasons = active.score(user_prefs, song)
        all_scored.append((song, base_score, base_reasons))

    # Initial sort by raw score (first iteration of the greedy loop starts here)
    all_scored.sort(key=lambda x: x[1], reverse=True)

    return _select_with_diversity(all_scored, k)
