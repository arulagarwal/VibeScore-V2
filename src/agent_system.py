import os
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Dict

import anthropic

try:
    from recommender import load_songs
except ImportError:
    from src.recommender import load_songs

_HERE = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_CSV = os.path.join(_HERE, '..', 'data', 'songs.csv')


# ── GuardrailResult ───────────────────────────────────────────────────────────

@dataclass
class GuardrailResult:
    is_clean: bool
    flagged_titles: List[str]
    safe_response: str


# ── SongKnowledgeBase ─────────────────────────────────────────────────────────

class SongKnowledgeBase:
    """RAG data layer: loads the song catalog and handles retrieval."""

    def __init__(self, csv_path: str = _DEFAULT_CSV):
        self._songs: List[Dict] = load_songs(csv_path)
        # Lowercase set used by the guardrail as ground truth
        self.valid_titles: set = {s['title'].lower() for s in self._songs}

    def retrieve(self, query: str, k: int) -> List[Dict]:
        """Return the top-k songs most relevant to a natural-language query."""
        query_lower = query.lower()
        scored = []
        for song in self._songs:
            score = 0
            if song.get('genre', '').lower() in query_lower:
                score += 3
            if song.get('mood', '').lower() in query_lower:
                score += 2
            for tag in str(song.get('mood_tags', '')).lower().split(','):
                if tag.strip() and tag.strip() in query_lower:
                    score += 1
            energy = float(song.get('energy', 0.5))
            if energy >= 0.75 and any(w in query_lower for w in ('energetic', 'high energy', 'upbeat', 'intense', 'pump', 'hype')):
                score += 1
            if energy <= 0.4 and any(w in query_lower for w in ('calm', 'chill', 'relax', 'low energy', 'soft', 'quiet', 'peaceful')):
                score += 1
            scored.append((song, score))
        scored.sort(key=lambda x: x[1], reverse=True)
        return [s for s, _ in scored[:k]]

    @property
    def all_songs(self) -> List[Dict]:
        return self._songs


# ── HallucinationGuardrail ────────────────────────────────────────────────────

class HallucinationGuardrail:
    """Validates agent responses to ensure only catalog songs are referenced."""

    _PATTERNS = [
        re.compile(r'"([^"]{3,60})"'),        # "Title"
        re.compile(r'\*\*([^*]{3,60})\*\*'),  # **Title**
        re.compile(r"'([^']{3,60})'"),         # 'Title'
    ]

    def validate(self, response: str, knowledge_base: SongKnowledgeBase) -> GuardrailResult:
        mentions = []
        for pattern in self._PATTERNS:
            mentions.extend(pattern.findall(response))

        # Filter to plausible song title lengths; ignore things already in the catalog
        flagged = [
            m for m in mentions
            if m.lower() not in knowledge_base.valid_titles and len(m.split()) <= 8
        ]

        if not flagged:
            return GuardrailResult(is_clean=True, flagged_titles=[], safe_response=response)

        note = (
            "\n\n*Note: Some referenced titles could not be verified against the catalog. "
            "Recommendations above are grounded in the available song data.*"
        )
        return GuardrailResult(
            is_clean=False,
            flagged_titles=flagged,
            safe_response=response + note,
        )


# ── Strategy Pattern ──────────────────────────────────────────────────────────

class ScoringModeConfig(ABC):
    """Abstract base: each mode configures system prompt and retrieval depth."""

    @property
    @abstractmethod
    def name(self) -> str: ...

    @property
    @abstractmethod
    def system_prompt(self) -> str: ...

    @property
    @abstractmethod
    def retrieval_k(self) -> int: ...


class BalancedMode(ScoringModeConfig):
    """Concise recommendations balancing genre, mood, and energy."""

    @property
    def name(self) -> str:
        return "Balanced"

    @property
    def retrieval_k(self) -> int:
        return 5

    @property
    def system_prompt(self) -> str:
        return (
            "You are VibeScore, a friendly music recommendation assistant. "
            "You help users discover songs from a curated catalog.\n\n"
            "RULES:\n"
            "- ONLY recommend songs explicitly listed in the CATALOG CONTEXT provided.\n"
            "- Never invent or suggest songs, artists, or albums not in the catalog.\n"
            "- Balance genre, mood, and energy when selecting recommendations.\n"
            "- Keep explanations concise: 1-2 sentences per song.\n"
            "- Format each recommendation as: **Song Title** by Artist — brief reason."
        )


class DeepDiveMode(ScoringModeConfig):
    """Rich analytical recommendations using all song attributes."""

    @property
    def name(self) -> str:
        return "Deep Dive"

    @property
    def retrieval_k(self) -> int:
        return 10

    @property
    def system_prompt(self) -> str:
        return (
            "You are VibeScore, an analytical music recommendation expert. "
            "You help users discover songs from a curated catalog with deep musical insight.\n\n"
            "RULES:\n"
            "- ONLY recommend songs explicitly listed in the CATALOG CONTEXT provided.\n"
            "- Never invent or suggest songs, artists, or albums not in the catalog.\n"
            "- For each song, discuss its valence (musical positivity), danceability, "
            "acousticness, energy level, and tempo where relevant.\n"
            "- Compare songs to each other and explain the emotional arc of your playlist.\n"
            "- Format each recommendation as: **Song Title** by Artist — detailed analysis."
        )


# ── VibeScoreAgent ────────────────────────────────────────────────────────────

class VibeScoreAgent:
    """
    Central orchestrator: retrieves grounded context, calls Claude,
    and validates the response through the hallucination guardrail.
    """

    def __init__(
        self,
        api_key: str,
        knowledge_base: SongKnowledgeBase,
        mode: ScoringModeConfig,
    ):
        self._client = anthropic.Anthropic(api_key=api_key)
        self.knowledge_base = knowledge_base
        self.mode = mode
        self._guardrail = HallucinationGuardrail()

    def chat(self, user_message: str, history: List[Dict]) -> str:
        """
        Full RAG pipeline:
        1. Retrieve top-k songs relevant to the user message
        2. Build augmented system prompt with catalog context
        3. Call Claude API (with prompt caching on the static system prompt)
        4. Validate response through hallucination guardrail
        5. Return safe response
        """
        retrieved = self.knowledge_base.retrieve(user_message, k=self.mode.retrieval_k)
        catalog_context = self._format_catalog(retrieved)

        # Static system prompt is cached; dynamic catalog context is not
        system = [
            {
                "type": "text",
                "text": self.mode.system_prompt,
                "cache_control": {"type": "ephemeral"},
            },
            {
                "type": "text",
                "text": f"CATALOG CONTEXT:\n{catalog_context}",
            },
        ]

        messages = list(history) + [{"role": "user", "content": user_message}]

        response = self._client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=1024,
            system=system,
            messages=messages,
        )
        raw_text = response.content[0].text
        result = self._guardrail.validate(raw_text, self.knowledge_base)
        return result.safe_response

    def _format_catalog(self, songs: List[Dict]) -> str:
        lines = []
        for s in songs:
            lines.append(
                f"- \"{s['title']}\" by {s['artist']} | "
                f"Genre: {s['genre']} | Mood: {s['mood']} | "
                f"Energy: {s['energy']} | Valence: {s.get('valence', 'N/A')} | "
                f"Danceability: {s.get('danceability', 'N/A')} | "
                f"Acousticness: {s.get('acousticness', 'N/A')} | "
                f"Tempo: {s.get('tempo_bpm', 'N/A')} BPM | "
                f"Tags: {s.get('mood_tags', '')}"
            )
        return '\n'.join(lines)
