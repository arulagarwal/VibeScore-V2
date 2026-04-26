import csv
import os
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Dict

from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

_HERE = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_CSV = os.path.join(_HERE, '..', 'data', 'songs.csv')


def load_songs(csv_path: str) -> List[Dict]:
    with open(csv_path, newline='', encoding='utf-8') as f:
        return list(csv.DictReader(f))


# ── GuardrailResult ───────────────────────────────────────────────────────────

@dataclass
class GuardrailResult:
    is_clean: bool
    flagged_titles: List[str]
    safe_response: str


# ── SongKnowledgeBase ─────────────────────────────────────────────────────────

class SongKnowledgeBase:
    """RAG data layer: embeds the song catalog into Chroma and handles vector retrieval."""

    def __init__(self, api_key: str, csv_path: str = _DEFAULT_CSV):
        self._songs: List[Dict] = load_songs(csv_path)
        self.valid_titles: set = {s['title'].lower() for s in self._songs}
        self._embeddings = GoogleGenerativeAIEmbeddings(
            model="models/text-embedding-004",
            google_api_key=api_key,
        )
        self._vectorstore = self.ingest_catalog()

    def ingest_catalog(self) -> Chroma:
        """Convert CSV rows into text documents and load them into a Chroma vector store."""
        documents = []
        for s in self._songs:
            text = (
                f"Title: {s['title']} | Artist: {s['artist']} | "
                f"Genre: {s['genre']} | Mood: {s['mood']} | "
                f"Energy: {s['energy']} | Valence: {s.get('valence', 'N/A')} | "
                f"Danceability: {s.get('danceability', 'N/A')} | "
                f"Acousticness: {s.get('acousticness', 'N/A')} | "
                f"Tempo: {s.get('tempo_bpm', 'N/A')} BPM | "
                f"Tags: {s.get('mood_tags', '')}"
            )
            documents.append(Document(page_content=text, metadata={"title": s['title']}))
        return Chroma.from_documents(documents, self._embeddings)

    def retrieve(self, query: str, k: int, diversity_penalty: bool = False) -> List[Dict]:
        """Return the top-k songs most semantically relevant to the query.

        When diversity_penalty=True, applies a greedy artist-diversity re-ranking
        (mirrors the _select_with_diversity logic from the original VibeScore 1.0).
        """
        title_to_song = {s['title'].lower(): s for s in self._songs}

        if not diversity_penalty:
            docs = self._vectorstore.similarity_search(query, k=k)
            return [
                title_to_song[doc.metadata['title'].lower()]
                for doc in docs
                if doc.metadata['title'].lower() in title_to_song
            ]

        # Fetch a wider pool so the diversity loop has enough candidates to choose from.
        fetch_k = min(k * 3, len(self._songs))
        docs_with_scores = self._vectorstore.similarity_search_with_relevance_scores(query, k=fetch_k)

        candidates = [
            (title_to_song[doc.metadata['title'].lower()], score)
            for doc, score in docs_with_scores
            if doc.metadata['title'].lower() in title_to_song
        ]

        # Greedy re-ranking: at each step subtract the penalty from every candidate
        # whose artist already appears in the selected set, then re-sort.
        _DIVERSITY_PENALTY = 1.0
        selected: List[Dict] = []
        seen_artists: set = set()
        remaining = list(candidates)

        while remaining and len(selected) < k:
            adjusted = [
                (song, score - (_DIVERSITY_PENALTY if song.get('artist') in seen_artists else 0.0))
                for song, score in remaining
            ]
            adjusted.sort(key=lambda x: x[1], reverse=True)
            best_song, _ = adjusted[0]
            selected.append(best_song)
            seen_artists.add(best_song.get('artist'))
            best_title = best_song.get('title')
            remaining = [(s, sc) for s, sc in remaining if s.get('title') != best_title]

        return selected

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
    """Abstract base: each mode configures system prompt, retrieval depth, and diversity."""

    @property
    @abstractmethod
    def name(self) -> str: ...

    @property
    @abstractmethod
    def system_prompt(self) -> str: ...

    @property
    @abstractmethod
    def retrieval_k(self) -> int: ...

    @property
    def diversity_penalty(self) -> bool:
        return False


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
    def diversity_penalty(self) -> bool:
        return True

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
    Central orchestrator: retrieves grounded context, calls Gemini,
    and validates the response through the hallucination guardrail.
    """

    def __init__(
        self,
        api_key: str,
        knowledge_base: SongKnowledgeBase,
        mode: ScoringModeConfig,
    ):
        self._llm = ChatGoogleGenerativeAI(
            model="gemini-3.0-flash",
            google_api_key=api_key,
        )
        self.knowledge_base = knowledge_base
        self.mode = mode
        self._guardrail = HallucinationGuardrail()

    def chat(
        self,
        user_message: str,
        history: List[Dict],
        prefetched_songs: List[Dict] = None,
    ) -> tuple:
        """
        RAG pipeline. Returns (response_text, is_clean).

        If prefetched_songs is provided, the retrieval step is skipped and the
        provided songs are used as the catalog context — this lets the caller
        (e.g. the Streamlit UI) run retrieval inside an observable status block
        and pass the results in.
        """
        retrieved = (
            prefetched_songs
            if prefetched_songs is not None
            else self.knowledge_base.retrieve(
                user_message,
                k=self.mode.retrieval_k,
                diversity_penalty=self.mode.diversity_penalty,
            )
        )
        catalog_context = self._format_catalog(retrieved)
        full_system = f"{self.mode.system_prompt}\n\nCATALOG CONTEXT:\n{catalog_context}"

        messages = [SystemMessage(content=full_system)]
        for msg in history:
            if msg["role"] == "user":
                messages.append(HumanMessage(content=msg["content"]))
            elif msg["role"] == "assistant":
                messages.append(AIMessage(content=msg["content"]))
        messages.append(HumanMessage(content=user_message))

        response = self._llm.invoke(messages)
        result = self._guardrail.validate(response.content, self.knowledge_base)
        return result.safe_response, result.is_clean

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
