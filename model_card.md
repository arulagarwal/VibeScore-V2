# Model Card: VibeScore 2.0

## 1. Model Identity

**Name:** VibeScore 2.0
**Base project:** VibeScore 1.0 (Module 3) — rule-based, CLI, math-only scoring
**This version:** Agentic RAG system — Gemini 3.0 Flash + ChromaDB + Streamlit chat UI
**Author:** Arul Agarwal

---

## 2. Intended Use

VibeScore 2.0 is a classroom demonstration of an agentic Retrieval-Augmented Generation (RAG) pipeline applied to music recommendation. A user describes what they want to hear in natural language — a "vibe" — and the system retrieves semantically matched songs from a 20-song catalog, passes them as grounded context to a Gemini LLM, and returns a curated playlist with explanations.

This system is designed for educational use. It is not intended to serve production traffic, handle real user data, or replace a full-scale recommendation platform.

---

## 3. From Math to Meaning — The Core Upgrade

### How VibeScore 1.0 Worked

VibeScore 1.0 (Module 3) scored songs using explicit arithmetic against a hard-coded user profile:

```
score = genre_match_bonus + mood_match_bonus + energy_similarity
```

The weights differed by mode (Balanced, Genre-First, Mood-First), but the underlying mechanism was always exact string equality. If a user wanted something "melancholic and cinematic," and no song carried exactly the `mood=melancholic` tag, the system awarded zero mood points regardless of how close the actual songs were.

### How VibeScore 2.0 Works

VibeScore 2.0 replaces the scoring loop with a two-stage pipeline:

**Stage 1 — Semantic Retrieval (ChromaDB)**
At startup, every song in `songs.csv` is serialized into a text document containing all 12 attributes and embedded using Google's `text-embedding-004` model. When the user sends a message, that query is embedded in the same vector space and a cosine similarity search returns the top-k most relevant songs. "Melancholic and cinematic" will retrieve *Neon Noir* (darkwave, moody, haunting tags) because the embeddings capture semantic proximity — not because a tag matched exactly.

**Stage 2 — LLM Reasoning (Gemini 3.0 Flash)**
The retrieved songs are injected as a `CATALOG CONTEXT` block into the system prompt. Gemini reads the real song data — valence, danceability, acousticness, tempo, mood tags — and reasons about them in natural language, explaining the emotional arc of the playlist rather than printing a score table.

### Trade-offs

| Property | VibeScore 1.0 | VibeScore 2.0 |
|---|---|---|
| Transparency | Full — every point explained | Partial — LLM reasoning is opaque |
| Semantic understanding | None — exact string match only | Strong — vector space captures synonyms and adjacent concepts |
| Determinism | Fully deterministic | Non-deterministic — same query may return different phrasing |
| Hallucination risk | Zero — math cannot invent songs | Present — mitigated by guardrail, never fully eliminated |
| Cost | Free — pure Python arithmetic | API cost per request (Gemini + embedding calls) |
| Latency | Milliseconds | Seconds (embedding + LLM inference) |
| Catalog size scalability | Linear scan, fine at 20 songs | Vector search, scales to millions |

The move from math to meaning introduces genuine capability — semantic understanding, natural language interaction, multi-turn memory — but it also introduces genuine risk. A math formula is auditable line by line. A language model's internal reasoning is not. That is why the guardrail exists.

---

## 4. Responsible Design

### The Diversity Penalty (Ported from VibeScore 1.0)

The original VibeScore 1.0 included a greedy artist-diversity re-ranker that applied a `-1.0` point penalty to any song by an artist already present in the selected set. This prevented the recommendation engine from filling a playlist with one artist who perfectly matched the profile.

VibeScore 2.0 ports this logic into `SongKnowledgeBase.retrieve()` and wires it through the Strategy pattern as a per-mode option. When `diversity_penalty=True` (active in Deep Dive mode), the retrieval layer:

1. Fetches `k * 3` candidates from ChromaDB to create a larger pool.
2. Converts distances to relevance scores.
3. Runs the same greedy loop: at each step, subtract 1.0 from the relevance score of any candidate whose artist is already selected, re-sort, pick the new best.
4. Returns the final `k` songs with no repeated artists.

This matters because semantic similarity naturally clusters. Ask for "chill lofi study music" and the embeddings will rank LoRoom's two tracks (*Midnight Coding*, *Focus Flow*) very close to each other. Without the penalty, both could surface in the top results — a filter bubble caused by the catalog structure, not the user's preference. The penalty forces the system to present breadth.

### The HallucinationGuardrail

Language models are fluent and confident. Left unchecked, Gemini may recommend a song that sounds plausible but does not exist in the catalog. The `HallucinationGuardrail` class is the system's mechanism for catching this before it reaches the user.

**How it works:**

1. After Gemini generates a response, three regex patterns extract every quoted string, bold-formatted string, and single-quoted string from the text — the formats the system prompt instructs Gemini to use for song titles.
2. Each extracted string is looked up in `valid_titles`, a lowercase set built directly from `songs.csv` at initialization time. This set is the single source of truth.
3. Any extracted string that (a) is not in `valid_titles` and (b) is ≤ 8 words long — the heuristic for "plausible song title rather than sentence fragment" — is flagged.
4. If flags exist, the original response is returned with an appended warning note explaining that some titles could not be verified. The response is not silently edited.

**Why append a note rather than strip the content?**
Silently removing hallucinated content would make the system look more reliable than it is. Annotating the response keeps the user informed, which is more honest and more useful. A user who sees the warning can form their own judgment about the recommendation quality. A user whose response was quietly censored cannot.

**Limitations of the guardrail:**
The regex approach catches titles that Gemini renders in the prescribed format. A response that describes a song without quoting or bolding its title would pass the guardrail undetected. The guardrail is a meaningful check, not a proof of correctness.

---

## 5. Data

The catalog is `data/songs.csv` — 20 songs, hand-curated for variety across 14 genres and a range of moods, energy levels, and decades. Each song carries 12 attributes:

`id, title, artist, genre, mood, energy, tempo_bpm, valence, danceability, acousticness, popularity, release_decade, mood_tags`

This same file serves three roles in the system:
- **Ingestion source** for ChromaDB embeddings (via `SongKnowledgeBase.ingest_catalog()`)
- **Ground truth** for the `HallucinationGuardrail`'s `valid_titles` set
- **Legacy input** for the VibeScore 1.0 scoring logic still exercised by the test suite

Because it was assembled to cover a wide variety of genres rather than to reflect actual listening patterns, the catalog does not represent real-world popularity distributions. Genre-sparse queries (e.g., "metal") will surface the single available song in that category regardless of how well it fits.

---

## 6. Evaluation

### VibeScore 1.0 Baseline (Module 3)

The original system was evaluated by running a "Chill Lofi" profile (`genre=lofi, mood=chill, energy=0.35`) through all three scoring modes. Key findings:

- Switching to Mood-First mode allowed non-lofi songs with high mood synergy to surface, confirming the dynamic weights worked.
- *Focus Flow* correctly triggered the repeat-artist penalty because LoRoom already held a higher slot via *Midnight Coding*.

These results established that the math was correct. They also made the limitations visible: the system could not handle any query phrased in natural language, and the exact-match mood logic created a hard ceiling on recommendation quality.

### VibeScore 2.0 Qualitative Assessment

The agentic system was evaluated manually against three prompt types:

**Semantic match test:** Prompt "something melancholic and cinematic for a late-night drive." The system retrieved *Neon Noir* (darkwave, moody), *Night Drive Loop* (synthwave, moody), and *Velvet Underground* (soul, romantic) — songs that shared emotional territory even though none carried the exact tag "cinematic." VibeScore 1.0 would have scored these near-zero against a `mood=cinematic` query.

**Hallucination probe:** Prompt designed to elicit a non-catalog suggestion ("recommend something like Radiohead"). Gemini stayed grounded and drew from the catalog context rather than inventing songs, though the guardrail is the backstop for cases where it does not.

**Diversity verification (Deep Dive mode):** A broad query ("upbeat dance songs") confirmed that with `diversity_penalty=True`, the 10 returned candidates contained no repeated artists, even though several LoRoom and Neon Echo tracks ranked highly by relevance alone.

---

## 7. AI Collaboration Log

This project was built in collaboration with Claude (Anthropic), used here as a coding assistant. The collaboration is documented because transparency about AI-assisted development is part of responsible practice.

### What Claude designed

Claude drafted the full OOP architecture for `src/agent_system.py` based on a plain-language spec provided during the planning phase. The four-class structure — `SongKnowledgeBase`, `HallucinationGuardrail`, `ScoringModeConfig`, `VibeScoreAgent` — and the decision to express the Strategy pattern as system-prompt + retrieval-k configurations (rather than numeric weights) were Claude's architectural proposals, reviewed and approved before any code was written.

Claude also proposed wiring the diversity re-ranker through the Strategy pattern as a `diversity_penalty` boolean property on `ScoringModeConfig`, so that `DeepDiveMode` could opt into diversity enforcement while `BalancedMode` used the simpler direct retrieval path.

### Where I corrected the stack

Claude's first implementation used the Anthropic SDK (`anthropic.Anthropic()`) and a keyword-scoring loop as a proxy for retrieval. Both were wrong for this project's requirements:

- **Wrong AI provider:** The spec required Google Gemini, not Anthropic.
- **Not a true RAG system:** Keyword scoring is not retrieval-augmented generation. A RAG system requires a vector database — embeddings, similarity search, grounded context injection. The first implementation had none of this and would not have qualified as the agentic RAG upgrade the project required.

I identified these as critical violations and issued a precise correction: remove `anthropic`, integrate `langchain-google-genai` with `ChatGoogleGenerativeAI(model="gemini-3.0-flash")` and `GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")`, replace the keyword loop with `Chroma.from_documents()` and `similarity_search`. The guardrail, Strategy modes, and Streamlit chat history logic were explicitly preserved.

Claude executed the correction without modifying any of the preserved components, and the architectural intent remained intact.

### Reflection on the collaboration

The exchange illustrates a real dynamic in AI-assisted development: the assistant can produce architecturally sound OOP structures quickly, but it will default to familiar patterns (in this case, its own SDK) unless the constraints are stated precisely. The correction made the project meaningfully better — not just technically compliant, but actually a RAG system in the full sense of the term, with real semantic retrieval instead of a simulation of it.

---

## 8. Personal Reflection

VibeScore 1.0 taught me that a recommendation is not a guess — it is a scoring function with its author's priorities baked in. VibeScore 2.0 taught me something harder: that the same statement is true of a language model, but the function is no longer readable.

When the math awarded +2.0 for a genre match, I could point to the line. When Gemini curates a playlist, I cannot. The system is more capable — it understands "cinematic late-night energy" in a way that `mood == "cinematic"` never could — but the reasoning is implicit. The guardrail and diversity penalty are not just features; they are my attempt to put some of that legibility back. They are the parts of the system I can still read and reason about.

The stack correction also left an impression. The first version I received used Anthropic's SDK. It compiled and ran. It produced recommendations. But it was not doing what I asked — it was doing a keyword sort and calling it RAG. Recognizing that difference, and being precise enough about it to get a correct rewrite, is a skill that matters as much as knowing how to write the code in the first place.
