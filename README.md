# 🎵 Music Recommender Simulation: VibeScore 1.0 (Pro)

## Project Summary

VibeScore 1.0 is a classroom simulation of a content-based music recommender system. The goal of this project is to represent a catalog of songs and a user's "taste profile" as structured data, design a transparent, math-based scoring rule to generate personalized recommendations, and critically evaluate the biases and limitations of that algorithm.

---

## How The System Works

Real-world music platforms like Spotify use two core approaches to generate recommendations. **Content-based filtering** analyzes the intrinsic properties of each song, while **collaborative filtering** mines the collective listening behavior of millions of users. 

**This simulation focuses exclusively on the content-based layer:**
* **Song Features:** Each song carries core attributes (genre, mood, energy) as well as complex attributes (popularity, release decade, acousticness, danceability, and detailed mood tags).
* **User Profile:** Stores preference targets: `favorite_genre`, `favorite_mood`, and `target_energy`.
* **Dynamic Scoring Modes:** The `Recommender` uses a Strategy pattern to allow users to switch weighting logic:
  * **Balanced Mode:** +2.0 Genre, +1.0 Mood, +0-1.0 Energy.
  * **Genre-First Mode:** +4.0 Genre, +1.0 Mood, +0-1.0 Energy.
  * **Mood-First Mode:** +2.0 Genre, +3.0 Mood, +0-1.0 Energy.
* **Diversity Penalty:** To prevent "filter bubbles" of a single artist, the ranking algorithm applies a strict **-1.0 penalty** to any song if its artist is already present higher up in the top recommendations. 
* **Selection:** The system sorts the catalog by final score and returns the Top 5 results with generated "reasons" explaining the math.

---

## Getting Started

### Setup

1. Create a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate      # Mac or Linux
   .venv\Scripts\activate         # Windows

    Install dependencies:
    Bash

    pip install -r requirements.txt

    Run the app:
    Bash

    python -m src.main

Experiments You Tried

To stress-test the dynamic strategies and the fairness logic, the "Chill Lofi" profile (genre=lofi, mood=chill, energy=0.35) was run through all three scoring modes. The output was formatted into an ASCII table using tabulate.
Results Output:

Observation: The system successfully adjusted rankings based on the active mode. Furthermore, the Diversity Penalty correctly triggered on Focus Flow, penalizing it by -1.0 because its artist (LoRoom) was already recommended higher up in the list (via Midnight Coding). This successfully forced the system to present a wider variety of artists.
Limitations and Risks

    Rigid String Matching: The system does not understand semantic similarity. The mood tags "intense" and "aggressive" are treated as completely unrelated, leading to missed recommendations.

    Tiny Catalog Bias: With only 20 songs, heavily weighting a specific feature (like in Genre-First mode) often forces the system to recommend a mathematically poor-fitting song simply because it's the only track available in that category.

For a deeper analysis, please read the full Model Card linked below.
Reflection

Read the full Model Card here

Building VibeScore 1.0 made concrete something that is easy to take for granted in apps like music streaming platforms: a recommendation is not a guess, it is the output of a scoring function. The most interesting discovery was seeing how adding a simple mathematical rule—like subtracting 1.0 point for a repeat artist—instantly made the playlist feel more human and less like a database query. It showed me how algorithmic fairness (like diversity) has to be explicitly programmed into the system, or the math will naturally create repetitive filter bubbles.