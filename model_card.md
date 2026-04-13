# 🎧 Model Card: Music Recommender Simulation

## 1. Model Name

**VibeScore 1.0 (Pro Edition)**

---

## 2. Intended Use

VibeScore 1.0 suggests songs from a fixed catalog based on three things a user tells it: favorite genre, preferred mood, and target energy. This system is a classroom simulation designed to make the inner workings of a scoring algorithm transparent, featuring modular scoring strategies and anti-bias diversity logic. It is not designed to serve real users at scale.

---

## 3. How the Model Works

The system scores songs by comparing their attributes to the user's profile. However, unlike static algorithms, VibeScore utilizes dynamic **Scoring Modes** to change the mathematical weights:

* **Balanced Mode:** +2.0 Genre, +1.0 Mood.
* **Genre-First Mode:** +4.0 Genre, +1.0 Mood.
* **Mood-First Mode:** +2.0 Genre, +3.0 Mood.

Energy similarity is always calculated as `1.0 - abs(song_energy - target_energy)`, awarding between 0.0 and 1.0 points. 

**Fairness Logic:** To prevent artist monopolization, the ranking engine includes a **Diversity Penalty**. As it builds the final list, if it encounters a song by an artist who is already in the top results, it subtracts 1.0 from that song's final score, pushing it down to give other creators a chance to surface.

---

## 4. Data

The catalog contains 20 songs stored in `data/songs.csv`. Each song has 13 attributes: a unique ID, title, artist, genre, broad mood label, energy (0–1), tempo in beats per minute, valence (musical positivity, 0–1), danceability (0–1), acousticness (0–1), a popularity score (0–100), release decade, and a set of detailed mood tags. 

The dataset spans 14 genres and was hand-crafted for this simulation. Because the catalog was assembled to cover a variety of genres and moods rather than to reflect real listening trends, it does not proportionally represent what people actually listen to.

---

## 5. Strengths

**Algorithmic Transparency & Control:** Every score comes with a printed explanation showing exactly which points were awarded and why. By allowing the user to switch between "Genre-First" and "Mood-First" modes, the system empowers the user to manually break out of filter bubbles.

**Built-in Diversity:** The anti-repetition penalty effectively stops the algorithm from spamming a user with a single artist just because that artist perfectly matches the mathematical profile. This makes the output feel more like a curated human playlist and less like a raw data sort.

---

## 6. Limitations and Bias

**Exact-string mood matching misses near synonyms:** The mood comparison uses simple equality. Words like "intense" and "aggressive" describe nearly the same emotional territory, yet the system awards zero mood points if they don't match exactly. A real user looking for intense music would likely find these omissions confusing.

**Twenty-song catalog limits meaningful differentiation:** When a user selects a niche genre (e.g., metal), the genre bonus essentially selects the single song available in that category and then ranks everything else by energy proximity. There is no meaningful competition among songs, meaning recommendations reflect catalog gaps more than actual user taste.

---

## 7. Evaluation

The system was evaluated by running the "Chill Lofi" profile (`genre=lofi, mood=chill, energy=0.35`) through all three scoring modes to probe different failure and success modes.

* **Ranking Shifts:** Switching to Mood-First mode successfully allowed non-lofi songs with high mood/energy synergy to compete more aggressively for the top slots, proving the dynamic weights worked.
* **Penalty Verification:** During the test, the song *Focus Flow* correctly triggered the "repeat artist penalty (-1.0)" because its artist (LoRoom) had already claimed a higher spot with *Midnight Coding*. This proved the fairness logic functioned as intended.

---

## 8. Future Work

**1. Semantic mood similarity instead of exact matching:**
Replace the binary mood comparison with a small lookup table or embedding that assigns partial credit for emotionally adjacent moods. For example, "aggressive" and "intense" might score 0.7 instead of 0. This reduces the dependency on whoever wrote each song's exact mood tag.

**2. Continuous Slider Weights:**
Instead of hard-coded "Modes," provide a UI where users can drag sliders to set their own exact floating-point weights for Genre, Mood, and Energy. This gives total personalization without requiring complex machine learning.

---

## 9. Personal Reflection

Building VibeScore 1.0 made concrete something that is easy to take for granted in apps like music streaming platforms: a recommendation is not a guess, it is the output of a scoring function with weights baked in by whoever wrote it. The most interesting discovery was seeing how adding a simple mathematical rule—like subtracting 1.0 point for a repeat artist—instantly made the playlist feel more human and less like a database query. It showed me how algorithmic fairness (like diversity) has to be explicitly programmed into the system, or the math will naturally create repetitive filter bubbles. The tension between simple, transparent math and fair, nuanced recommendations became very clear.