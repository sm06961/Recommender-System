"""
recommender.py
==============
Core recommendation engine for the Movie Recommender project.

This module implements three classic recommendation strategies using only the
Python standard library so that it runs in any environment without third-party
dependencies:

    1. Content-based filtering  -> TF-IDF vectors + cosine similarity over
       movie metadata (genres + plot keywords).
    2. Collaborative filtering  -> user-based nearest neighbors using cosine
       similarity over the user/item rating matrix.
    3. Hybrid filtering         -> a weighted blend of the two signals above.

Foundation / credit
--------------------
The design of this engine is adapted from widely published MovieLens
tutorials, in particular the content-based and item-similarity patterns
demonstrated by Banik (2018) in *Hands-On Recommendation Systems with Python*
and the scikit-learn / MovieLens cosine-similarity examples. The original
tutorials rely on pandas and scikit-learn; this version re-implements the same
mathematics from scratch to make the algorithms explicit and dependency-free.

Author: <Your Name>
"""

from __future__ import annotations

import csv
import math
import os
import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

# A small English stop-word list so common words do not dominate TF-IDF.
STOP_WORDS = {
    "a", "an", "the", "and", "or", "of", "to", "in", "on", "with", "for",
    "his", "her", "its", "their", "who", "whom", "that", "this", "as", "at",
    "by", "from", "into", "is", "are", "was", "were", "be", "been", "being",
    "he", "she", "they", "them", "it", "but", "not", "must", "one", "two",
    "new", "out", "up", "after", "before", "during", "while", "against",
}


@dataclass
class Movie:
    """A single catalog item."""
    movie_id: int
    title: str
    year: int
    genres: List[str]
    description: str
    tokens: List[str] = field(default_factory=list)


def tokenize(text: str) -> List[str]:
    """Lower-case, split on non-letters, and drop stop words / short tokens."""
    words = re.findall(r"[a-zA-Z]+", text.lower())
    return [w for w in words if w not in STOP_WORDS and len(w) > 2]


class RecommenderEngine:
    """Loads data and produces content-based, collaborative, and hybrid picks."""

    def __init__(self, data_dir: str = DATA_DIR) -> None:
        self.data_dir = data_dir
        self.movies: Dict[int, Movie] = {}
        self.title_to_id: Dict[str, int] = {}
        # ratings[user_id][movie_id] = rating
        self.ratings: Dict[int, Dict[int, float]] = defaultdict(dict)
        # item_users[movie_id] = {user_id: rating}
        self.item_users: Dict[int, Dict[int, float]] = defaultdict(dict)
        self._tfidf: Dict[int, Dict[str, float]] = {}
        self._load_movies()
        self._load_ratings()
        self._build_tfidf()

    # ------------------------------------------------------------------ #
    # Data loading
    # ------------------------------------------------------------------ #
    def _load_movies(self) -> None:
        path = os.path.join(self.data_dir, "movies.csv")
        with open(path, newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                genres = [g.strip() for g in row["genres"].split("|") if g.strip()]
                movie = Movie(
                    movie_id=int(row["movieId"]),
                    title=row["title"].strip(),
                    year=int(row["year"]),
                    genres=genres,
                    description=row["description"].strip(),
                )
                # Feature "soup": genres are weighted (repeated) so they count
                # more heavily than plot words, then merged with keywords.
                soup = [g.lower() for g in genres for _ in range(3)]
                soup += tokenize(movie.description)
                movie.tokens = soup
                self.movies[movie.movie_id] = movie
                self.title_to_id[movie.title.lower()] = movie.movie_id

    def _load_ratings(self) -> None:
        path = os.path.join(self.data_dir, "ratings.csv")
        with open(path, newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                uid = int(row["userId"])
                mid = int(row["movieId"])
                rating = float(row["rating"])
                self.ratings[uid][mid] = rating
                self.item_users[mid][uid] = rating

    # ------------------------------------------------------------------ #
    # Content-based filtering
    # ------------------------------------------------------------------ #
    def _build_tfidf(self) -> None:
        """Compute TF-IDF weights for every movie's token soup."""
        n_docs = len(self.movies)
        # Document frequency for each term.
        df: Dict[str, int] = defaultdict(int)
        for movie in self.movies.values():
            for term in set(movie.tokens):
                df[term] += 1

        for mid, movie in self.movies.items():
            tf: Dict[str, int] = defaultdict(int)
            for term in movie.tokens:
                tf[term] += 1
            vec: Dict[str, float] = {}
            for term, count in tf.items():
                idf = math.log((1 + n_docs) / (1 + df[term])) + 1.0
                vec[term] = count * idf
            self._tfidf[mid] = self._normalize(vec)

    @staticmethod
    def _normalize(vec: Dict[str, float]) -> Dict[str, float]:
        norm = math.sqrt(sum(v * v for v in vec.values()))
        if norm == 0:
            return vec
        return {k: v / norm for k, v in vec.items()}

    @staticmethod
    def _cosine(a: Dict[str, float], b: Dict[str, float]) -> float:
        """Cosine similarity of two sparse (already L2-normalized) vectors."""
        if len(a) > len(b):
            a, b = b, a
        return sum(weight * b.get(term, 0.0) for term, weight in a.items())

    def similar_to_movie(self, title: str, top_n: int = 5) -> List[Tuple[Movie, float]]:
        """Return the most content-similar movies to a given title."""
        mid = self.resolve_title(title)
        if mid is None:
            return []
        target = self._tfidf[mid]
        scored = []
        for other_id, vec in self._tfidf.items():
            if other_id == mid:
                continue
            scored.append((self.movies[other_id], self._cosine(target, vec)))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_n]

    def recommend_by_genre(self, genre: str, top_n: int = 5) -> List[Movie]:
        """Simple catalog filter used by the chatbot for genre requests."""
        genre = genre.lower()
        matches = [m for m in self.movies.values()
                   if any(genre in g.lower() for g in m.genres)]
        # Rank by average rating so better-liked titles surface first.
        matches.sort(key=lambda m: self.average_rating(m.movie_id), reverse=True)
        return matches[:top_n]

    # ------------------------------------------------------------------ #
    # Collaborative filtering (user-based k-nearest-neighbors)
    # ------------------------------------------------------------------ #
    def _user_similarity(self, user_a: int, user_b: int) -> float:
        ra, rb = self.ratings[user_a], self.ratings[user_b]
        shared = set(ra) & set(rb)
        if not shared:
            return 0.0
        dot = sum(ra[m] * rb[m] for m in shared)
        na = math.sqrt(sum(v * v for v in ra.values()))
        nb = math.sqrt(sum(v * v for v in rb.values()))
        if na == 0 or nb == 0:
            return 0.0
        return dot / (na * nb)

    def recommend_for_user(self, user_id: int, top_n: int = 5,
                           k_neighbors: int = 3) -> List[Tuple[Movie, float]]:
        """Predict unseen movies for a known user via neighbor rating blends."""
        if user_id not in self.ratings:
            return []
        seen = set(self.ratings[user_id])

        neighbors = []
        for other in self.ratings:
            if other == user_id:
                continue
            sim = self._user_similarity(user_id, other)
            if sim > 0:
                neighbors.append((other, sim))
        neighbors.sort(key=lambda x: x[1], reverse=True)
        neighbors = neighbors[:k_neighbors]

        scores: Dict[int, float] = defaultdict(float)
        weight_sum: Dict[int, float] = defaultdict(float)
        for other, sim in neighbors:
            for mid, rating in self.ratings[other].items():
                if mid in seen:
                    continue
                scores[mid] += sim * rating
                weight_sum[mid] += sim

        predictions = []
        for mid, total in scores.items():
            predicted = total / weight_sum[mid] if weight_sum[mid] else 0.0
            predictions.append((self.movies[mid], predicted))
        predictions.sort(key=lambda x: x[1], reverse=True)
        return predictions[:top_n]

    # ------------------------------------------------------------------ #
    # Hybrid filtering
    # ------------------------------------------------------------------ #
    def recommend_hybrid(self, user_id: int, top_n: int = 5,
                         content_weight: float = 0.5) -> List[Tuple[Movie, float]]:
        """Blend collaborative predictions with content similarity to the
        user's favorite (highest-rated) movie."""
        if user_id not in self.ratings or not self.ratings[user_id]:
            return []

        # Collaborative component (normalized to 0..1 on a 5-point scale).
        collab = {m.movie_id: score / 5.0
                  for m, score in self.recommend_for_user(user_id, top_n=50)}

        # Content component: similarity to the user's top-rated movie.
        fav_id = max(self.ratings[user_id], key=self.ratings[user_id].get)
        content = {m.movie_id: score
                   for m, score in self.similar_to_movie(
                       self.movies[fav_id].title, top_n=50)}

        seen = set(self.ratings[user_id])
        combined: Dict[int, float] = defaultdict(float)
        for mid in set(collab) | set(content):
            if mid in seen:
                continue
            combined[mid] = (content_weight * content.get(mid, 0.0)
                             + (1 - content_weight) * collab.get(mid, 0.0))

        ranked = [(self.movies[mid], score) for mid, score in combined.items()]
        ranked.sort(key=lambda x: x[1], reverse=True)
        return ranked[:top_n]

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    def average_rating(self, movie_id: int) -> float:
        users = self.item_users.get(movie_id, {})
        if not users:
            return 0.0
        return sum(users.values()) / len(users)

    def resolve_title(self, title: str) -> int | None:
        """Match a user-typed title to a catalog id (exact or fuzzy substring)."""
        key = title.strip().lower()
        if key in self.title_to_id:
            return self.title_to_id[key]
        # Fall back to substring / token overlap matching.
        best_id, best_score = None, 0.0
        query_tokens = set(tokenize(key)) or {key}
        for name, mid in self.title_to_id.items():
            if key in name:
                return mid
            overlap = len(query_tokens & set(tokenize(name)))
            if overlap > best_score:
                best_id, best_score = mid, overlap
        return best_id if best_score > 0 else None

    def all_genres(self) -> List[str]:
        genres = set()
        for m in self.movies.values():
            genres.update(m.genres)
        return sorted(genres)


if __name__ == "__main__":
    engine = RecommenderEngine()
    print("Loaded", len(engine.movies), "movies and",
          len(engine.ratings), "users.\n")

    print("Content-based: movies similar to 'The Matrix'")
    for movie, score in engine.similar_to_movie("The Matrix"):
        print(f"  {movie.title} ({movie.year})  score={score:.3f}")

    print("\nCollaborative: recommendations for user 1")
    for movie, score in engine.recommend_for_user(1):
        print(f"  {movie.title} ({movie.year})  predicted={score:.2f}")

    print("\nHybrid: recommendations for user 1")
    for movie, score in engine.recommend_hybrid(1):
        print(f"  {movie.title} ({movie.year})  score={score:.3f}")
