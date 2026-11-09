"""
chatbot.py
==========
A rule-based conversational agent that sits on top of the RecommenderEngine.

This is the project's primary interface variation: instead of a static script
or notebook, the user *talks* to the recommender in natural language. The agent
parses intents with lightweight pattern matching and routes each request to the
appropriate recommendation strategy (content-based, collaborative, or hybrid).

Example conversation
--------------------
    You: recommend something like Inception
    You: I feel like watching a comedy
    You: what should user 1 watch
    You: surprise me

Author: <Your Name>
"""

from __future__ import annotations

import random
import re
from typing import List, Tuple

from recommender import Movie, RecommenderEngine


class RecommenderChatbot:
    """Natural-language front end for the recommendation engine."""

    GREETINGS = ("hi", "hello", "hey", "yo", "greetings")

    def __init__(self) -> None:
        self.engine = RecommenderEngine()
        self.current_user: int | None = None

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def respond(self, message: str) -> str:
        """Return the agent's reply to a single user message."""
        text = message.strip().lower()
        if not text:
            return "Say something and I'll find you a movie!"

        # --- small talk -------------------------------------------------
        if any(text == g or text.startswith(g + " ") for g in self.GREETINGS):
            return ("Hi! I'm your movie buddy. Try: 'something like The Matrix', "
                    "'recommend a horror movie', or 'what should user 1 watch?'")
        if "help" in text:
            return self._help_text()
        if any(w in text for w in ("bye", "exit", "quit", "goodbye")):
            return "Enjoy the show! Goodbye."
        if "genre" in text and ("list" in text or "what" in text):
            return "Genres I know: " + ", ".join(self.engine.all_genres())

        # --- intent: recommend for a specific user id -------------------
        user_match = re.search(r"user\s+(\d+)", text)
        if user_match:
            uid = int(user_match.group(1))
            return self._user_recommendations(uid)

        # --- intent: "something like <title>" ---------------------------
        like_match = re.search(r"(?:like|similar to|such as)\s+(.+)", text)
        if like_match:
            return self._similar_recommendations(like_match.group(1))

        # --- intent: random / surprise ----------------------------------
        if any(w in text for w in ("surprise", "random", "anything", "whatever")):
            return self._surprise()

        # --- intent: genre request --------------------------------------
        genre = self._detect_genre(text)
        if genre:
            return self._genre_recommendations(genre)

        # --- fallback: treat the whole message as a title ---------------
        if self.engine.resolve_title(text) is not None:
            return self._similar_recommendations(text)

        return ("I didn't quite catch that. Try 'something like Toy Story', "
                "'recommend a sci-fi movie', or type 'help'.")

    # ------------------------------------------------------------------ #
    # Intent handlers
    # ------------------------------------------------------------------ #
    def _user_recommendations(self, uid: int) -> str:
        recs = self.engine.recommend_hybrid(uid)
        if not recs:
            recs = self.engine.recommend_for_user(uid)
        if not recs:
            return (f"I don't have a rating history for user {uid}. "
                    f"Known users are 1-{len(self.engine.ratings)}.")
        return (f"Based on user {uid}'s taste, I'd suggest:\n"
                + self._format(recs))

    def _similar_recommendations(self, title: str) -> str:
        mid = self.engine.resolve_title(title)
        if mid is None:
            return (f"I couldn't find '{title.strip()}' in the catalog. "
                    "Try another title.")
        matched = self.engine.movies[mid]
        recs = self.engine.similar_to_movie(matched.title)
        return (f"Because you like {matched.title} ({matched.year}), try:\n"
                + self._format(recs))

    def _genre_recommendations(self, genre: str) -> str:
        movies = self.engine.recommend_by_genre(genre)
        if not movies:
            return f"I don't have any {genre} movies right now."
        scored = [(m, self.engine.average_rating(m.movie_id)) for m in movies]
        return (f"Top {genre.title()} picks:\n" + self._format(scored))

    def _surprise(self) -> str:
        movie = random.choice(list(self.engine.movies.values()))
        return (f"How about {movie.title} ({movie.year})? "
                f"It's a {'/'.join(movie.genres)} film. {movie.description}")

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    def _detect_genre(self, text: str) -> str | None:
        for genre in self.engine.all_genres():
            if genre.lower() in text:
                return genre
        aliases = {
            "funny": "Comedy", "scary": "Horror", "space": "Sci-Fi",
            "love": "Romance", "superhero": "Action", "cartoon": "Animation",
        }
        for alias, genre in aliases.items():
            if alias in text:
                return genre
        return None

    @staticmethod
    def _format(items: List[Tuple[Movie, float]]) -> str:
        lines = []
        for i, (movie, score) in enumerate(items, 1):
            lines.append(f"  {i}. {movie.title} ({movie.year}) "
                         f"- {'/'.join(movie.genres)}")
        return "\n".join(lines)

    @staticmethod
    def _help_text() -> str:
        return (
            "Here is what I can do:\n"
            "  - 'something like The Dark Knight'  (content-based)\n"
            "  - 'recommend a comedy'              (genre filter)\n"
            "  - 'what should user 3 watch?'       (collaborative + hybrid)\n"
            "  - 'surprise me'                     (random pick)\n"
            "  - 'list genres'                     (show catalog genres)\n"
            "  - 'bye'                             (exit)"
        )


def run_cli() -> None:
    """Run the chatbot as an interactive terminal session."""
    bot = RecommenderChatbot()
    print("=" * 60)
    print("  MOVIE RECOMMENDER CHATBOT")
    print("  Type 'help' for options or 'bye' to quit.")
    print("=" * 60)
    print("Bot: Hi! What are you in the mood to watch?")
    while True:
        try:
            user_in = input("You: ")
        except (EOFError, KeyboardInterrupt):
            print("\nBot: Goodbye!")
            break
        reply = bot.respond(user_in)
        print("Bot:", reply)
        if any(w in user_in.lower() for w in ("bye", "exit", "quit")):
            break


if __name__ == "__main__":
    run_cli()
