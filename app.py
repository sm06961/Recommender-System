"""
app.py
======
Single entry point for the Movie Recommender project.

Usage:
    py app.py            # launches the graphical chatbot (default)
    py app.py --cli      # runs the text chatbot in the terminal
    py app.py --demo     # prints a non-interactive demo of all 3 strategies

Author: <Your Name>
"""

from __future__ import annotations

import sys


def run_demo() -> None:
    from recommender import RecommenderEngine
    engine = RecommenderEngine()
    print(f"Loaded {len(engine.movies)} movies and "
          f"{len(engine.ratings)} users.\n")

    print("[Content-based] Similar to 'Inception':")
    for movie, score in engine.similar_to_movie("Inception"):
        print(f"   {movie.title} ({movie.year})  score={score:.3f}")

    print("\n[Collaborative] Picks for user 4:")
    for movie, score in engine.recommend_for_user(4):
        print(f"   {movie.title} ({movie.year})  predicted={score:.2f}")

    print("\n[Hybrid] Picks for user 4:")
    for movie, score in engine.recommend_hybrid(4):
        print(f"   {movie.title} ({movie.year})  score={score:.3f}")


def main() -> None:
    args = set(sys.argv[1:])
    if "--demo" in args:
        run_demo()
    elif "--cli" in args:
        from chatbot import run_cli
        run_cli()
    else:
        try:
            from gui import main as gui_main
            gui_main()
        except Exception as exc:  # e.g. no display available
            print(f"Could not start GUI ({exc}). Falling back to CLI.\n")
            from chatbot import run_cli
            run_cli()


if __name__ == "__main__":
    main()
