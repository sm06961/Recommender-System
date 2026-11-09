# Movie Recommender Chatbot

A dependency-free movie recommendation system with a conversational interface.
Built for an AI course assignment to demonstrate the theory and practice of
recommendation systems.

## What it does

The system recommends movies using three classic strategies, all implemented
from scratch in pure Python (standard library only):

| Strategy | Technique | Where it's used |
|----------|-----------|-----------------|
| **Content-based** | TF-IDF vectors + cosine similarity over genres & plot keywords | "something like *Inception*" |
| **Collaborative** | User-based k-nearest-neighbors on the rating matrix | "what should user 4 watch?" |
| **Hybrid** | Weighted blend of content + collaborative signals | default user recommendations |

The **interface variation** for this project is a **conversational chatbot**,
available both as a terminal agent and as a Tkinter GUI window.

## Project structure

```
Rec System/
├── app.py            # single entry point (GUI / CLI / demo)
├── recommender.py    # core engine: content-based, collaborative, hybrid
├── chatbot.py        # rule-based natural-language agent
├── gui.py            # Tkinter chat window
├── requirements.txt  # optional dependencies (core needs none)
├── data/
│   ├── movies.csv    # 30-movie catalog (title, year, genres, plot)
│   └── ratings.csv   # 8 users, ~56 ratings
└── README.md
```

## Running it

No installation is required for the core app — it uses only the Python
standard library (including Tkinter, which ships with Python on Windows).

```powershell
py app.py            # launch the graphical chatbot (default)
py app.py --cli      # run the text chatbot in the terminal
py app.py --demo     # print a non-interactive demo of all three strategies
```

### Example chatbot prompts

```
something like The Dark Knight
recommend a comedy
what should user 3 watch?
surprise me
list genres
help
```

## How it works

1. **Data loading** – `movies.csv` and `ratings.csv` are parsed into in-memory
   dictionaries.
2. **Feature engineering** – each movie's genres (weighted ×3) and plot
   keywords are combined into a "token soup," then converted to TF-IDF vectors.
3. **Content similarity** – cosine similarity ranks the catalog against a target
   title.
4. **Collaborative filtering** – user similarity is computed with cosine
   similarity over shared ratings; the top-k neighbors' ratings are blended into
   predicted scores for unseen movies.
5. **Hybrid** – collaborative predictions and content similarity to the user's
   favorite film are combined with a tunable weight.
6. **Chatbot** – parses the user's intent with lightweight pattern matching and
   routes to the correct strategy.

## Credits / foundation

This project is adapted from widely published MovieLens recommendation
tutorials, in particular:

- Banik, R. (2018). *Hands-On Recommendation Systems with Python*. Packt.
- scikit-learn documentation, `TfidfVectorizer` and `cosine_similarity`.
- GroupLens Research, the MovieLens dataset (structure inspiration).

The original tutorials rely on **pandas** and **scikit-learn**. This version
re-implements the TF-IDF, cosine-similarity, and user-KNN mathematics from
scratch so the algorithms are explicit and the app runs with zero dependencies,
and it adds a **conversational chatbot + GUI** interface that the tutorials do
not provide.

## Optional: pandas/scikit-learn version

`requirements.txt` lists the libraries used by the original tutorials. They are
**not needed** to run this project but are documented for reference.
