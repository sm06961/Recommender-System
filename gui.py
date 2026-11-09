"""
gui.py
======
A simple Tkinter chat window for the Movie Recommender.

This provides a graphical variation of the conversational interface. It reuses
the exact same RecommenderChatbot brain used by the command-line version, so the
recommendation logic is shared and not duplicated. Tkinter ships with the Python
standard library, so no extra installation is required.

Run with:  py gui.py
Author: <Your Name>
"""

from __future__ import annotations

import tkinter as tk
from tkinter import scrolledtext

from chatbot import RecommenderChatbot


class ChatGUI:
    def __init__(self, root: tk.Tk) -> None:
        self.bot = RecommenderChatbot()
        self.root = root
        root.title("Movie Recommender Chatbot")
        root.geometry("620x560")
        root.configure(bg="#1e1e2e")

        header = tk.Label(
            root, text="🎬  Movie Recommender", font=("Segoe UI", 16, "bold"),
            bg="#1e1e2e", fg="#89b4fa", pady=10,
        )
        header.pack(fill=tk.X)

        self.history = scrolledtext.ScrolledText(
            root, wrap=tk.WORD, state=tk.DISABLED, font=("Segoe UI", 11),
            bg="#181825", fg="#cdd6f4", insertbackground="#cdd6f4",
            padx=12, pady=12, borderwidth=0,
        )
        self.history.pack(fill=tk.BOTH, expand=True, padx=12, pady=(0, 8))
        self.history.tag_config("bot", foreground="#a6e3a1")
        self.history.tag_config("user", foreground="#f9e2af")

        entry_frame = tk.Frame(root, bg="#1e1e2e")
        entry_frame.pack(fill=tk.X, padx=12, pady=(0, 12))

        self.entry = tk.Entry(
            entry_frame, font=("Segoe UI", 11), bg="#313244", fg="#cdd6f4",
            insertbackground="#cdd6f4", relief=tk.FLAT,
        )
        self.entry.pack(side=tk.LEFT, fill=tk.X, expand=True, ipady=8, padx=(0, 8))
        self.entry.bind("<Return>", self._on_send)
        self.entry.focus()

        send_btn = tk.Button(
            entry_frame, text="Send", command=self._on_send,
            font=("Segoe UI", 10, "bold"), bg="#89b4fa", fg="#1e1e2e",
            relief=tk.FLAT, padx=16, activebackground="#74c7ec",
        )
        send_btn.pack(side=tk.RIGHT)

        self._post("Bot", "Hi! What are you in the mood to watch? "
                           "Type 'help' to see what I can do.")

    def _on_send(self, event=None) -> None:
        message = self.entry.get().strip()
        if not message:
            return
        self.entry.delete(0, tk.END)
        self._post("You", message)
        reply = self.bot.respond(message)
        self._post("Bot", reply)

    def _post(self, who: str, text: str) -> None:
        tag = "user" if who == "You" else "bot"
        self.history.configure(state=tk.NORMAL)
        self.history.insert(tk.END, f"{who}: ", tag)
        self.history.insert(tk.END, f"{text}\n\n")
        self.history.configure(state=tk.DISABLED)
        self.history.see(tk.END)


def main() -> None:
    root = tk.Tk()
    ChatGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
