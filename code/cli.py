# cli.py
from __future__ import annotations


def banner(title):
    line = "=" * 58
    print(line)
    print(f"{title:^58}")
    print("MovieLens 1M".center(58))
    print(line)


def menu():
    print("\n[1] - Select a user ID")
    print("[2] - Select a random user ID")
    print("[3] - Exit")


def prompt_choice():
    while True:
        c = input("Choose an option: ").strip()
        if c in {"1", "2", "3"}:
            return c
        print("Invalid choice. Please enter 1, 2, or 3.")


def prompt_user_id(user_exists_fn):
    while True:
        raw = input("Enter user ID (or 'x' to cancel): ").strip()
        if not raw:
            continue
        if raw.lower() == "x":
            return "__CANCEL__"
        if user_exists_fn(raw):
            return raw
        print("INVALID INPUT - That user ID does not exist in this dataset.")


def show_recs(recommend_fn, user_id, k=10):
    recs = recommend_fn(user_id=user_id, k=k)

    print("\n" + "-" * 58)
    print(f"Top {len(recs)} recommendations for user {user_id}")
    print("-" * 58)

    if not recs:
        print("No recommendations available.")
        return

    for rank, (title, score) in enumerate(recs, start=1):
        print(f"{rank:02d}. {title}  (pred: {score:.2f})")
