from __future__ import annotations


def banner(title):
    line = "=" * 58
    print(line)
    print(f"{title:^58}")
    # print("MovieLens 1M".center(58))
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


def prompt_user_id(is_valid_user):
    while True:
        user_input = input("Enter user ID (or 'x' to exit): ").strip()
        if not user_input:
            continue

        if user_input.lower() == "x":
            return "__CANCEL__"

        if is_valid_user(user_input):
            return user_input
        print("INVALID INPUT - That user ID does not exist in this dataset.")


def _truncate(text, max_len):
    text = str(text)

    if len(text) <= max_len:
        return text

    if max_len <= 3:
        return text[:max_len]

    return text[: max_len - 3] + "..."


def _format_table(headers, rows, max_col_widths=None):
    headers = [str(h) for h in headers]
    rows = [[str(c) for c in r] for r in rows]

    ncols = len(headers)

    if max_col_widths is None:
        max_col_widths = [9999] * ncols
    else:
        max_col_widths = list(max_col_widths)
        while len(max_col_widths) < ncols:
            max_col_widths.append(9999)

    widths = []
    for j in range(ncols):
        col_items = [headers[j]]
        for r in rows:
            col_items.append(r[j])

        widest = max(len(x) for x in col_items)
        widths.append(min(widest, max_col_widths[j]))

    headers = [_truncate(headers[j], widths[j]) for j in range(ncols)]
    rows = [[_truncate(r[j], widths[j]) for j in range(ncols)] for r in rows]

    def make_line(ch):
        parts = []
        for w in widths:
            parts.append(ch * (w + 2))
        return "+" + "+".join(parts) + "+"

    def make_row(values):
        cells = []
        for j in range(ncols):
            cells.append(values[j].ljust(widths[j]))
        return "| " + " | ".join(cells) + " |"

    lines = []
    lines.append(make_line("-"))
    lines.append(make_row(headers))
    lines.append(make_line("="))

    for r in rows:
        lines.append(make_row(r))

    lines.append(make_line("-"))
    return "\n".join(lines)


def show_recommendations(recs, start_idx, page_size=10):
    end_idx = start_idx + page_size
    chunk = recs[start_idx:end_idx]

    if not chunk:
        print("\nNo more recommendations available.")
        return False

    rows = []
    for rank, (title, score) in enumerate(chunk, start=start_idx + 1):
        rows.append([str(rank), title, f"{score:.2f}"])

    table = _format_table(
        headers=["Rank", "Title", "Pred"],
        rows=rows,
        max_col_widths=[10, 50, 10],
    )
    print(table)
    return True


def print_banner(system_name):
    print("=" * 58)
    print(f"{'THE MOVIE RECOMMENDER':^58}")
    print(f"{system_name:^58}")
    print("=" * 58)


def print_personalised_message() -> None:
    print(
        "\nThese recommendations are personalised for you based on your past ratings "
        "and users with similar tastes."
    )


def prompt_after_table():
    while True:
        choice = input(
            "\nWould you like to see more recommendations? [1] Yes  [2] Exit: "
        ).strip()
        if choice in {"1", "2"}:
            return choice
        print("Invalid choice. Please enter 1 or 2.")
