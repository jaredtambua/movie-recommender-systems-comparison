import hybrid_engine

if __name__ == "__main__":
    hybrid_engine.initialise()  # will load cache if present
    uids = sorted(hybrid_engine.user_map.keys())
    with open("user_ids.txt", "w", encoding="utf-8") as f:
        for u in uids[:200]:  # export first 200 (or all if you want)
            f.write(str(u) + "\n")
    print("Wrote user_ids.txt")
