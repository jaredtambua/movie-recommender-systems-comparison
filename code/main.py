import cli
import svd_engine as engine


def main():
    engine.initialise()

    while True:
        cli.banner("THE MOVIE RECOMMENDER (Hybrid: SVD + CBF)")
        cli.menu()
        c = cli.prompt_choice()

        if c == "3":
            print("Goodbye.")
            return

        if c == "1":
            uid = cli.prompt_user_id(engine.user_exists)
            if uid == "__CANCEL__":
                continue
            cli.show_recs(engine.recommend, uid, k=10)

        if c == "2":
            uid = engine.random_user_id()
            print(f"\nSelected random user: {uid}")
            cli.show_recs(engine.recommend, uid, k=10)


if __name__ == "__main__":
    main()
