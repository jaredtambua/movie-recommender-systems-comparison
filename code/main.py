import cli
import hybrid_engine as engine


def main():
    engine.initialise()

    while True:
        cli.print_banner("Hybrid Recommender (SVD + CBF)")
        cli.menu()
        c = cli.prompt_choice()

        if c == "3":
            print("Goodbye.")
            return

        if c == "1":
            uid = cli.prompt_user_id(engine.user_exists)
            if uid == "__CANCEL__":
                return

            recs = engine.recommend(uid, k=100)
            start = 0
            first_page = True

            while True:
                shown = cli.show_recommendations(recs, start, page_size=10)
                if not shown:
                    return

                if first_page:
                    cli.print_personalised_message()
                    first_page = False

                choice2 = cli.prompt_after_table()
                if choice2 == "2":
                    return

                start += 10

        if c == "2":
            uid = engine.random_user_id()
            print(f"\nSelected random user: {uid}")

            recs = engine.recommend(uid, k=100)
            start = 0
            first_page = True

            while True:
                shown = cli.show_recommendations(recs, start, page_size=10)
                if not shown:
                    return

                if first_page:
                    cli.print_personalised_message()
                    first_page = False

                choice2 = cli.prompt_after_table()
                if choice2 == "2":
                    return

                start += 10


if __name__ == "__main__":
    main()
