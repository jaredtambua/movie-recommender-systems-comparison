import sys

from src import cli
from src import hybrid_engine
from src import ncf_engine

RECOMMENDATION_COUNT = 100
PAGE_SIZE = 10


def main():
    while True:
        cli.print_banner("Movie Recommender System")
        print("\nChoose a recommender:")
        print("[1] - Hybrid Recommender (SVD + CBF)")
        print("[2] - Neural Collaborative Filtering (NCF)")
        print("[3] - Exit")

        choice = input("Choose an option: ").strip()

        if choice == "1":
            run_recommender(hybrid_engine, "Hybrid Recommender (SVD + CBF)")

        elif choice == "2":
            run_recommender(ncf_engine, "Neural Collaborative Filtering (NCF)")

        elif choice == "3":
            exit_program()

        else:
            print("Invalid choice. Please enter 1, 2, or 3.")


def run_recommender(engine, system_name):
    engine.initialise()

    while True:
        cli.print_banner(system_name)
        cli.menu()
        choice = cli.prompt_choice()

        if choice == "1":
            user_id = cli.prompt_user_id(engine.user_exists)

            if user_id == "__CANCEL__":
                exit_program()

            print(f"\nDisplaying recommendations for User: {user_id}")
            show_recommendations(engine, user_id)

        elif choice == "2":
            user_id = engine.random_user_id()
            print(f"\nDisplaying recommendations for User: {user_id}")

            show_recommendations(engine, user_id)

        elif choice == "3":
            exit_program()


def show_recommendations(engine, user_id):
    recommendations = engine.recommend(user_id, k=RECOMMENDATION_COUNT)
    start = 0
    first_page = True

    while True:
        shown = cli.show_recommendations(
            recommendations,
            start,
            page_size=PAGE_SIZE,
        )

        if not shown:
            exit_program()

        if first_page:
            cli.print_personalised_message()
            first_page = False

        choice = cli.prompt_after_table()

        if choice == "2":
            exit_program()

        start += PAGE_SIZE


def exit_program():
    print("Goodbye.")
    sys.exit(0)


if __name__ == "__main__":
    main()
