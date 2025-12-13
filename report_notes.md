Dataset - Movies
https://www.kaggle.com/datasets/garymk/movielens-25m-dataset



SVD with biases
https://zhangyk8.github.io/teaching/file_spring2018/Improving_regularized_singular_value_decomposition_for_collaborative_filtering.pdf

“We propose extending the set of predictors with the following methods: addition of biases to the regularized SVD …”


NCF
https://arxiv.org/pdf/1708.05031
“When it comes to model the key factor in collaborative filtering – the interaction between user and item features – they still resorted to matrix factorization and applied an inner product on the latent features of users and items. By replacing the inner product with a neural architecture that can learn an arbitrary function from data, we present a general framework named NCF.”



Approach - CF
- Conventional, straightforward, and the movielens dataset (consists of movies.csv and ratings.csv), is formatted in a way to make CF approach very natural
- Item metadata isn't super content rich -> not the best for CBF methods. (Movies are described by singular tags with relevance scores)
    - Although this means a hybrid apporach is possible




- Very naturally extendable to a NCF (neural collaborative feature) for part 2.







Two complementary methods are employed to evaluate the recommender systems:

First, we have, Rating Prediction Quality:
This assesses how accurately the model predicts explicit ratings. We evaluate this using Root Mean Squared Error (RMSE), where lower values indicate a more accurate rating prediction.

Secondly, we have, Recommendation Quality:
This evaluates whether the system successfully ranks relevant items near the top of a user’s recommendation list. Specifically, we check whether previously hidden items appear within the top-K recommendations.

To obtain these metrics, we adopt a held-out evaluation procedure: where a subset of known user–item ratings is temporarily removed from the dataset, and each model is then trained on the remaining interactions and subsequently asked to predict the withheld ratings.

