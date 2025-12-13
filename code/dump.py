import pandas as pd
import numpy as np
import random

random.seed(0)
np.random.seed(0)

# Load MovieLens 1M ratings
ratings_df = pd.read_csv(
    "../dataset/ratings.dat",
    sep="::",
    engine="python",  # needed because the separator is multi-character
    encoding="latin-1",  # needed due to accented characters
    names=["userId", "movieId", "rating", "timestamp"],
)

# Load MovieLens 1M movies (for later CBF)
movies_df = pd.read_csv(
    "../dataset/movies.dat",
    sep="::",
    engine="python",
    encoding="latin-1",
    names=["movieId", "title", "genres"],
)

print(ratings_df.head())
print(movies_df.head())
print("Full ratings shape:", ratings_df.shape)

from collections import Counter
import numpy as np

# Count ratings per user on the FULL ratings_df (before sampling)
user_counts = ratings_df["userId"].value_counts()

DENSE_THRESHOLD = 100  # or whatever you used later


def user_weight(u):
    c = user_counts[u]
    base = np.log1p(c)  # keep your original shape
    # Boost dense users
    if c >= DENSE_THRESHOLD:
        return base * 100  # 20x more likely than sparse users
    else:
        return base


weights = ratings_df["userId"].map(user_weight)

ratings_sample = ratings_df.sample(
    n=100_000,
    weights=weights,
    random_state=0,
).reset_index(drop=True)

from collections import Counter

user_counts_sample = Counter(ratings_sample["userId"])

print("Users with <5 ratings:", sum(1 for u, c in user_counts_sample.items() if c < 5))
print(
    "Users with 5-20 ratings:",
    sum(1 for u, c in user_counts_sample.items() if 5 <= c <= 20),
)
print(
    "Users with >20 ratings:", sum(1 for u, c in user_counts_sample.items() if c > 20)
)
print("Max ratings by a user:", max(user_counts_sample.values()))
print(
    "Mean ratings per user:", sum(user_counts_sample.values()) / len(user_counts_sample)
)

from collections import Counter

# Count how many ratings each item received in the sampled dataset
item_counts_sample = Counter(ratings_sample["movieId"])

print(
    "Items with <5 ratings:", sum(1 for item, c in item_counts_sample.items() if c < 5)
)

print(
    "Items with 5-20 ratings:",
    sum(1 for item, c in item_counts_sample.items() if 5 <= c <= 20),
)

print(
    "Items with >20 ratings:",
    sum(1 for item, c in item_counts_sample.items() if c > 20),
)

print("Max ratings for an item:", max(item_counts_sample.values()))

print(
    "Mean ratings per item:", sum(item_counts_sample.values()) / len(item_counts_sample)
)

# Map user & item IDs to dense indices
user_map = {
    old_id: new_id for new_id, old_id in enumerate(ratings_sample["userId"].unique())
}
ratings_sample["user_idx"] = ratings_sample["userId"].map(user_map)

item_map = {
    old_id: new_id for new_id, old_id in enumerate(ratings_sample["movieId"].unique())
}
ratings_sample["item_idx"] = ratings_sample["movieId"].map(item_map)

reverse_item_map = {new: old for old, new in item_map.items()}

triplets = [
    (int(user), int(item), float(rating))
    for user, item, rating in zip(
        ratings_sample["user_idx"],
        ratings_sample["item_idx"],
        ratings_sample["rating"],
    )
]

triplets[:10]


num_users = ratings_sample["userId"].nunique()
num_items = ratings_sample["movieId"].nunique()
num_interactions = len(ratings_sample)

density = num_interactions / (num_users * num_items)
sparsity = 1.0 - density

print(f"Users: {num_users}")
print(f"Items: {num_items}")
print(f"Interactions: {num_interactions}")
print(f"Density:  {density:.6f}")
print(f"Sparsity: {sparsity:.6f}")


from collections import Counter
import random

# Count #ratings per user on the full 100K
user_counts_all = Counter(u for u, _, _ in triplets)

DENSE_THRESHOLD = 20

dense_users = {u for u, c in user_counts_all.items() if c >= DENSE_THRESHOLD}
cold_users = {u for u, c in user_counts_all.items() if c < DENSE_THRESHOLD}

print("Total users:", len(user_counts_all))
print("Dense users (>=20):", len(dense_users))
print("Cold users (<20):", len(cold_users))

# Split triplets into dense and cold groups
dense_triplets = [t for t in triplets if t[0] in dense_users]
cold_triplets = [t for t in triplets if t[0] in cold_users]

print("Dense triplets:", len(dense_triplets))
print("Cold triplets :", len(cold_triplets))


dense_triplets_shuffled = dense_triplets.copy()
random.shuffle(dense_triplets_shuffled)

n_dense = len(dense_triplets_shuffled)

train_end = int(0.6 * n_dense)
val_end = int(0.8 * n_dense)  # 60% train, 20% val, 20% test

train_dense_triplets = dense_triplets_shuffled[:train_end]
val_dense_triplets = dense_triplets_shuffled[train_end:val_end]
test_dense_triplets = dense_triplets_shuffled[val_end:]

print("Train (dense):", len(train_dense_triplets))
print("Val   (dense):", len(val_dense_triplets))
print("Test  (dense):", len(test_dense_triplets))

train_triplets = train_dense_triplets + val_dense_triplets


num_users = ratings_sample["user_idx"].nunique()
num_items = ratings_sample["item_idx"].nunique()

num_factors = 20
learning_rate = 0.02
reg = 0.05
num_epochs = 10

# Global mean from dense TRAIN only
global_mean = np.mean([r for (_, _, r) in train_triplets])

user_bias = np.zeros(num_users, dtype=np.float32)
item_bias = np.zeros(num_items, dtype=np.float32)
user_factors = 0.1 * np.random.randn(num_users, num_factors).astype(np.float32)
item_factors = 0.1 * np.random.randn(num_items, num_factors).astype(np.float32)

print("Global mean (dense train):", global_mean)


def svd_score(u, i):
    return (
        global_mean
        + user_bias[u]
        + item_bias[i]
        + np.dot(user_factors[u], item_factors[i])
    )


def rmse(triplets):
    se = 0.0
    for u, i, r in triplets:
        pred = svd_score(u, i)
        se += (r - pred) ** 2
    return np.sqrt(se / len(triplets))


def train_svd_sgd(train_triplets, val_triplets, num_epochs, learning_rate, reg):
    global user_bias, item_bias, user_factors, item_factors

    best_val_rmse = None

    for epoch in range(num_epochs):
        random.shuffle(train_triplets)

        squared_e_train = 0.0

        for u, i, r in train_triplets:
            # Current prediction
            pred = (
                global_mean
                + user_bias[u]
                + item_bias[i]
                + np.dot(user_factors[u], item_factors[i])
            )
            err = r - pred
            squared_e_train += err**2

            # Cache old factors so we don't use partially updated values
            pu = user_factors[u].copy()
            qi = item_factors[i].copy()

            # Update biases
            user_bias[u] += learning_rate * (err - reg * user_bias[u])
            item_bias[i] += learning_rate * (err - reg * item_bias[i])

            # Update latent factors
            user_factors[u] += learning_rate * (err * qi - reg * pu)
            item_factors[i] += learning_rate * (err * pu - reg * qi)

        train_rmse = np.sqrt(squared_e_train / len(train_triplets))
        val_rmse = rmse(val_triplets)

        if best_val_rmse is None or val_rmse < best_val_rmse:
            best_val_rmse = val_rmse

        # print(
        #     f"Epoch {epoch+1}/{num_epochs} "
        #     f"- Train RMSE: {train_rmse} | Val RMSE: {val_rmse}"
        # )

    print("Best validation RMSE:", best_val_rmse)
    return best_val_rmse


# after tuning

best_val = train_svd_sgd(
    train_triplets=train_triplets,
    val_triplets=val_dense_triplets,
    num_epochs=num_epochs,
    learning_rate=learning_rate,
    reg=reg,
)


def test_svd_config(
    num_factors=20, learning_rate=0.01, reg=0.02, num_epochs=5, verbose=False
):
    global user_bias, item_bias, user_factors, item_factors, global_mean

    if verbose:
        print("=" * 50)
        print(f"Testing config:")
        print(f"  num_factors   = {num_factors}")
        print(f"  learning_rate = {learning_rate}")
        print(f"  reg           = {reg}")
        print(f"  epochs        = {num_epochs}")
        print("=" * 50)

    # Re-init parameters fresh for each run
    global_mean = np.mean([r for (_, _, r) in train_triplets])
    user_bias = np.zeros(num_users, dtype=np.float32)
    item_bias = np.zeros(num_items, dtype=np.float32)
    user_factors = 0.1 * np.random.randn(num_users, num_factors).astype(np.float32)
    item_factors = 0.1 * np.random.randn(num_items, num_factors).astype(np.float32)

    # Train SVD (you may want verbose=False in train_svd_sgd too)
    best_val_rmse = train_svd_sgd(
        train_triplets=train_triplets,
        val_triplets=val_dense_triplets,
        num_epochs=num_epochs,
        learning_rate=learning_rate,
        reg=reg,
    )

    return {
        "k": num_factors,
        "lr": learning_rate,
        "reg": reg,
        "epochs": num_epochs,
        "rmse": best_val_rmse,
    }


results = []

for k in [10, 20, 30, 40]:
    for lr in [0.02, 0.01, 0.005]:
        for reg in [0.005, 0.02, 0.05]:
            res = test_svd_config(
                num_factors=k,
                learning_rate=lr,
                reg=reg,
                num_epochs=10,
                verbose=False,  # no spam
            )
            results.append(res)

            # Sort by RMSE ascending
results_sorted = sorted(results, key=lambda x: x["rmse"])
best = results_sorted[0]

print("=== Best configuration found ===")
print(f"k       = {best['k']}")
print(f"lr      = {best['lr']}")
print(f"reg     = {best['reg']}")
print(f"epochs  = {best['epochs']}")
print(f"RMSE    = {best['rmse']:.4f}")


# Building CBF model

# Build genre vocabulary from movies_df
all_genres = set()

for g_str in movies_df["genres"]:
    for g in str(g_str).split("|"):
        g = g.strip()
        # print(g)

        if g and g != "(no genres listed)":
            all_genres.add(g)

all_genres = sorted(all_genres)
genre_to_idx = {g: idx for idx, g in enumerate(all_genres)}
num_genres = len(all_genres)

print("Number of genres:", num_genres)
print("Genres:", all_genres)


# Build item_genre_matrix aligned with internal item_idx
item_genre_matrix = np.zeros((num_items, num_genres), dtype=np.float32)

# Map movieId -> genres string for quick lookup
movie_genres_map = dict(zip(movies_df["movieId"], movies_df["genres"]))

for internal_i in range(num_items):
    movie_id = reverse_item_map[internal_i]  # original MovieID
    g_str = movie_genres_map.get(movie_id, "")

    for g in str(g_str).split("|"):
        g = g.strip()

        if g in genre_to_idx:
            item_genre_matrix[internal_i, genre_to_idx[g]] = 1.0  # one hot-encoding

            from collections import defaultdict

# Build user_ratings from TRAIN
user_ratings = defaultdict(list)
for u, i, r in train_triplets:
    user_ratings[u].append((i, r))

# User profiles: weighted average of genres of positively rated items
user_profiles = np.zeros((num_users, num_genres), dtype=np.float32)

for u, rated_list in user_ratings.items():
    profile = np.zeros(num_genres, dtype=np.float32)
    total_weight = 0.0


# Compute mean rating per user (using TRAIN only)
user_mean = {}
for u, rated_list in user_ratings.items():
    ratings = [r for (_, r) in rated_list]
    user_mean[u] = np.mean(ratings)

    for i, r in rated_list:
        # Emphasize movies greater than the average
        w = max(r - user_mean[u], 0.0)  # DONT NORMALISE FOR STRONGER HARSH SIGNALS

        if w <= 0:
            continue

        profile += w * item_genre_matrix[i]
        total_weight += w

    if total_weight > 0:
        profile /= total_weight  # normalise
        user_profiles[u] = profile
    # else: stays zero


user_rated_items = defaultdict(set)
for u, i, r in train_triplets:
    user_rated_items[u].add(i)


def cbf_score(u, i):
    user_vec = user_profiles[u]
    item_vec = item_genre_matrix[i]

    # If user profile or item vector is all zeros, fall back to global mean
    norm_u = np.linalg.norm(user_vec)
    norm_i = np.linalg.norm(item_vec)
    if norm_u == 0.0 or norm_i == 0.0:
        return global_mean  # from SVD setup

    sim = np.dot(user_vec, item_vec) / (norm_u * norm_i)  # cosine similarity

    # For our non-negative vectors, sim is between 0 and 1.
    # Map similarity [0,1] to rating [1,5]:
    return 1.0 + 4.0 * sim


from collections import Counter

user_train_counts = Counter(u for u, _, _ in train_triplets)
item_train_counts = Counter(i for _, i, _ in train_triplets)

print("Example user counts:", list(user_train_counts.items())[:10])
print("Example item counts:", list(item_train_counts.items())[:10])


ITEM_MIN_CF = 5  # below this, CF is considered unreliable for the item
ALPHA = 0.9  # CF weight when item is NOT sparse


def hybrid_score(u, i):
    n_i = item_train_counts.get(i, 0)  # #ratings for item i in TRAIN

    cf_pred = svd_score(u, i)  # CF (SVD) prediction
    cbf_pred = cbf_score(u, i)  # CBF prediction

    # If the ITEM is sparse, trust only CBF (CF is unreliable here)
    if n_i < ITEM_MIN_CF:
        return cbf_pred

    # Otherwise: blend CF + CBF (CBF still helps in all other cases)
    return ALPHA * cf_pred + (1.0 - ALPHA) * cbf_pred


def rmse_for_scorer(triplets, score_fn):
    se = 0.0
    for u, i, r in triplets:
        pred = score_fn(u, i)
        se += (r - pred) ** 2
    return np.sqrt(se / len(triplets))


all_test_triplets = test_dense_triplets + cold_triplets

print("=== CF (SVD) ===")
rmse_dense_cf = rmse_for_scorer(test_dense_triplets, svd_score)
rmse_cold_cf = rmse_for_scorer(cold_triplets, svd_score)
rmse_all_cf = rmse_for_scorer(all_test_triplets, svd_score)
print(f"RMSE dense users : {rmse_dense_cf:.4f}")
print(f"RMSE cold users  : {rmse_cold_cf:.4f}")
print(f"RMSE all users   : {rmse_all_cf:.4f}")

print("\n=== CBF ===")
rmse_dense_cbf = rmse_for_scorer(test_dense_triplets, cbf_score)
rmse_cold_cbf = rmse_for_scorer(cold_triplets, cbf_score)
rmse_all_cbf = rmse_for_scorer(all_test_triplets, cbf_score)
print(f"RMSE dense users : {rmse_dense_cbf:.4f}")
print(f"RMSE cold users  : {rmse_cold_cbf:.4f}")
print(f"RMSE all users   : {rmse_all_cbf:.4f}")

print("\n=== Hybrid (CF+CBF) ===")
rmse_dense_h = rmse_for_scorer(test_dense_triplets, hybrid_score)
rmse_cold_h = rmse_for_scorer(cold_triplets, hybrid_score)
rmse_all_h = rmse_for_scorer(all_test_triplets, hybrid_score)
print(f"RMSE dense users : {rmse_dense_h:.4f}")
print(f"RMSE cold users  : {rmse_cold_h:.4f}")
print(f"RMSE all users   : {rmse_all_h:.4f}")
