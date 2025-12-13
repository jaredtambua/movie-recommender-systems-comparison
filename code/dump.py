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


ratings_sample["userId"].nunique()


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

ratings_sample["user_idx"].value_counts().head()
# ratings_sample["user_idx"].value_counts().min(), ratings_sample["user_idx"].value_counts().max()

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

train_triplets = train_dense_triplets

num_users = ratings_sample["user_idx"].nunique()
num_items = ratings_sample["item_idx"].nunique()

num_factors = 20
learning_rate = 0.02
reg = 0.05
num_epochs = 20

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

        print(
            f"Epoch {epoch+1}/{num_epochs} "
            f"- Train RMSE: {train_rmse} | Val RMSE: {val_rmse}"
        )

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
import numpy as np

# Build user_ratings from TRAIN
user_ratings = defaultdict(list)
for u, i, r in train_triplets:
    user_ratings[u].append((i, r))

user_profiles = np.zeros((num_users, num_genres), dtype=np.float32)

for u, rated_list in user_ratings.items():
    # mean rating for this user (TRAIN only)
    ratings = [r for (_, r) in rated_list]
    mu = float(np.mean(ratings))

    profile = np.zeros(num_genres, dtype=np.float32)
    total_weight = 0.0

    for i, r in rated_list:
        w = max(r - mu, 0.0)  # only above-mean contribute
        if w <= 0:
            continue
        profile += w * item_genre_matrix[i]
        total_weight += w

    if total_weight > 0:
        user_profiles[u] = profile / total_weight
    # else: leave as all zeros


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


# Serendipity measure


def build_user_ratings(triplets):
    user_ratings = defaultdict(list)
    for u, i, r in triplets:
        user_ratings[u].append((i, r))
    return user_ratings


def cosine_similarity(a, b):
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0.0 or nb == 0.0:
        return None
    return float(np.dot(a, b) / (na * nb))


def item_unexpectedness(u, i):
    """1 - cosine similarity between user profile and item genres."""
    u_vec = user_profiles[u]
    i_vec = item_genre_matrix[i]

    sim = cosine_similarity(u_vec, i_vec)
    if sim is None:
        return None
    return 1.0 - sim


def top_k_for_user(u, user_ratings, score_fn, top_k):
    """
    user_ratings[u] is a list of (item, true_rating) for that user in TEST.
    """
    scored = []
    for i, r_true in user_ratings[u]:
        pred = score_fn(u, i)
        scored.append((pred, i, r_true))

    scored.sort(reverse=True, key=lambda x: x[0])
    return scored[:top_k]


def serendipity_for_user(u, user_ratings, score_fn, top_k=10, rel_thresh=4.0):
    top = top_k_for_user(u, user_ratings, score_fn, top_k)

    num = 0.0
    den = 0

    for _, i, r_true in top:
        if r_true < rel_thresh:
            continue  # only count actually relevant items

        unexp = item_unexpectedness(u, i)
        if unexp is None:
            continue

        num += unexp
        den += 1

    if den == 0:
        return None

    return num / den


# Building NCF implementation -- RS2

import pandas as pd
import numpy as np

users_file = "../dataset/users.dat"

user_columns = ["user_id", "gender", "age", "occupation", "zipcode"]

users_raw = pd.read_csv(
    users_file,
    sep="::",
    engine="python",
    names=user_columns,
)

print(users_raw.shape)

users_raw.head()


# Make a working copy
users_df = users_raw.copy()

# Map gender to 0/1
users_df["gender"] = users_df["gender"].map({"M": 0, "F": 1}).astype(np.int64)

# Ensure age and occupation are integers
users_df["age"] = users_df["age"].astype(np.int64)
users_df["occupation"] = users_df["occupation"].astype(np.int64)

users_df.head()

# Add internal user_idx using the same mapping as CF/CBF
users_df["user_idx"] = users_df["user_id"].map(user_map)

# Keep only users that actually appear in ratings_sample
users_df = users_df.dropna(subset=["user_idx"]).copy()
users_df["user_idx"] = users_df["user_idx"].astype(int)

# For sanity: sort by user_idx and keep only what we need for RS2
user_features_df = users_df[["user_idx", "gender", "age", "occupation"]].sort_values(
    "user_idx"
)

print(
    "Num distinct internal users in features:", user_features_df["user_idx"].nunique()
)
print("num_users from earlier:", num_users)

user_features_df.head()


# Ensure we have one feature row per internal user_idx in order 0..num_users-1
user_features_df = user_features_df.set_index("user_idx").reindex(range(num_users))

assert user_features_df.isna().sum().sum() == 0

user_gender = user_features_df["gender"].values.astype(np.int64)
user_age = user_features_df["age"].values.astype(np.int64)
user_occupation = user_features_df["occupation"].values.astype(np.int64)

print("user_gender shape:", user_gender.shape)
print("user_age shape:", user_age.shape)
print("user_occupation shape:", user_occupation.shape)


import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

num_genders = int(len(np.unique(user_gender)))
max_age_code = int(user_age.max())  # age buckets like 1,18,25,...
num_occupations = int(user_occupation.max() + 1)

print("num_users:", num_users)
print("num_items:", num_items)
print("num_genders:", num_genders)
print("max_age_code:", max_age_code)
print("num_occupations:", num_occupations)


import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F


class NCFHybrid(nn.Module):
    def __init__(
        self,
        num_users,
        num_items,
        num_genders,
        max_age_code,
        num_occupations,
        num_genres,
        item_genre_matrix,  # numpy array (num_items, num_genres)
        emb_dim=16,
        hidden_dim=64,
        genre_emb_dim=8,
    ):
        super().__init__()

        # === User tower ===
        self.user_emb = nn.Embedding(num_users, emb_dim)

        self.gender_emb = nn.Embedding(num_genders, 2)
        self.age_emb = nn.Embedding(max_age_code + 1, 2)
        self.occ_emb = nn.Embedding(num_occupations, 4)

        user_vec_dim = emb_dim + 2 + 2 + 4
        self.user_mlp = nn.Sequential(
            nn.Linear(user_vec_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, emb_dim),  # back to item emb dim
        )

        # === Item tower ===
        self.item_emb = nn.Embedding(num_items, emb_dim)

        # Genre embeddings
        self.genre_emb = nn.Embedding(num_genres, genre_emb_dim)

        # Store the multi-hot genre indicators as a buffer (not trainable)
        self.register_buffer(
            "item_genre_multi_hot",
            torch.tensor(item_genre_matrix, dtype=torch.float32),
        )

        # Concatenated item vector: [item_emb | summed_genre_emb]
        item_vec_dim = emb_dim + genre_emb_dim
        self.item_mlp = nn.Sequential(
            nn.Linear(item_vec_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, emb_dim),  # final item repr
        )

        # === Interaction MLP (NeuMF-style) ===
        self.interaction_mlp = nn.Sequential(
            nn.Linear(emb_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

        self.global_bias = nn.Parameter(
            torch.tensor(float(global_mean), dtype=torch.float32)
        )

    def forward(self, user_idx, item_idx, gender, age, occupation):
        # --- user representation ---
        u_base = self.user_emb(user_idx)
        g = self.gender_emb(gender)
        a = self.age_emb(age)
        o = self.occ_emb(occupation)
        u_concat = torch.cat([u_base, g, a, o], dim=-1)
        u_repr = self.user_mlp(u_concat)  # [B, emb_dim]

        # --- item representation ---
        i_base = self.item_emb(item_idx)  # [B, emb_dim]

        # multi-hot genre indicators for these items: [B, num_genres]
        g_multi = self.item_genre_multi_hot[item_idx]

        # summed genre embedding per item: [B, genre_emb_dim]
        genre_vec = g_multi @ self.genre_emb.weight  # matmul

        i_concat = torch.cat([i_base, genre_vec], dim=-1)
        i_repr = self.item_mlp(i_concat)  # [B, emb_dim]

        # --- interaction ---
        x = torch.cat([u_repr, i_repr], dim=-1)
        out = self.interaction_mlp(x).squeeze(-1) + self.global_bias

        return torch.clamp(out, 1.0, 5.0)


ncf_model = NCFHybrid(
    num_users=num_users,
    num_items=num_items,
    num_genders=num_genders,
    max_age_code=max_age_code,
    num_occupations=num_occupations,
    num_genres=num_genres,
    item_genre_matrix=item_genre_matrix,
    emb_dim=16,
    hidden_dim=64,
    genre_emb_dim=8,
).to(device)


class NCFDataset(Dataset):
    """
    Each example: (user_idx, item_idx, gender, age, occupation, rating)
    """

    def __init__(self, triplets):
        self.users = np.array([int(u) for (u, _, _) in triplets], dtype=np.int64)
        self.items = np.array([int(i) for (_, i, _) in triplets], dtype=np.int64)
        self.ratings = np.array([float(r) for (_, _, r) in triplets], dtype=np.float32)

    def __len__(self):
        return len(self.ratings)

    def __getitem__(self, idx):
        u = self.users[idx]
        i = self.items[idx]
        r = self.ratings[idx]

        g = user_gender[u]
        a = user_age[u]
        o = user_occupation[u]

        return (
            torch.tensor(u, dtype=torch.long),
            torch.tensor(i, dtype=torch.long),
            torch.tensor(g, dtype=torch.long),
            torch.tensor(a, dtype=torch.long),
            torch.tensor(o, dtype=torch.long),
            torch.tensor(r, dtype=torch.float32),
        )


# Dense-only datasets (since SVD also trains on dense)
train_dataset_dense = NCFDataset(train_triplets)
val_dataset_dense = NCFDataset(val_dense_triplets)
val_dataset_all = NCFDataset(all_test_triplets)  # for overall eval


batch_size = 1024

train_loader = DataLoader(train_dataset_dense, batch_size=batch_size, shuffle=True)
val_loader_dense = DataLoader(val_dataset_dense, batch_size=batch_size, shuffle=False)
val_loader_all = DataLoader(val_dataset_all, batch_size=batch_size, shuffle=False)


criterion = nn.MSELoss()
optimiser = torch.optim.Adam(ncf_model.parameters(), lr=0.001, weight_decay=0.00001)


def eval_rmse(model, data_loader):
    model.eval()
    se = 0.0
    n = 0
    with torch.no_grad():
        for u, i, g, a, o, r in data_loader:
            u = u.to(device)
            i = i.to(device)
            g = g.to(device)
            a = a.to(device)
            o = o.to(device)
            r = r.to(device)

            preds = model(u, i, g, a, o)
            se += torch.sum((preds - r) ** 2).item()
            n += r.numel()
    return np.sqrt(se / n)


num_epochs = 20
best_val_rmse = None
best_state = None

for epoch in range(1, num_epochs + 1):
    ncf_model.train()
    running_loss = 0.0
    count = 0

    for u, i, g, a, o, r in train_loader:
        u = u.to(device)
        i = i.to(device)
        g = g.to(device)
        a = a.to(device)
        o = o.to(device)
        r = r.to(device)

        optimiser.zero_grad()
        preds = ncf_model(u, i, g, a, o)
        loss = criterion(preds, r)
        loss.backward()
        optimiser.step()

        running_loss += loss.item() * r.size(0)
        count += r.size(0)

    train_rmse = np.sqrt(running_loss / count)
    val_rmse = eval_rmse(ncf_model, val_loader_dense)

    print(
        f"Epoch {epoch:02d} | train RMSE: {train_rmse:.4f} | val (dense) RMSE: {val_rmse:.4f}"
    )

    if best_val_rmse is None or val_rmse < best_val_rmse:
        best_val_rmse = val_rmse
        best_state = ncf_model.state_dict()

print("Best dense-val RMSE (NCF):", best_val_rmse)
if best_state is not None:
    ncf_model.load_state_dict(best_state)


def ncf_score(u, i):
    ncf_model.eval()
    g = int(user_gender[u])
    a = int(user_age[u])
    o = int(user_occupation[u])

    with torch.no_grad():
        u_t = torch.tensor([u], dtype=torch.long, device=device)
        i_t = torch.tensor([i], dtype=torch.long, device=device)
        g_t = torch.tensor([g], dtype=torch.long, device=device)
        a_t = torch.tensor([a], dtype=torch.long, device=device)
        o_t = torch.tensor([o], dtype=torch.long, device=device)

        pred = ncf_model(u_t, i_t, g_t, a_t, o_t).item()

    return float(pred)


print("\n================ FINAL MODEL COMPARISON ================\n")

recommenders = [
    ("CF (SVD)", svd_score),
    ("CBF", cbf_score),
    ("Hybrid (SVD + CBF)", hybrid_score),
    ("NCF (user.dat enriched)", ncf_score),
]

header = f"{'Model':30} {'RMSE dense':>10} {'RMSE cold':>10} {'RMSE all':>10}"
print(header)
print("-" * len(header))

for name, fn in recommenders:
    rmse_dense = rmse_for_scorer(test_dense_triplets, fn)
    rmse_cold = rmse_for_scorer(cold_triplets, fn)
    rmse_all = rmse_for_scorer(all_test_triplets, fn)
    print(f"{name:30} {rmse_dense:10.4f} {rmse_cold:10.4f} {rmse_all:10.4f}")

print("\n========================================================\n")


def serendipity_for_scorer(triplets, score_fn, top_k=10, rel_thresh=4.0):
    # if you *already* have user_ratings, just pass it instead:
    user_ratings = build_user_ratings(triplets)

    user_ser = []
    for u in user_ratings.keys():
        s_u = serendipity_for_user(u, user_ratings, score_fn, top_k, rel_thresh)
        if s_u is not None:
            user_ser.append(s_u)

    if not user_ser:
        return 0.0

    return sum(user_ser) / len(user_ser)


print("\n=== Serendipity (Top-10, rel>=4.0) ===")
for name, fn in [
    ("CF (SVD)", svd_score),
    ("CBF", cbf_score),
    ("Hybrid", hybrid_score),
    ("NCF (user.dat enriched)", ncf_score),
]:

    s_dense = serendipity_for_scorer(test_dense_triplets, fn)
    s_cold = serendipity_for_scorer(cold_triplets, fn)
    s_all = serendipity_for_scorer(all_test_triplets, fn)
    print(
        f"{name} -> Serendipity dense: {s_dense:.4f} | "
        f"cold: {s_cold:.4f} | all: {s_all:.4f}"
    )
