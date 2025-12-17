import pickle
from pathlib import Path
from collections import Counter, defaultdict
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)

_READY = False

# shared mapping / titles
user_map = None
reverse_item_map = None
movieid_to_title = None

num_users = None
num_items = None
global_mean = None

# features
user_gender = None
user_age = None
user_occupation = None

# item genres
num_genres = None
item_genre_matrix = None

# model
device = None
ncf_model = None

# training triplets
train_triplets = None
val_triplets = None
test_triplets = None

user_rated_items = None


def _paths():
    here = Path(__file__).resolve().parent
    dataset = (here / ".." / "dataset").resolve()
    return {
        "ratings": dataset / "ratings.dat",
        "movies": dataset / "movies.dat",
        "users": dataset / "users.dat",
    }


class NCFHybrid(nn.Module):
    def __init__(
        self,
        num_users,
        num_items,
        num_genders,
        max_age_code,
        num_occupations,
        num_genres,
        item_genre_matrix,
        emb_dim=16,
        hidden_dim=64,
        genre_emb_dim=8,
        global_mean=3.5,
    ):
        super().__init__()

        self.user_emb = nn.Embedding(num_users, emb_dim)

        self.gender_emb = nn.Embedding(num_genders, 2)
        self.age_emb = nn.Embedding(max_age_code + 1, 2)
        self.occ_emb = nn.Embedding(num_occupations, 4)

        user_vec_dim = emb_dim + 2 + 2 + 4
        self.user_mlp = nn.Sequential(
            nn.Linear(user_vec_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, emb_dim),
        )

        self.item_emb = nn.Embedding(num_items, emb_dim)
        self.genre_emb = nn.Embedding(num_genres, genre_emb_dim)

        self.register_buffer(
            "item_genre_multi_hot",
            torch.tensor(item_genre_matrix, dtype=torch.float32),
        )

        item_vec_dim = emb_dim + genre_emb_dim
        self.item_mlp = nn.Sequential(
            nn.Linear(item_vec_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, emb_dim),
        )

        self.interaction_mlp = nn.Sequential(
            nn.Linear(emb_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

        self.global_bias = nn.Parameter(
            torch.tensor(float(global_mean), dtype=torch.float32)
        )

    def forward(self, user_idx, item_idx, gender, age, occupation):
        user_emb = self.user_emb(user_idx)
        gender_emb = self.gender_emb(gender)
        age_emb = self.age_emb(age)
        occ_emb = self.occ_emb(occupation)

        u_concat = torch.cat([user_emb, gender_emb, age_emb, occ_emb], dim=-1)
        u_repr = self.user_mlp(u_concat)

        item_emb = self.item_emb(item_idx)
        g_multi = self.item_genre_multi_hot[item_idx]
        genre_vec = g_multi @ self.genre_emb.weight

        i_concat = torch.cat([item_emb, genre_vec], dim=-1)
        i_repr = self.item_mlp(i_concat)

        x = torch.cat([u_repr, i_repr], dim=-1)
        out = self.interaction_mlp(x).squeeze(-1) + self.global_bias

        return torch.clamp(out, 1.0, 5.0)


class NCFDataset(Dataset):
    def __init__(self, triplets):
        self.users = np.array([int(u) for (u, _, _) in triplets], dtype=np.int64)
        self.items = np.array([int(i) for (_, i, _) in triplets], dtype=np.int64)
        self.ratings = np.array([float(r) for (_, _, r) in triplets], dtype=np.float32)

    def __len__(self):
        return len(self.ratings)

    def __getitem__(self, idx):
        u = int(self.users[idx])
        i = int(self.items[idx])
        r = float(self.ratings[idx])

        g = int(user_gender[u])
        a = int(user_age[u])
        o = int(user_occupation[u])

        return (
            torch.tensor(u, dtype=torch.long),
            torch.tensor(i, dtype=torch.long),
            torch.tensor(g, dtype=torch.long),
            torch.tensor(a, dtype=torch.long),
            torch.tensor(o, dtype=torch.long),
            torch.tensor(r, dtype=torch.float32),
        )


def initialise(sample_n=100000):
    global _READY
    global user_map, reverse_item_map, movieid_to_title
    global num_users, num_items, global_mean
    global user_gender, user_age, user_occupation
    global num_genres, item_genre_matrix
    global device, ncf_model
    global train_triplets, val_triplets, test_triplets, user_rated_items

    if _READY:
        return

    if _load_cache():
        return

    if not _load_shared_preprocessed():
        raise RuntimeError(
            "Missing cache/preprocessed_shared.pkl. "
            "Run hybrid_engine once with ../dataset present to generate it."
        )

    p = _paths()

    # ratings_df = pd.read_csv(
    #     p["ratings"],
    #     sep="::",
    #     engine="python",
    #     encoding="latin-1",
    #     names=["userId", "movieId", "rating", "timestamp"],
    # )
    # movies_df = pd.read_csv(
    #     p["movies"],
    #     sep="::",
    #     engine="python",
    #     encoding="latin-1",
    #     names=["movieId", "title", "genres"],
    # )
    # movieid_to_title = dict(zip(movies_df["movieId"], movies_df["title"]))

    # # weighted sampling (same as RS1)
    # user_counts_full = ratings_df["userId"].value_counts()
    # DENSE_THRESHOLD_FULL = 100

    # def user_weight(u):
    #     c = user_counts_full[u]
    #     base = np.log1p(c)
    #     return base * 100 if c >= DENSE_THRESHOLD_FULL else base

    # weights = ratings_df["userId"].map(user_weight)
    # ratings_sample = ratings_df.sample(
    #     n=sample_n,
    #     weights=weights,
    #     random_state=0,
    # ).reset_index(drop=True)

    # # mappings
    # user_map = {old: new for new, old in enumerate(ratings_sample["userId"].unique())}
    # ratings_sample["user_idx"] = ratings_sample["userId"].map(user_map)

    # item_map = {old: new for new, old in enumerate(ratings_sample["movieId"].unique())}
    # ratings_sample["item_idx"] = ratings_sample["movieId"].map(item_map)
    # reverse_item_map = {new: old for old, new in item_map.items()}

    # num_users = int(ratings_sample["user_idx"].nunique())
    # num_items = int(ratings_sample["item_idx"].nunique())

    # triplets = [
    #     (int(u), int(i), float(r))
    #     for u, i, r in zip(
    #         ratings_sample["user_idx"],
    #         ratings_sample["item_idx"],
    #         ratings_sample["rating"],
    #     )
    # ]

    # # ---------- Dense users split ----------
    # user_counts_all = Counter(u for u, _, _ in triplets)
    # DENSE_THRESHOLD = 20
    # dense_users = {u for u, c in user_counts_all.items() if c >= DENSE_THRESHOLD}

    # dense_triplets = [t for t in triplets if t[0] in dense_users]
    # random.shuffle(dense_triplets)

    # n_dense = len(dense_triplets)
    # train_end = int(0.6 * n_dense)
    # val_end = int(0.8 * n_dense)

    # train_triplets = dense_triplets[:train_end]
    # val_triplets = dense_triplets[train_end:val_end]  # optional, may not use
    # test_triplets = dense_triplets[val_end:]

    _build_features_from_dataset()

    # global_mean = float(np.mean([r for _, _, r in train_triplets]))

    # # item genres matrix
    # all_genres = set()
    # for g_str in movies_df["genres"]:
    #     for g in str(g_str).split("|"):
    #         g = g.strip()

    #         if g and g != "(no genres listed)":
    #             all_genres.add(g)

    # all_genres = sorted(all_genres)
    # genre_to_idx = {g: idx for idx, g in enumerate(all_genres)}
    # num_genres = len(all_genres)

    # item_genre_matrix = np.zeros((num_items, num_genres), dtype=np.float32)
    # movie_genres_map = dict(zip(movies_df["movieId"], movies_df["genres"]))

    # for internal_i in range(num_items):
    #     movie_id = reverse_item_map[internal_i]
    #     g_str = movie_genres_map.get(movie_id, "")

    #     for g in str(g_str).split("|"):
    #         g = g.strip()

    #         if g in genre_to_idx:
    #             item_genre_matrix[internal_i, genre_to_idx[g]] = 1.0

    # # user features from users.dat
    # users_raw = pd.read_csv(
    #     p["users"],
    #     sep="::",
    #     engine="python",
    #     names=["user_id", "gender", "age", "occupation", "zipcode"],
    # )

    # users_df = users_raw.copy()
    # users_df["gender"] = users_df["gender"].map({"M": 0, "F": 1}).astype(np.int64)
    # users_df["age"] = users_df["age"].astype(np.int64)
    # users_df["occupation"] = users_df["occupation"].astype(np.int64)

    # users_df["user_idx"] = users_df["user_id"].map(user_map)
    # users_df = users_df.dropna(subset=["user_idx"]).copy()
    # users_df["user_idx"] = users_df["user_idx"].astype(int)

    # feat = users_df[["user_idx", "gender", "age", "occupation"]].sort_values("user_idx")
    # feat = feat.set_index("user_idx").reindex(range(num_users))

    # if feat.isna().sum().sum() != 0:
    #     raise RuntimeError("Missing user features after reindexing; mapping mismatch.")

    # user_gender = feat["gender"].values.astype(np.int64)
    # user_age = feat["age"].values.astype(np.int64)
    # user_occupation = feat["occupation"].values.astype(np.int64)

    # model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_genders = int(len(np.unique(user_gender)))
    max_age_code = int(user_age.max())
    num_occupations = int(user_occupation.max() + 1)

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
        global_mean=global_mean,
    ).to(device)

    # training
    train_dataset = NCFDataset(train_triplets)
    train_loader = DataLoader(train_dataset, batch_size=1024, shuffle=True)

    criterion = nn.MSELoss()
    optimiser = torch.optim.Adam(ncf_model.parameters(), lr=0.001, weight_decay=0.00001)

    ncf_model.train()
    num_epochs = 15
    for epoch in range(1, num_epochs + 1):
        running = 0
        n = 0
        for u, i, g, a, o, r in train_loader:
            u, i, g, a, o, r = (
                u.to(device),
                i.to(device),
                g.to(device),
                a.to(device),
                o.to(device),
                r.to(device),
            )

            optimiser.zero_grad()
            pred = ncf_model(u, i, g, a, o)
            loss = criterion(pred, r)
            loss.backward()
            optimiser.step()

            running += loss.item() * r.size(0)
            n += r.size(0)

        if epoch in {1, 5, 10, 15}:
            train_rmse = float(np.sqrt(running / n))
            print(f"[NCF init] Epoch {epoch}/{num_epochs} | train RMSE: {train_rmse}")

    # exclude already-rated (train)
    user_rated_items = defaultdict(set)
    for u, i, _ in train_triplets:
        user_rated_items[u].add(i)

    _READY = True

    _save_cache()
    # print("[Hybrid] Saved cache and preprocessed data.")


def user_exists(user_id):
    if not _READY:
        raise RuntimeError("Call initialise() first.")
    try:
        uid = int(user_id)
    except ValueError:
        return False
    return uid in user_map


def random_user_id():
    if not _READY:
        raise RuntimeError("Call initialise() first.")
    return str(random.choice(list(user_map.keys())))


# Returns list[(title, predicted_rating)] using NCF
def recommend(user_id, k=10):
    if not _READY:
        raise RuntimeError("Call initialise() first.")

    u_old = int(user_id)
    u = user_map[u_old]
    rated = user_rated_items.get(u, set())

    candidates = [i for i in range(num_items) if i not in rated]
    if not candidates:
        return []

    ncf_model.eval()

    gender = int(user_gender[u])
    age = int(user_age[u])
    occupation = int(user_occupation[u])

    scores = []
    batch_size = 2048

    with torch.no_grad():

        for start in range(0, len(candidates), batch_size):

            # Select a batch of candidate items to score
            items_batch = candidates[start : start + batch_size]
            B = len(items_batch)

            user_batch = torch.full((B,), u, device=device)
            gender_batch = torch.full((B,), gender, device=device)
            age_batch = torch.full((B,), age, device=device)
            occupation_batch = torch.full((B,), occupation, device=device)

            item_batch = torch.tensor(items_batch, device=device)

            predictions = ncf_model(
                user_batch,
                item_batch,
                gender_batch,
                age_batch,
                occupation_batch,
            )

            for score, item_idx in zip(predictions.cpu().numpy(), items_batch):
                scores.append((float(score), item_idx))

    scores.sort(reverse=True, key=lambda x: x[0])
    top = scores[:k]

    out = []
    for s, i in top:
        movie_id = reverse_item_map[i]
        title = movieid_to_title.get(movie_id, f"(movieId={movie_id})")
        out.append((title, float(s)))

    return out


def _cache_dir():
    here = Path(__file__).resolve().parent
    dir = (here / "cache").resolve()
    dir.mkdir(parents=True, exist_ok=True)
    return dir


def _cache_paths():
    dir = _cache_dir()
    return {
        "pt": dir / "ncf_model.pt",
        "npz": dir / "ncf_arrays.npz",
        "pkl": dir / "ncf_meta.pkl",
    }


def _save_cache():
    paths = _cache_paths()

    # torch weights
    torch.save(ncf_model.state_dict(), paths["pt"])

    # numpy arrays needed at runtime
    np.savez_compressed(
        paths["npz"],
        item_genre_matrix=item_genre_matrix,
        user_gender=user_gender,
        user_age=user_age,
        user_occupation=user_occupation,
        global_mean=np.array([global_mean], dtype=np.float32),
    )

    meta = {
        "user_map": user_map,
        "reverse_item_map": reverse_item_map,
        "movieid_to_title": movieid_to_title,
        "num_users": num_users,
        "num_items": num_items,
        "num_genres": num_genres,
        "user_rated_items": {u: list(s) for u, s in user_rated_items.items()},
    }
    with open(paths["pkl"], "wb") as f:
        pickle.dump(meta, f)


def _load_cache():
    global _READY
    global user_map, reverse_item_map, movieid_to_title
    global num_users, num_items, num_genres, global_mean
    global item_genre_matrix, user_gender, user_age, user_occupation
    global device, ncf_model, user_rated_items

    paths = _cache_paths()
    if not (paths["pt"].exists() and paths["npz"].exists() and paths["pkl"].exists()):
        return False

    try:
        with open(paths["pkl"], "rb") as f:
            meta = pickle.load(f)

        user_map = meta["user_map"]
        reverse_item_map = meta["reverse_item_map"]
        movieid_to_title = meta["movieid_to_title"]
        num_users = meta["num_users"]
        num_items = meta["num_items"]
        num_genres = meta["num_genres"]
        user_rated_items = defaultdict(
            set, {u: set(v) for u, v in meta["user_rated_items"].items()}
        )

        arrays = np.load(paths["npz"], allow_pickle=False)
        item_genre_matrix = arrays["item_genre_matrix"]
        user_gender = arrays["user_gender"]
        user_age = arrays["user_age"]
        user_occupation = arrays["user_occupation"]
        global_mean = float(arrays["global_mean"][0])

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Recreate model with same dimensions, then load weights
        num_genders = int(len(np.unique(user_gender)))
        max_age_code = int(user_age.max())
        num_occupations = int(user_occupation.max() + 1)

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
            global_mean=global_mean,
        ).to(device)

        ncf_model.load_state_dict(torch.load(paths["pt"], map_location=device))
        ncf_model.eval()

        _READY = True
        print("[NCF] Loaded cached model/artifacts.")
        return True

    except Exception as e:
        print(f"[NCF] Cache load failed, will retrain. Reason: {e}")
        return False


def _build_features_from_dataset():
    global movieid_to_title
    global num_users, num_items, num_genres, global_mean
    global item_genre_matrix, user_gender, user_age, user_occupation

    p = _paths()

    movies_df = pd.read_csv(
        p["movies"],
        sep="::",
        engine="python",
        encoding="latin-1",
        names=["movieId", "title", "genres"],
    )
    movieid_to_title = dict(zip(movies_df["movieId"], movies_df["title"]))

    num_users = len(user_map)
    num_items = len(reverse_item_map)

    global_mean = float(np.mean([r for _, _, r in train_triplets]))

    all_genres = set()
    for g_str in movies_df["genres"]:
        for g in str(g_str).split("|"):
            g = g.strip()
            if g and g != "(no genres listed)":
                all_genres.add(g)

    all_genres = sorted(all_genres)
    genre_to_idx = {g: idx for idx, g in enumerate(all_genres)}
    num_genres = len(all_genres)

    item_genre_matrix = np.zeros((num_items, num_genres), dtype=np.float32)
    movie_genres_map = dict(zip(movies_df["movieId"], movies_df["genres"]))

    for internal_i in range(num_items):
        movie_id = reverse_item_map[internal_i]
        g_str = movie_genres_map.get(movie_id, "")

        for g in str(g_str).split("|"):
            g = g.strip()

            if g in genre_to_idx:
                item_genre_matrix[internal_i, genre_to_idx[g]] = 1.0

    # users.dat -> features aligned to user_idx
    users_raw = pd.read_csv(
        p["users"],
        sep="::",
        engine="python",
        names=["user_id", "gender", "age", "occupation", "zipcode"],
    )

    users_df = users_raw.copy()
    users_df["gender"] = users_df["gender"].map({"M": 0, "F": 1}).astype(np.int64)
    users_df["age"] = users_df["age"].astype(np.int64)
    users_df["occupation"] = users_df["occupation"].astype(np.int64)

    users_df["user_idx"] = users_df["user_id"].map(user_map)
    users_df = users_df.dropna(subset=["user_idx"]).copy()
    users_df["user_idx"] = users_df["user_idx"].astype(int)

    feat = users_df[["user_idx", "gender", "age", "occupation"]].sort_values("user_idx")
    feat = feat.set_index("user_idx").reindex(range(num_users))

    if feat.isna().sum().sum() != 0:
        raise RuntimeError("Missing user features after reindexing; mapping mismatch.")

    user_gender = feat["gender"].values.astype(np.int64)
    user_age = feat["age"].values.astype(np.int64)
    user_occupation = feat["occupation"].values.astype(np.int64)


def _shared_prep_path():
    return _cache_dir() / "preprocessed_shared.pkl"


def _load_shared_preprocessed():
    global user_map, reverse_item_map
    global train_triplets, val_triplets, test_triplets

    path = _shared_prep_path()
    if not path.exists():
        return False

    with open(path, "rb") as f:
        payload = pickle.load(f)

    user_map = payload["user_map"]
    reverse_item_map = payload["reverse_item_map"]
    train_triplets = payload["train_triplets"]
    val_triplets = payload["val_triplets"]
    test_triplets = payload["test_triplets"]
    return True
