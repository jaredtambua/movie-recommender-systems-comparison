import pickle
from pathlib import Path
from collections import Counter, defaultdict
import random
import numpy as np
import pandas as pd

random.seed(0)
np.random.seed(0)

# ========= Globals set during initialise() =========
_READY = False

ratings_sample = None
movies_df = None

user_map = None  # original userId -> user_idx
item_map = None  # original movieId -> item_idx
reverse_item_map = None  # item_idx -> original movieId

num_users = None
num_items = None

# CF bits
global_mean = None
user_bias = None
item_bias = None
user_factors = None
item_factors = None

# CBF bits
genre_to_idx = None
num_genres = None
item_genre_matrix = None
user_profiles = None

# hybrid bits
train_triplets = None
val_triplets = None
test_triplets = None

item_train_counts = None
ITEM_MIN_CF = 5
ALPHA = 0.9

# exclude already-rated
user_rated_items = None

# titles
movieid_to_title = None


def _paths():
    here = Path(__file__).resolve().parent
    dataset = (here / ".." / "dataset").resolve()
    return {
        "ratings": dataset / "ratings.dat",
        "movies": dataset / "movies.dat",
    }


# Loads data, samples ratings, trains SVD on dense users, builds CBF profiles, and prepares a hybrid scorer.
def initialise(sample_n=100000):
    global _READY
    global ratings_sample, movies_df
    global user_map, item_map, reverse_item_map
    global num_users, num_items
    global global_mean, user_bias, item_bias, user_factors, item_factors
    global genre_to_idx, num_genres, item_genre_matrix, user_profiles
    global train_triplets, val_triplets, test_triplets, item_train_counts, user_rated_items
    global movieid_to_title

    if _READY:
        return

    # Try load cache first
    if _load_cache():
        if not _load_shared_preprocessed():
            _write_shared_from_cache_only()
        return

    p = _paths()

    ratings_df = pd.read_csv(
        p["ratings"],
        sep="::",
        engine="python",
        encoding="latin-1",
        names=["userId", "movieId", "rating", "timestamp"],
    )
    movies_df = pd.read_csv(
        p["movies"],
        sep="::",
        engine="python",
        encoding="latin-1",
        names=["movieId", "title", "genres"],
    )

    movieid_to_title = dict(zip(movies_df["movieId"], movies_df["title"]))

    # ---------- Weighted sampling  ----------
    user_counts_full = ratings_df["userId"].value_counts()
    DENSE_THRESHOLD_WEIGHT = 100

    def user_weight(u):
        c = user_counts_full[u]
        base = np.log1p(c)
        return base * 100 if c >= DENSE_THRESHOLD_WEIGHT else base

    weights = ratings_df["userId"].map(user_weight)
    ratings_sample = ratings_df.sample(
        n=sample_n,
        weights=weights,
        random_state=0,
    ).reset_index(drop=True)

    # ---------- Map IDs to dense indices ----------
    user_map = {old: new for new, old in enumerate(ratings_sample["userId"].unique())}
    ratings_sample["user_idx"] = ratings_sample["userId"].map(user_map)

    item_map = {old: new for new, old in enumerate(ratings_sample["movieId"].unique())}
    ratings_sample["item_idx"] = ratings_sample["movieId"].map(item_map)
    reverse_item_map = {new: old for old, new in item_map.items()}

    num_users = int(ratings_sample["user_idx"].nunique())
    num_items = int(ratings_sample["item_idx"].nunique())

    triplets = [
        (int(u), int(i), float(r))
        for u, i, r in zip(
            ratings_sample["user_idx"],
            ratings_sample["item_idx"],
            ratings_sample["rating"],
        )
    ]

    # ---------- Dense users split ----------
    user_counts_all = Counter(u for u, _, _ in triplets)
    DENSE_THRESHOLD = 20
    dense_users = {u for u, c in user_counts_all.items() if c >= DENSE_THRESHOLD}

    dense_triplets = [t for t in triplets if t[0] in dense_users]
    sparse_triplets = [t for t in triplets if t[0] not in dense_users]  # <-- NEW

    random.shuffle(dense_triplets)

    n_dense = len(dense_triplets)
    train_end = int(0.6 * n_dense)
    val_end = int(0.8 * n_dense)

    train_triplets = dense_triplets[:train_end]
    val_triplets = dense_triplets[train_end:val_end]
    test_dense = dense_triplets[val_end:]

    test_triplets = test_dense + sparse_triplets

    random.shuffle(test_triplets)

    _save_shared_preprocessed(train_triplets, val_triplets, test_triplets)
    print(f"[Hyrbid] saved preprocessed dataset")

    # ---------- Train SVD (SGD) ----------
    num_factors = 20
    learning_rate = 0.02
    reg = 0.05
    num_epochs = 20

    global_mean = float(np.mean([r for (_, _, r) in train_triplets]))

    user_bias = np.zeros(num_users, dtype=np.float32)
    item_bias = np.zeros(num_items, dtype=np.float32)
    user_factors = (0.1 * np.random.randn(num_users, num_factors)).astype(np.float32)
    item_factors = (0.1 * np.random.randn(num_items, num_factors)).astype(np.float32)

    def svd_pred(u, i):
        return (
            global_mean
            + user_bias[u]
            + item_bias[i]
            + float(np.dot(user_factors[u], item_factors[i]))
        )

    def rmse(trips):
        squared_err = 0
        for u, i, r in trips:
            err = r - svd_pred(u, i)
            squared_err += err**2
        return float(np.sqrt(squared_err / len(trips)))

    for epoch in range(1, num_epochs + 1):
        random.shuffle(train_triplets)
        for u, i, r in train_triplets:
            pred = svd_pred(u, i)
            err = r - pred

            pu = user_factors[u].copy()
            qi = item_factors[i].copy()

            # stochastic gradient descent update for SVD parameters
            user_bias[u] += learning_rate * (err - reg * user_bias[u])
            item_bias[i] += learning_rate * (err - reg * item_bias[i])

            user_factors[u] += learning_rate * (err * qi - reg * pu)
            item_factors[i] += learning_rate * (err * pu - reg * qi)

        if epoch in {1, 5, 10, 20}:
            print(
                f"[Hybrid init] Epoch {epoch}/{num_epochs} | val RMSE: {rmse(val_triplets)}"
            )

    # ---------- Build CBF (genres) ----------
    all_genres = set()
    for g_str in movies_df["genres"]:
        for g in str(g_str).split("|"):
            g = g.strip()

            if g:
                all_genres.add(g)

    all_genres = sorted(all_genres)
    genre_to_idx = {g: idx for idx, g in enumerate(all_genres)}
    num_genres = len(all_genres)

    # multi-hot encoding matrix
    item_genre_matrix = np.zeros((num_items, num_genres), dtype=np.float32)
    movie_genres_map = dict(zip(movies_df["movieId"], movies_df["genres"]))

    for internal_i in range(num_items):
        movie_id = reverse_item_map[internal_i]
        g_str = movie_genres_map.get(movie_id, "")

        for g in str(g_str).split("|"):
            g = g.strip()

            if g in genre_to_idx:
                item_genre_matrix[internal_i, genre_to_idx[g]] = 1.0

    # ---------- User profiles from TRAIN ----------
    user_ratings = defaultdict(list)
    for u, i, r in train_triplets:
        user_ratings[u].append((i, r))

    user_profiles = np.zeros((num_users, num_genres), dtype=np.float32)

    for u, rated_list in user_ratings.items():
        ratings_u = [r for (_, r) in rated_list]
        mu = float(np.mean(ratings_u))

        profile = np.zeros(num_genres, dtype=np.float32)
        total_w = 0.0

        for i, r in rated_list:
            w = max(r - mu, 0.0)

            if w <= 0:
                continue

            profile += w * item_genre_matrix[i]
            total_w += w

        if total_w > 0:
            user_profiles[u] = profile / total_w

    # ---------- For hybrid weighting ----------
    item_train_counts = Counter(i for _, i, _ in train_triplets)

    # ---------- For excluding already-rated items ----------
    user_rated_items = defaultdict(set)
    for u, i, _ in train_triplets:
        user_rated_items[u].add(i)

    _READY = True
    _save_cache()
    # print("[Hybrid] Saved cache.")


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

    keys = list(user_map.keys())
    choice = random.choice(keys)
    user_id = str(choice)

    return user_id


def _svd_score(u, i):
    return float(
        global_mean
        + user_bias[u]
        + item_bias[i]
        + np.dot(user_factors[u], item_factors[i])
    )


def _cbf_score(user_id, item_id):
    user_profile = user_profiles[user_id]
    item_features = item_genre_matrix[item_id]

    user_norm = float(np.linalg.norm(user_profile))
    item_norm = float(np.linalg.norm(item_features))

    if user_norm == 0.0 or item_norm == 0.0:
        return float(global_mean)

    cos_sim = float(np.dot(user_profile, item_features) / (user_norm * item_norm))

    return 1.0 + 4.0 * cos_sim


def _hybrid_score(u, i):
    n_i = item_train_counts.get(i, 0)
    cf = _svd_score(u, i)
    cbf = _cbf_score(u, i)

    if n_i < ITEM_MIN_CF:
        return cbf
    return ALPHA * cf + (1.0 - ALPHA) * cbf


# Returns: list[(title, predicted_rating)]
def recommend(user_id, k=10):
    if not _READY:
        raise RuntimeError("Call initialise() first.")

    u_old = int(user_id)
    u = user_map[u_old]

    # candidate items = all items not rated by user in TRAIN
    rated = user_rated_items.get(u, set())
    candidates = [i for i in range(num_items) if i not in rated]

    # score candidates
    scored = []
    for i in candidates:
        s = _hybrid_score(u, i)
        scored.append((s, i))

    scored.sort(reverse=True, key=lambda x: x[0])
    top = scored[:k]

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
        "npz": dir / "hybrid_artifacts.npz",
        "pkl": dir / "hybrid_metadata.pkl",
    }


def _save_cache():
    paths = _cache_paths()

    # Save big numeric arrays
    np.savez_compressed(
        paths["npz"],
        global_mean=np.array([global_mean], dtype=np.float32),
        user_bias=user_bias,
        item_bias=item_bias,
        user_factors=user_factors,
        item_factors=item_factors,
        item_genre_matrix=item_genre_matrix,
        user_profiles=user_profiles,
    )

    # Save Python objects (mappings/titles/counts/sets)
    meta = {
        "user_map": user_map,
        "item_map": item_map,
        "reverse_item_map": reverse_item_map,
        "movieid_to_title": movieid_to_title,
        "item_train_counts": dict(item_train_counts),
        "user_rated_items": {u: list(s) for u, s in user_rated_items.items()},
        "num_users": num_users,
        "num_items": num_items,
        "ITEM_MIN_CF": ITEM_MIN_CF,
        "ALPHA": ALPHA,
    }
    with open(paths["pkl"], "wb") as f:
        pickle.dump(meta, f)


def _load_cache():
    global _READY
    global global_mean, user_bias, item_bias, user_factors, item_factors
    global item_genre_matrix, user_profiles
    global user_map, item_map, reverse_item_map, movieid_to_title
    global item_train_counts, user_rated_items, num_users, num_items
    global ITEM_MIN_CF, ALPHA

    paths = _cache_paths()
    if not paths["npz"].exists() or not paths["pkl"].exists():
        return False

    try:
        arrays = np.load(paths["npz"], allow_pickle=False)
        global_mean = float(arrays["global_mean"][0])
        user_bias = arrays["user_bias"]
        item_bias = arrays["item_bias"]
        user_factors = arrays["user_factors"]
        item_factors = arrays["item_factors"]
        item_genre_matrix = arrays["item_genre_matrix"]
        user_profiles = arrays["user_profiles"]

        with open(paths["pkl"], "rb") as f:
            metadata = pickle.load(f)

        user_map = metadata["user_map"]
        item_map = metadata["item_map"]
        reverse_item_map = metadata["reverse_item_map"]
        movieid_to_title = metadata["movieid_to_title"]
        item_train_counts = Counter(metadata["item_train_counts"])
        user_rated_items = defaultdict(
            set, {u: set(v) for u, v in metadata["user_rated_items"].items()}
        )
        num_users = metadata["num_users"]
        num_items = metadata["num_items"]
        ITEM_MIN_CF = metadata["ITEM_MIN_CF"]
        ALPHA = metadata["ALPHA"]

        _READY = True
        print("[Hybrid] Loaded cached model/artifacts.")
        return True

    except Exception as e:
        print(f"[Hybrid] Cache load failed, will rebuild. Reason: {e}")
        return False


def _shared_prep_path():
    return _cache_dir() / "preprocessed_shared.pkl"


def _save_shared_preprocessed(train_trips, val_trips, test_trips):
    payload = {
        "user_map": user_map,
        "item_map": item_map,
        "reverse_item_map": reverse_item_map,
        "train_triplets": train_trips,
        "val_triplets": val_trips,
        "test_triplets": test_trips,
        "dense_threshold": 20,
        "seed": 0,
    }
    with open(_shared_prep_path(), "wb") as f:
        pickle.dump(payload, f)


def _load_shared_preprocessed():
    global user_map, item_map, reverse_item_map
    global train_triplets, val_triplets, test_triplets

    path = _shared_prep_path()
    if not path.exists():
        return False

    with open(path, "rb") as f:
        payload = pickle.load(f)

    user_map = payload["user_map"]
    item_map = payload["item_map"]
    reverse_item_map = payload["reverse_item_map"]
    train_triplets = payload["train_triplets"]
    val_triplets = payload["val_triplets"]
    test_triplets = payload["test_triplets"]
    return True


def _write_shared_from_cache_only():
    raise RuntimeError(
        "preprocessed_shared.pkl missing. "
        "Please run once with ../dataset present to generate it, "
        "or include preprocessed_shared.pkl in your submission."
    )
