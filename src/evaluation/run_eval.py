import pickle
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch

from src import hybrid_engine as hybrid
from src import ncf_engine as ncf


def _load_shared_payload():
    project_root = Path(__file__).resolve().parents[2]

    models_dir = project_root / "models"
    path = models_dir / "preprocessed_shared.pkl"

    if not path.exists():
        raise RuntimeError(
            f"Missing {path}. "
            "Please ensure the models directory contains the pretrained artefacts."
        )

    with open(path, "rb") as f:
        return pickle.load(f)


# RMSE
def rmse(triplets, score_fn):
    squared_e = 0.0
    n = 0
    for u, i, r in triplets:
        pred = float(score_fn(int(u), int(i)))
        err = float(r) - pred
        squared_e += err * err
        n += 1
    return float(np.sqrt(squared_e / max(n, 1)))


# Serendipity helpers
def build_user_ratings(triplets):
    user_ratings = defaultdict(list)
    for u, i, r in triplets:
        user_ratings[int(u)].append((int(i), float(r)))
    return user_ratings


def cosine_similarity(a, b):
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0.0 or nb == 0.0:
        return None
    return float(np.dot(a, b) / (na * nb))


def top_k_for_user(u, user_ratings, score_fn, top_k):
    scored = []
    for i, r_true in user_ratings[u]:
        pred = float(score_fn(u, i))
        scored.append((pred, i, r_true))

    scored.sort(reverse=True, key=lambda x: x[0])
    return scored[:top_k]


def serendipity_for_user(
    u,
    user_ratings,
    score_fn,
    user_profiles,
    item_genre_matrix,
    top_k=10,
    rel_thresh=4.0,
):

    def item_unexpectedness(u_, i_):
        u_vec = user_profiles.get(u_)
        if u_vec is None:
            return None

        i_vec = item_genre_matrix[i_]
        sim = cosine_similarity(u_vec, i_vec)
        if sim is None:
            return None
        return 1.0 - sim

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


def build_user_profiles_from_train(train_triplets, item_genre_matrix):
    accum = defaultdict(lambda: np.zeros(item_genre_matrix.shape[1], dtype=np.float32))
    counts = defaultdict(int)

    for u, i, _r in train_triplets:
        u = int(u)
        i = int(i)
        accum[u] += item_genre_matrix[i]
        counts[u] += 1

    user_profiles = {}
    for u, vec in accum.items():
        c = counts[u]
        if c > 0:
            user_profiles[u] = vec / float(c)
        else:
            user_profiles[u] = vec  # all zeros

    return user_profiles


def mean_serendipity(
    test_triplets,
    train_triplets,
    score_fn,
    item_genre_matrix,
    top_k=10,
    rel_thresh=4.0,
):
    user_ratings = build_user_ratings(test_triplets)
    user_profiles = build_user_profiles_from_train(train_triplets, item_genre_matrix)

    vals = []
    for u in user_ratings.keys():
        s = serendipity_for_user(
            u,
            user_ratings,
            score_fn,
            user_profiles,
            item_genre_matrix,
            top_k=top_k,
            rel_thresh=rel_thresh,
        )
        if s is not None:
            vals.append(float(s))

    return float(np.mean(vals)) if vals else 0.0, len(vals), len(user_ratings)


def main(top_k=10, rel_thresh=4.0):
    payload = _load_shared_payload()
    train_triplets = payload["train_triplets"]
    test_triplets = payload["test_triplets"]

    # Initialise engines (should load cached artefacts; no ../dataset needed if caches exist)
    hybrid.initialise()
    ncf.initialise()

    # Need item_genre_matrix for serendipity
    item_genre_matrix = getattr(ncf, "item_genre_matrix", None)
    if item_genre_matrix is None:
        raise RuntimeError(
            "ncf.item_genre_matrix not available; cannot compute serendipity."
        )

    # Predictors
    hybrid_predict = hybrid._hybrid_score

    def ncf_predict(u, i):
        g = int(ncf.user_gender[u])
        a = int(ncf.user_age[u])
        o = int(ncf.user_occupation[u])
        with torch.no_grad():
            return ncf.ncf_model(
                torch.tensor([u], dtype=torch.long, device=ncf.device),
                torch.tensor([i], dtype=torch.long, device=ncf.device),
                torch.tensor([g], dtype=torch.long, device=ncf.device),
                torch.tensor([a], dtype=torch.long, device=ncf.device),
                torch.tensor([o], dtype=torch.long, device=ncf.device),
            ).item()

    # RMSE
    hybrid_rmse = rmse(test_triplets, hybrid_predict)
    ncf_rmse = rmse(test_triplets, ncf_predict)

    # Serendipity
    hybrid_ser, h_used, h_total_users = mean_serendipity(
        test_triplets,
        train_triplets,
        hybrid_predict,
        item_genre_matrix,
        top_k=top_k,
        rel_thresh=rel_thresh,
    )
    ncf_ser, n_used, n_total_users = mean_serendipity(
        test_triplets,
        train_triplets,
        ncf_predict,
        item_genre_matrix,
        top_k=top_k,
        rel_thresh=rel_thresh,
    )

    print("================================")
    print(f"Shared test split size : {len(test_triplets)}")
    print(f"Serendipity settings   : top_k={top_k}, rel_thresh={rel_thresh}")
    print("--------------------------------")
    print(f"HYBRID RMSE               : {hybrid_rmse:.4f}")
    print(f"HYBRID Serendipity, Top-{top_k}: {hybrid_ser:.4f}")
    print("--------------------------------")
    print(f"NCF    RMSE               : {ncf_rmse:.4f}")
    print(f"NCF    Serendipity, Top-{top_k}: {ncf_ser:.4f}")
    print("================================")


if __name__ == "__main__":
    main(top_k=10, rel_thresh=4.0)
