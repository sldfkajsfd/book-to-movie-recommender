"""Input files and string-ID alignment. Movie ratings are evaluation-only."""
import json
import hashlib
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import sparse


def iter_metadata(path):
    seen = set()
    with Path(path).open(encoding="utf-8-sig") as handle:
        for line in handle:
            row = json.loads(line)
            work = row["work_id"]
            if not isinstance(work, str) or not work or work in seen:
                raise ValueError(f"Invalid or duplicate work ID in {path}")
            seen.add(work)
            yield row


def ratings(path):
    frame = pd.read_csv(path, dtype={"user_id": str, "work_id": str},
                        usecols=["user_id", "work_id", "rating", "timestamp"])
    if frame.duplicated(["user_id", "work_id"]).any():
        raise ValueError("Duplicate user/work rating")
    if not frame.rating.isin([1, 2, 3, 4, 5]).all():
        raise ValueError("Ratings must be integers from 1 to 5")
    return frame


def book_preferences(data, book_ids):
    users = pd.read_csv(data / "users_content_filtered.csv", dtype=str).user_id
    if users.duplicated().any():
        raise ValueError("Duplicate user ID")
    users = np.asarray(sorted(users), dtype=str)
    frame = ratings(data / "books_ratings.csv")
    rows = frame.user_id.map({u: i for i, u in enumerate(users)})
    cols = frame.work_id.map({v: i for i, v in enumerate(book_ids)})
    if rows.isna().any():
        raise ValueError("Book rating user missing from user list")
    def matrix(mask):
        mask = mask & cols.notna()
        return sparse.csr_matrix((np.ones(int(mask.sum()), dtype=np.float32),
            (rows[mask].to_numpy(dtype=np.int32), cols[mask].to_numpy(dtype=np.int32))),
            shape=(len(users), len(book_ids)))
    return users, matrix(frame.rating >= 4), matrix(frame.rating >= 1)


def fingerprint(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def select_top_k(indices, scores, movie_id_ranks, k):
    """Partial-select by score, resolve all boundary ties by lexical ID rank."""
    positive = scores > 0
    indices, scores = indices[positive], scores[positive]
    n = len(scores)
    boundary_tie = False
    if n > k:
        threshold = np.partition(scores, n-k)[n-k]
        above = np.flatnonzero(scores > threshold)
        tied = np.flatnonzero(scores == threshold)
        needed = k-len(above)
        boundary_tie = len(tied) > needed
        if len(tied) > needed:
            tie_ranks = movie_id_ranks[indices[tied]]
            tied = tied[np.argpartition(tie_ranks, needed-1)[:needed]]
        chosen = np.concatenate((above,tied))
        indices, scores = indices[chosen], scores[chosen]
    order = np.lexsort((movie_id_ranks[indices], -scores))
    return indices[order], scores[order], n, boundary_tie
