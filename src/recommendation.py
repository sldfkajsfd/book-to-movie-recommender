"""Equal-weight positive-book profiles and unique-work cosine Top 10."""
import csv
import numpy as np
from scipy import sparse
from sklearn.preprocessing import normalize
from threadpoolctl import threadpool_limits
from .data import select_top_k
from .hybrid import mix_profiles


def recommend(users, positive, books, movie_ids, movies, aliases, path, inferred=None, k=10):
    counts = np.asarray(positive.sum(axis=1)).ravel()
    eligible = np.flatnonzero(counts > 0)
    ranks = np.empty(len(movie_ids), dtype=np.int32)
    ranks[np.argsort(movie_ids)] = np.arange(len(movie_ids))
    groups = [aliases[work] for work in movie_ids]
    transposed = movies.T.tocsr() if sparse.issparse(movies) else movies.T
    with path.open("w", encoding="utf-8", newline="") as handle, threadpool_limits(limits=4):
        writer = csv.writer(handle)
        writer.writerow(["user_id", "work_id", "canonical_work_id", "rank", "score"])
        for start in range(0, len(eligible), 8):
            rows = eligible[start:start+8]
            weights = positive[rows].multiply((1 / counts[rows]).astype(np.float32)[:, None]).tocsr()
            profiles = normalize(weights @ books, copy=False)
            if inferred is not None:
                profiles = mix_profiles(profiles, inferred[rows], books)
            scores = profiles @ transposed
            for local, user in enumerate(rows):
                if sparse.issparse(scores):
                    row = scores.getrow(local); ids, values = row.indices, row.data
                else:
                    ids, values = np.arange(len(movie_ids)), scores[local]
                limit = k
                while True:
                    chosen, selected_scores, available, _ = select_top_k(ids, values, ranks, limit)
                    seen, selected = set(), []
                    for movie, score in zip(chosen, selected_scores):
                        group = groups[movie]
                        if group not in seen:
                            seen.add(group); selected.append((movie, score))
                        if len(selected) == k:
                            break
                    if len(selected) == k or len(chosen) == available:
                        break
                    limit *= 2
                for rank, (movie, score) in enumerate(selected, 1):
                    writer.writerow([users[user], movie_ids[movie], groups[movie], rank, float(score)])
            if start % 2048 == 0:
                print(f"Recommendations: {start}/{len(eligible)}", flush=True)
    return users[eligible]
