# 사용자가 평가한 책만으로 취향을 판단하는 로직인 content_based의 한계를 보완하여 Hybrid 방법을 구현한다.
#

import numpy as np
from scipy import sparse
from sklearn.preprocessing import normalize
from .data import select_top_k


def inferred_books(positive, observed, book_ids, book_limits=(20,), neighbors=100):
    """Score unseen books once, then build weights for each requested cutoff."""
    book_limits = tuple(dict.fromkeys(book_limits))
    max_books = max(book_limits)
    source = positive
    transpose = source.T.tocsr()
    norms = np.sqrt(np.asarray(source.multiply(source).sum(axis=1)).ravel())
    user_ranks = np.arange(source.shape[0])
    book_ranks = np.empty(len(book_ids), dtype=np.int32)
    book_ranks[np.argsort(book_ids)] = np.arange(len(book_ids))
    weights_by_limit = {limit: ([], [], []) for limit in book_limits}
    for start in range(0, source.shape[0], 128):
        similarities = (source[start:start+128] @ transpose).tocsr()
        for local in range(similarities.shape[0]):
            user = start + local
            left, right = similarities.indptr[local:local+2]
            ids = similarities.indices[left:right]
            mask = ids != user
            ids = ids[mask]
            values = similarities.data[left:right][mask] / (norms[user] * norms[ids])
            ids, values, _, _ = select_top_k(ids, values, user_ranks, neighbors)
            if not len(ids):
                continue
            # Match the original reduction over a fixed-width row of neighbor scores.
            total = np.pad(values, (0, neighbors - len(values))).sum()
            weights = sparse.csr_matrix((values / total,
                (np.zeros(len(ids), dtype=int), ids)), shape=(1, source.shape[0]))
            scores = (weights @ source).tocsr()
            scores -= scores.multiply(observed[user])
            scores.eliminate_zeros()
            chosen, values, _, _ = select_top_k(scores.indices, scores.data, book_ranks, max_books)
            if len(chosen):
                for limit, (rows, cols, weights) in weights_by_limit.items():
                    selected = chosen[:limit]
                    selected_scores = values[:limit]
                    total = np.pad(selected_scores, (0, limit - len(selected))).sum()
                    rows.extend([user] * len(selected))
                    cols.extend(selected)
                    weights.extend(selected_scores / total)
        if start % 2048 == 0:
            print(f"CF: {start}/{source.shape[0]}", flush=True)
    return {
        limit: sparse.csr_matrix((np.asarray(weights, dtype=np.float32), (rows, cols)),
                                 shape=source.shape)
        for limit, (rows, cols, weights) in weights_by_limit.items()
    }


def mix_profiles(direct, inferred, book_vectors, cf_weight=.5):
    if not 0 <= cf_weight <= 1:
        raise ValueError("CF weight must be between 0 and 1")
    available = np.diff(inferred.indptr) > 0
    if cf_weight == 0 or not available.any():
        return direct.copy()
    additional = normalize(inferred[available] @ book_vectors, copy=False)
    combined = normalize((1 - cf_weight) * direct[available] + cf_weight * additional, copy=False)
    if sparse.issparse(direct):
        result = direct.tolil()
        result[available] = combined
        return result.tocsr()
    result = direct.copy()
    result[available] = combined
    return result
