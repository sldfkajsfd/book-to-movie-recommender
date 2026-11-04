"""Book-only user KNN: 100 neighbors, 20 unseen books, equal profile blend."""
import numpy as np
from scipy import sparse
from sklearn.preprocessing import normalize
from .data import select_top_k


def inferred_books(positive, observed, book_ids, cf_mask, neighbors=100, limit=20):
    # Keep the original baseline book population for CF, mapped to current book rows.
    source = positive.multiply(cf_mask).tocsr()
    source.eliminate_zeros()
    transpose = source.T.tocsr()
    norms = np.sqrt(np.asarray(source.multiply(source).sum(axis=1)).ravel())
    user_ranks = np.arange(source.shape[0])
    book_ranks = np.empty(len(book_ids), dtype=np.int32)
    book_ranks[np.argsort(book_ids)] = np.arange(len(book_ids))
    out_rows, out_cols, out_values = [], [], []
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
            chosen, values, _, _ = select_top_k(scores.indices, scores.data, book_ranks, limit)
            if len(chosen):
                out_rows.extend([user] * len(chosen)); out_cols.extend(chosen)
                total = np.pad(values, (0, limit - len(values))).sum()
                out_values.extend(values / total)
        if start % 2048 == 0:
            print(f"CF: {start}/{source.shape[0]}", flush=True)
    return sparse.csr_matrix((np.asarray(out_values, dtype=np.float32), (out_rows, out_cols)),
                             shape=source.shape)


def mix_profiles(direct, inferred, book_vectors):
    available = np.diff(inferred.indptr) > 0
    if not available.any():
        return direct.copy()
    additional = normalize(inferred[available] @ book_vectors, copy=False)
    combined = normalize(.5 * direct[available] + .5 * additional, copy=False)
    if sparse.issparse(direct):
        result = direct.tolil()
        result[available] = combined
        return result.tocsr()
    result = direct.copy()
    result[available] = combined
    return result
