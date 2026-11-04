"""Observed-rating metrics; never used to construct a recommendation score."""
import csv
import json
import numpy as np
from .data import ratings

def ranking_metrics(recommendations,positive_movies,k=10):
    """Binary metrics on an already-fixed ranking; missing slots contribute zero."""
    if not positive_movies:
        raise ValueError("Ranking metrics require at least one positive movie")
    if len(recommendations)>k or len(set(recommendations))!=len(recommendations):
        raise ValueError("Recommendation ranking must contain unique IDs and at most K items")
    hits=np.asarray([int(movie in positive_movies) for movie in recommendations]+[0]*(k-len(recommendations)),dtype=np.float64)
    discounts=1/np.log2(np.arange(k,dtype=np.float64)+2)
    count=int(hits.sum())
    dcg=float(hits @ discounts)
    idcg=float(discounts[:min(k,len(positive_movies))].sum())
    return {"hits_at_10":count,"precision_at_10":count/k,"recall_at_10":count/len(positive_movies),
            "ndcg_at_10":dcg/idcg,"dcg_at_10":dcg,"idcg_at_10":idcg}


def evaluate(data, recommendation_path, users, aliases):
    frame = ratings(data / "movies_ratings.csv")
    original_positive = set(frame.loc[frame.rating >= 4, "user_id"])
    cohort = set(users) & original_positive
    if not cohort:
        raise ValueError("No eligible movie evaluation users")
    frame["canonical"] = frame.work_id.map(aliases)
    if frame.canonical.isna().any():
        raise ValueError("Movie rating has no metadata group")
    latest = frame.sort_values(["timestamp", "work_id"]).drop_duplicates(["user_id", "canonical"], keep="last")
    truth = latest.loc[latest.rating >= 4].groupby("user_id").canonical.agg(set).to_dict()
    lists = {u: [] for u in cohort}
    with recommendation_path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["user_id"] in lists:
                lists[row["user_id"]].append(row["canonical_work_id"])
    keys = ["precision_at_10", "recall_at_10", "ndcg_at_10"]
    totals = dict.fromkeys(keys, 0.)
    hits = 0
    for user, recommended in lists.items():
        positive = truth.get(user, set())
        if not positive:
            continue
        metric = ranking_metrics(recommended, positive, 10)
        hits += metric["hits_at_10"]
        for key in keys:
            totals[key] += metric[key]
    return {"evaluation_users": len(cohort), "hits_at_10": hits,
            **{key: value / len(cohort) for key, value in totals.items()}}
