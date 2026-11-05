import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description="책 취향 기반 영화 추천")
    parser.add_argument("command", choices=["run", "results"])
    parser.add_argument("--model", choices=["tfidf", "minilm", "all"], default="all")
    parser.add_argument("--device", choices=["cpu", "gpu"], default="cpu")
    parser.add_argument("--data", type=Path, default=ROOT / "data")
    parser.add_argument("--output", type=Path, default=ROOT / "results")
    args = parser.parse_args()
    output = args.output
    if args.command == "results":
        path = output / "metrics.json"
        if not path.exists():
            parser.error("결과가 없습니다. 먼저 run을 실행하세요.")
        print(path.read_text(encoding="utf-8"))
        return
    import numpy as np
    from scipy import sparse
    from src.data import book_preferences, fingerprint
    from src.features import build_tfidf, build_minilm
    from src.editions import build_aliases
    from src.hybrid import inferred_books
    from src.content_based import recommend
    from src.evaluation import evaluate

    output.mkdir(parents=True, exist_ok=True)
    cache = output / "cache"
    cache.mkdir(exist_ok=True)
    data = args.data
    names = ["books_ratings.csv", "movies_ratings.csv", "books_tfidf_metadata.jsonl",
             "movies_tfidf_metadata.jsonl", "users_content_filtered.csv"]
    stop_path = ROOT / "stopwords.json"
    signature = {name: fingerprint(data / name) for name in names}
    signature["stopwords"] = fingerprint(stop_path)
    signature["source"] = {f.name: fingerprint(f) for f in (ROOT / "src").glob("*.py")}
    signature["entry"] = fingerprint(Path(__file__))
    protocol_path = cache / "inputs.json"
    if protocol_path.exists() and json.loads(protocol_path.read_text(encoding="utf-8")) != signature:
        parser.error("입력 또는 코드가 기존 캐시와 다릅니다. --output으로 새 결과 폴더를 지정하세요.")
    protocol_path.write_text(json.dumps(signature, indent=2), encoding="utf-8")
    if not (cache / "tfidf_complete.json").exists():
        build_tfidf(data, cache, json.loads(stop_path.read_text(encoding="utf-8")))
        (cache / "tfidf_complete.json").write_text("{}", encoding="utf-8")
    with np.load(cache / "catalog.npz") as catalog:
        book_ids, movie_ids = catalog["book_ids"], catalog["movie_ids"]


    # Build two tables with users as rows and books as columns.
    # positive: set the entry to 1 if the user rated the book 4 or 5, for checking which books are contained
    # observed: marks rated books so CF can exclude them from new candidates.
    users, positive, observed = book_preferences(data, book_ids)
    cf_path = cache / "inferred_books.npz"
    if not cf_path.exists():
        sparse.save_npz(cf_path, inferred_books(positive, observed, book_ids))
    cf = sparse.load_npz(cf_path)
    aliases, _, _ = build_aliases(data / "movies_tfidf_metadata.jsonl")
    all_metrics = {}
    if (output / "metrics.json").exists():
        all_metrics = json.loads((output / "metrics.json").read_text(encoding="utf-8"))
    for representation in (["tfidf", "minilm"] if args.model == "all" else [args.model]):
        if representation == "tfidf":
            books = sparse.load_npz(cache / "books_tfidf.npz")
            movies = sparse.load_npz(cache / "movies_tfidf.npz")
        else:
            model_dir = ROOT / "models/minilm"
            model_names = ["tokenizer.json", "model.onnx" if args.device == "gpu" else "model_quint8_avx2.onnx"]
            model_signature = {name: fingerprint(model_dir / name) for name in model_names}
            model_signature["device"] = args.device
            marker = cache / "minilm_complete.json"
            if marker.exists() and json.loads(marker.read_text(encoding="utf-8")) != model_signature:
                parser.error("임베딩 장치/모델이 캐시와 다릅니다. 다른 --output을 사용하세요.")
            if not marker.exists():
                build_minilm(data, cache, args.device)
                marker.write_text(json.dumps(model_signature), encoding="utf-8")
            books = np.load(cache / "books_minilm.npy", mmap_mode="r")
            movies = np.load(cache / "movies_minilm.npy", mmap_mode="r")
        for use_cf in (False, True):
            name = representation + ("_hybrid" if use_cf else "")
            path = output / f"{name}_recommendations.csv"
            active_users = recommend(users, positive, books, movie_ids, movies, aliases, path,
                                     inferred=cf if use_cf else None)
            all_metrics[name] = evaluate(data, path, active_users, aliases)
            (output / "metrics.json").write_text(json.dumps(all_metrics, indent=2), encoding="utf-8")
            print(name, all_metrics[name], flush=True)
        del books, movies


if __name__ == "__main__":
    main()
