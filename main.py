import argparse
import json
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from time import perf_counter

ROOT = Path(__file__).resolve().parent


def write_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    temporary.replace(path)


def main():
    started = perf_counter()
    parser = argparse.ArgumentParser(description="책 취향 기반 영화 추천")
    parser.add_argument("command", choices=["run", "results"])
    parser.add_argument("--model", choices=["tfidf", "minilm", "all"], default="all")
    parser.add_argument("--device", choices=["cpu", "cpu_fp32", "gpu", "cuda"], default="cpu")
    parser.add_argument("--model-dir", type=Path, default=ROOT / "models/minilm")
    parser.add_argument("--batch-size", type=int, help="MiniLM inference batch size")
    parser.add_argument("--cf-percent", type=int, nargs="+", default=[20],
                        help="CF 반영 비율 (1~100). 여러 값을 주면 같은 후보로 비교")
    parser.add_argument("--cf-books", type=int, nargs="+", default=[50],
                        help="추론 책의 최대 수. 여러 값을 주면 비율과 교차 비교")
    parser.add_argument("--data", type=Path, default=ROOT / "data")
    parser.add_argument("--output", type=Path, default=ROOT / "results")
    args = parser.parse_args()
    if args.batch_size is not None and args.batch_size < 1:
        parser.error("--batch-size must be positive.")
    if any(not 1 <= percent <= 100 for percent in args.cf_percent):
        parser.error("--cf-percent에는 1~100 사이의 정수를 지정하세요.")
    if any(limit < 1 for limit in args.cf_books):
        parser.error("--cf-books에는 양의 정수를 지정하세요.")
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
    timings = {"device": args.device, "model": args.model, "stages": {}}

    def record_stage(name, start, cached=False):
        timings["stages"][name] = {
            "seconds": round(perf_counter() - start, 3), "cached": cached,
        }
        timings["elapsed_seconds"] = round(perf_counter() - started, 3)
        write_json(output / "run_stats.json", timings)

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
    signature["libraries"] = {name: version(name) for name in (
        "numpy", "scipy", "pandas", "scikit-learn", "joblib")}
    protocol_path = cache / "inputs.json"
    if protocol_path.exists() and json.loads(protocol_path.read_text(encoding="utf-8")) != signature:
        parser.error("입력 또는 코드가 기존 캐시와 다릅니다. --output으로 새 결과 폴더를 지정하세요.")
    write_json(protocol_path, signature)
    stage_start = perf_counter()
    tfidf_cached = (cache / "tfidf_complete.json").exists()
    if not tfidf_cached:
        build_tfidf(data, cache, json.loads(stop_path.read_text(encoding="utf-8")))
        write_json(cache / "tfidf_complete.json", {})
    record_stage("tfidf", stage_start, tfidf_cached)
    with np.load(cache / "catalog.npz") as catalog:
        book_ids, movie_ids = catalog["book_ids"], catalog["movie_ids"]


    # Build two tables with users as rows and books as columns.
    # positive: set the entry to 1 if the user rated the book 4 or 5, for checking which books are contained
    # observed: marks rated books so CF can exclude them from new candidates.
    users, positive, observed = book_preferences(data, book_ids)
    stage_start = perf_counter()
    cf_paths = {limit: cache / f"inferred_books_{limit}.npz" for limit in args.cf_books}
    missing = [limit for limit, path in cf_paths.items() if not path.exists()]
    if missing:
        for limit, weights in inferred_books(positive, observed, book_ids, missing).items():
            temporary = cf_paths[limit].with_suffix(".tmp.npz")
            sparse.save_npz(temporary, weights)
            temporary.replace(cf_paths[limit])
    cf_by_limit = {limit: sparse.load_npz(path) for limit, path in cf_paths.items()}
    record_stage("book_cf", stage_start, not missing)
    aliases, _, _ = build_aliases(data / "movies_tfidf_metadata.jsonl")
    all_metrics = {}
    if (output / "metrics.json").exists():
        all_metrics = json.loads((output / "metrics.json").read_text(encoding="utf-8"))
    for representation in (["tfidf", "minilm"] if args.model == "all" else [args.model]):
        if representation == "tfidf":
            books = sparse.load_npz(cache / "books_tfidf.npz")
            movies = sparse.load_npz(cache / "movies_tfidf.npz")
        else:
            stage_start = perf_counter()
            model_names = ["tokenizer.json", "model_quint8_avx2.onnx" if args.device == "cpu" else "model.onnx"]
            model_signature = {name: fingerprint(args.model_dir / name) for name in model_names}
            model_signature["device"] = args.device
            model_signature["batch_size"] = args.batch_size
            runtime = {"cpu": "onnxruntime", "cpu_fp32": "onnxruntime", "cuda": "onnxruntime-gpu", "gpu": "openvino"}
            # Colab's GPU package also provides the CPU provider for FP32 comparisons.
            try:
                model_signature["runtime_version"] = version(runtime[args.device])
            except PackageNotFoundError:
                if args.device not in {"cpu", "cpu_fp32"}:
                    raise
                model_signature["runtime_version"] = version("onnxruntime-gpu")
            model_signature["tokenizers_version"] = version("tokenizers")
            marker = cache / "minilm_complete.json"
            minilm_cached = marker.exists()
            if minilm_cached and json.loads(marker.read_text(encoding="utf-8")) != model_signature:
                parser.error("임베딩 장치/모델이 캐시와 다릅니다. 다른 --output을 사용하세요.")
            if not minilm_cached:
                build_minilm(data, cache, args.device, args.model_dir, args.batch_size)
                write_json(marker, model_signature)
            books = np.load(cache / "books_minilm.npy", mmap_mode="r")
            movies = np.load(cache / "movies_minilm.npy", mmap_mode="r")
            record_stage("minilm_embeddings", stage_start, minilm_cached)
        variants = [(representation, 0, 0)]
        for percent in dict.fromkeys(args.cf_percent):
            for limit in cf_paths:
                suffix = "_hybrid" if (percent, limit) == (50, 20) else f"_hybrid_cf{percent}_books{limit}"
                variants.append((representation + suffix, percent, limit))
        for name, percent, limit in variants:
            stage_start = perf_counter()
            path = output / f"{name}_recommendations.csv"
            temporary = path.with_suffix(".tmp")
            active_users = recommend(users, positive, books, movie_ids, movies, aliases, temporary,
                                     inferred=cf_by_limit.get(limit), cf_weight=percent / 100)
            all_metrics[name] = {
                "cf_percent": percent,
                "cf_books": limit,
                **evaluate(data, temporary, active_users, aliases),
            }
            temporary.replace(path)
            write_json(output / "metrics.json", all_metrics)
            record_stage(f"recommend_and_evaluate_{name}", stage_start)
            print(name, all_metrics[name], flush=True)
        del books, movies


if __name__ == "__main__":
    main()
