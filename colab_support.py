"""CUDA verification and recoverable runs for the Colab notebook."""
import csv
import json
import os
import shutil
import subprocess
import sys
from itertools import groupby, islice
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter


def compare_encoders(data, model_dir, batch_size):
    """Compare CPU and CUDA on the same FP32 ONNX graph and sample inputs."""
    import numpy as np
    from scipy import sparse
    from src.content_based import recommend
    from src.data import iter_metadata
    from src.features import Encoder, DIMENSIONS

    samples = {
        domain: [row["tfidf_text"] for row in islice(
            iter_metadata(data / f"{domain}_tfidf_metadata.jsonl"), count)]
        for domain, count in (("books", 24), ("movies", 64))
    }
    if len(samples["books"]) < 6 or len(samples["movies"]) < 10:
        raise ValueError("CUDA comparison needs at least 6 books and 10 movies")
    book_count = len(samples["books"])
    texts = samples["books"] + samples["movies"]

    with TemporaryDirectory() as tmp:
        start = perf_counter()
        encoder = Encoder(4, "cpu_fp32", model_dir, batch_size)
        cpu, _ = encoder.encode(texts)
        cpu_seconds = perf_counter() - start
        del encoder
        start = perf_counter()
        encoder = Encoder(4, "cuda", model_dir, batch_size, Path(tmp) / "cuda_profile")
        gpu, _ = encoder.encode(texts)
        cuda_seconds = perf_counter() - start
        providers = encoder.session.get_providers()
        profile = Path(encoder.session.end_profiling())
        nodes = json.loads(profile.read_text(encoding="utf-8"))
        gpu_nodes = sum(event.get("cat") == "Node"
                        and event.get("args", {}).get("provider") == "CUDAExecutionProvider"
                        for event in nodes)
        del encoder
        if gpu_nodes == 0:
            raise RuntimeError("No executed CUDA nodes were found in the ONNX Runtime profile")
        for vectors in (cpu, gpu):
            if (vectors.shape != (len(texts), DIMENSIONS) or not np.isfinite(vectors).all()
                    or not np.allclose(np.linalg.norm(vectors, axis=1), 1, atol=1e-5)):
                raise ValueError("Expected finite, normalized 384-dimensional vectors")
        cosine = np.einsum("ij,ij->i", cpu, gpu).clip(-1, 1)

        # Fixed example histories isolate the encoder difference from CF fitting.
        user_ids = np.array(["example1", "example2"])
        positive = sparse.csr_matrix(
            (np.ones(4, dtype=np.float32), ([0, 0, 1, 1], [0, 1, 2, 3])),
            shape=(2, book_count))
        inferred = sparse.csr_matrix(
            (np.ones(2, dtype=np.float32), ([0, 1], [4, 5])), shape=positive.shape)
        movie_ids = np.array([f"movie_{i:03}" for i in range(len(samples["movies"]))])
        aliases = {movie: movie for movie in movie_ids}
        overlaps = {}
        for name, cf in (("content_based", None), ("hybrid", inferred)):
            ranks = []
            for device, vectors in (("cpu", cpu), ("cuda", gpu)):
                path = Path(tmp) / f"{device}_{name}.csv"
                recommend(user_ids, positive, vectors[:book_count], movie_ids,
                          vectors[book_count:], aliases, path, inferred=cf, cf_weight=.2)
                with path.open(encoding="utf-8", newline="") as handle:
                    ranks.append({
                        user: [r["canonical_work_id"] for r in rows]
                        for user, rows in groupby(csv.DictReader(handle), key=lambda r: r["user_id"])
                    })
            overlaps[name] = {
                user: {
                    "common_movies": len(set(ranks[0][user]) & set(ranks[1][user])),
                    "same_order": ranks[0][user] == ranks[1][user],
                    "cpu_returned": len(ranks[0][user]),
                    "cuda_returned": len(ranks[1][user]),
                } for user in user_ids
            }

    passed = bool(cosine.min() >= .999 and all(
        result["common_movies"] >= min(9, result["cpu_returned"], result["cuda_returned"])
        and result["cpu_returned"] == result["cuda_returned"] > 0
        for model in overlaps.values() for result in model.values()))
    return {
        "passed": passed, "providers": providers, "cuda_node_events": gpu_nodes,
        "sample_texts": len(texts), "min_cosine": float(cosine.min()),
        "mean_cosine": float(cosine.mean()), "max_absolute_error": float(np.max(np.abs(cpu - gpu))),
        "sample_seconds_including_session_init": {"cpu_fp32": cpu_seconds, "cuda": cuda_seconds},
        "ranking_comparison": overlaps,
    }


def copy_file(source, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".partial")
    shutil.copy2(source, temporary)
    temporary.replace(destination)


def checkpoint(output, drive_output):
    """Copy completed outputs. A stage marker is published after its payloads."""
    cache = output / "cache"
    if not (cache / "inputs.json").exists():
        return
    saved_signature = drive_output / "cache/inputs.json"
    if saved_signature.exists() and json.loads(saved_signature.read_text(encoding="utf-8")) != json.loads(
            (cache / "inputs.json").read_text(encoding="utf-8")):
        raise ValueError("Drive checkpoint belongs to different inputs or code. Choose a new run name.")
    files = [cache / "inputs.json"]
    if (cache / "tfidf_complete.json").exists():
        files += [cache / name for name in (
            "books_tfidf.npz", "movies_tfidf.npz", "catalog.npz", "tfidf.joblib", "tfidf_complete.json")]
    files.extend(sorted(cache.glob("inferred_books_*.npz")))
    if (cache / "minilm_complete.json").exists():
        files += [cache / name for name in ("books_minilm.npy", "movies_minilm.npy", "minilm_complete.json")]
    files.extend(sorted(output.glob("*_recommendations.csv")))
    files.extend(output / name for name in ("metrics.json", "run_stats.json", "resource_stats.json")
                 if (output / name).is_file())
    for source in files:
        destination = drive_output / source.relative_to(output)
        source_stat = source.stat()
        if destination.exists():
            dest_stat = destination.stat()
            if dest_stat.st_size == source_stat.st_size and dest_stat.st_mtime_ns == source_stat.st_mtime_ns:
                continue
        copy_file(source, destination)


def gpu_process_mib(pid):
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,used_gpu_memory", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, check=True, timeout=4)
        rows = [line.split(",") for line in result.stdout.splitlines()]
        values = [int(mib.strip()) for proc_id, mib in rows if int(proc_id.strip()) == pid]
        return sum(values)
    except (OSError, ValueError, subprocess.SubprocessError):
        return None


def monitored_run(command, repo, output, drive_output):
    """Sample the run process every five seconds and checkpoint completed stages."""
    import psutil

    from main import write_json
    start = perf_counter()
    stats = {"peak_process_rss_mib": 0., "peak_system_ram_used_mib": 0.,
             "peak_cpu_percent": 0., "peak_gpu_process_mib": None, "sample_interval_seconds": 5}
    with subprocess.Popen(command, cwd=repo, env={**os.environ, "PYTHONUNBUFFERED": "1"}) as process:
        monitored = psutil.Process(process.pid)
        monitored.cpu_percent()
        try:
            while process.poll() is None:
                try:
                    stats["peak_process_rss_mib"] = max(stats["peak_process_rss_mib"],
                                                       monitored.memory_info().rss / 2**20)
                    stats["peak_system_ram_used_mib"] = max(stats["peak_system_ram_used_mib"],
                                                            psutil.virtual_memory().used / 2**20)
                    stats["peak_cpu_percent"] = max(stats["peak_cpu_percent"], monitored.cpu_percent())
                except psutil.NoSuchProcess:
                    break
                gpu = gpu_process_mib(process.pid)
                if gpu is not None:
                    stats["peak_gpu_process_mib"] = max(stats["peak_gpu_process_mib"] or 0, gpu)
                checkpoint(output, drive_output)
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    continue
        finally:
            if process.poll() is None:
                process.terminate()
                process.wait()
            stats["elapsed_seconds_including_checkpoints"] = round(perf_counter() - start, 3)
            stats["returncode"] = process.returncode
            write_json(output / "resource_stats.json", stats)
            checkpoint(output, drive_output)
    if process.returncode:
        raise subprocess.CalledProcessError(process.returncode, command)
    return stats


def validate_rankings(path):
    """Check rank continuity and uniqueness in each user's final list."""
    users = short_lists = 0
    with path.open(encoding="utf-8", newline="") as handle:
        for _, group in groupby(csv.DictReader(handle), key=lambda row: row["user_id"]):
            rows = list(group)
            count = len(rows)
            if (count > 10 or [int(row["rank"]) for row in rows] != list(range(1, count + 1))
                    or len({row["canonical_work_id"] for row in rows}) != count):
                raise ValueError(f"Invalid ranks or duplicate film in {path.name}")
            users += 1
            short_lists += count < 10
    if users == 0:
        raise ValueError(f"No recommendation rows in {path.name}")
    return {"users_with_recommendations": users, "short_lists": short_lists}


def main():
    import argparse
    from main import write_json

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["check", "run"])
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--drive-output", required=True, type=Path)
    parser.add_argument("--batch-size", default=32, type=int)
    parser.add_argument("--device", choices=["cuda", "cpu_fp32"], default="cuda")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    args.drive_output.mkdir(parents=True, exist_ok=True)
    if args.command == "check":
        path = args.output / "cuda_check.json"
        write_json(path, {"passed": False, "status": "verification started"})
        copy_file(path, args.drive_output / path.name)
        check = compare_encoders(args.data, args.model_dir, args.batch_size)
        write_json(path, check)
        copy_file(path, args.drive_output / path.name)
        print(json.dumps(check, indent=2))
        if not check["passed"]:
            raise SystemExit("CPU/CUDA sample comparison did not meet the stated thresholds.")
        return
    repo = Path(__file__).resolve().parent
    command = [sys.executable, str(repo / "main.py"), "run", "--model", "all",
               "--device", args.device, "--data", str(args.data), "--model-dir", str(args.model_dir),
               "--output", str(args.output), "--batch-size", str(args.batch_size),
               "--cf-percent", "20", "--cf-books", "50"]
    monitored_run(command, repo, args.output, args.drive_output)
    metrics = json.loads((args.output / "metrics.json").read_text(encoding="utf-8"))
    expected = {"tfidf": (0, 0), "tfidf_hybrid_cf20_books50": (20, 50),
                "minilm": (0, 0), "minilm_hybrid_cf20_books50": (20, 50)}
    checks = {}
    for name, (percent, books) in expected.items():
        result = metrics[name]
        if (result["cf_percent"], result["cf_books"]) != (percent, books):
            raise ValueError(f"Unexpected Hybrid settings: {name}")
        for key in ("precision_at_10", "recall_at_10", "ndcg_at_10"):
            if not 0 <= result[key] <= 1:
                raise ValueError(f"Invalid {key}: {name}")
        checks[name] = validate_rankings(args.output / f"{name}_recommendations.csv")
    path = args.output / "ranking_checks.json"
    write_json(path, checks)
    copy_file(path, args.drive_output / path.name)
    print(json.dumps(checks, indent=2))


if __name__ == "__main__":
    main()
