"""Book-only TF-IDF and the fixed MiniLM text encoder."""
from collections import Counter
from pathlib import Path
import sqlite3
import tempfile
import time
import json
import os
import sys
import joblib
import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from .data import iter_metadata

ROOT = Path(__file__).resolve().parents[1]
DEPS = ROOT / "models"
MODEL = DEPS / "minilm"
BATCH = 32
DIMENSIONS = 384
MAX_LENGTH = 256

def validated_records(path):
    seen = set()
    for source_row, record in enumerate(iter_metadata(path)):
        work_id, text = record.get("work_id"), record.get("tfidf_text")
        if not isinstance(work_id, str) or not work_id.strip() or work_id != work_id.strip():
            raise ValueError(f"{path.name}: invalid work_id at source row {source_row}")
        if work_id in seen:
            raise ValueError(f"{path.name}: duplicate work_id at source row {source_row}")
        if not isinstance(text, str):
            raise ValueError(f"{path.name}: tfidf_text must be a string")
        seen.add(work_id)
        yield source_row, work_id, text

def build_vocabulary(path, database_path, analyzer, config):
    """Exact corpus TF and document DF via bounded counters and SQLite upserts."""
    connection = sqlite3.connect(database_path)
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA synchronous=NORMAL")
    connection.execute("PRAGMA cache_size=-32768")
    connection.execute("PRAGMA temp_store=FILE")
    connection.execute("CREATE TABLE terms(term TEXT PRIMARY KEY, tf INTEGER NOT NULL, df INTEGER NOT NULL) WITHOUT ROWID")
    term_frequency, document_frequency = Counter(), Counter()

    def flush():
        if not term_frequency:
            return
        with connection:
            connection.executemany(
                "INSERT INTO terms VALUES (?, ?, ?) ON CONFLICT(term) DO UPDATE SET tf=tf+excluded.tf, df=df+excluded.df",
                ((term, count, document_frequency[term]) for term, count in term_frequency.items()),
            )
        term_frequency.clear()
        document_frequency.clear()

    documents, total_occurrences, no_terms = 0, 0, 0
    started = time.perf_counter()
    try:
        for source_row, _, text in validated_records(path):
            counts = Counter(analyzer(text))
            documents = source_row + 1
            total_occurrences += sum(counts.values())
            no_terms += not bool(counts)
            term_frequency.update(counts)
            document_frequency.update(counts.keys())
            if len(term_frequency) >= 120_000:
                flush()
            if documents % 25_000 == 0:
                print(f"Vocabulary: {documents:,} books, elapsed {time.perf_counter()-started:.1f}s", flush=True)
        flush()
        if not documents:
            raise ValueError("No book metadata records")
        candidates = connection.execute("SELECT COUNT(*) FROM terms").fetchone()[0]
        eligible = connection.execute(
            "SELECT COUNT(*) FROM terms WHERE df>=? AND df<=?",
            (config["min_df"], config["max_df"] * documents),
        ).fetchone()[0]
        selected = connection.execute(
            "SELECT term,tf,df FROM terms WHERE df>=? AND df<=? ORDER BY tf DESC,term ASC LIMIT ?",
            (config["min_df"], config["max_df"] * documents, config["max_features"]),
        ).fetchall()
        if not selected:
            raise ValueError("No vocabulary retained under configured thresholds")
        selected.sort(key=lambda row: row[0])
        connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        return selected, {
            "book_fit_documents": documents, "candidate_features": candidates,
            "features_after_df_filter": eligible, "selected_features": len(selected),
            "selected_unigrams": sum(" " not in term for term, _, _ in selected),
            "selected_bigrams": sum(" " in term for term, _, _ in selected),
            "analyzed_feature_occurrences": total_occurrences,
            "books_without_terms_after_stopwords": no_terms,
            "selection_rule": "DF thresholds, corpus TF descending, lexical ascending ties; final indices lexical ascending",
            "elapsed_seconds": round(time.perf_counter()-started, 3),
        }
    finally:
        connection.close()

def batched_transform(vectorizer, texts, batch_size):
    matrices, batch = [], []
    for text in texts:
        batch.append(text)
        if len(batch) == batch_size:
            matrices.append(vectorizer.transform(batch))
            batch.clear()
    if batch:
        matrices.append(vectorizer.transform(batch))
    if not matrices:
        raise ValueError("No movie metadata records")
    return sparse.vstack(matrices, format="csr", dtype=np.float32)

class Encoder:
    def __init__(self, threads, device):
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
        from tokenizers import Tokenizer

        self.tokenizer = Tokenizer.from_file(str(MODEL / "tokenizer.json"))
        self.tokenizer.no_padding()
        self.tokenizer.enable_truncation(max_length=MAX_LENGTH)
        self.device = device
        self.batch_size = 1 if device == "cpu_fp32" else BATCH
        if device == "gpu":
            import openvino as ov
            from openvino import opset13 as ops

            self.model_path = MODEL / "model.onnx"
            core = ov.Core()
            graph = core.read_model(self.model_path)
            mask_port = graph.input("attention_mask")
            mask = ops.convert(ops.unsqueeze(mask_port.get_node(), [2]), "f32")
            hidden = ops.convert(graph.output(0).get_node().input_value(0), "f32")
            mean = ops.divide(ops.reduce_sum(ops.multiply(hidden, mask), [1]),
                ops.reduce_sum(mask, [1]))
            graph = ov.Model([mean], graph.get_parameters())
            self.inputs = {port.get_any_name() for port in graph.inputs}
            self.session = core.compile_model(graph, "GPU", {"INFERENCE_PRECISION_HINT": "f16"})
            self.backend = "OpenVINO Intel GPU, FP16 inference hint on official ONNX"
            self.runtime_version = ov.__version__
        else:
            import onnxruntime as ort

            self.model_path = MODEL / ("model.onnx" if device == "cpu_fp32" else "model_quint8_avx2.onnx")
            options = ort.SessionOptions()
            options.intra_op_num_threads = threads
            options.inter_op_num_threads = 1
            if device == "cpu_fp32":
                options.enable_cpu_mem_arena = False
                options.enable_mem_pattern = False
                options.intra_op_num_threads = min(threads, 2)
            self.session = ort.InferenceSession(str(self.model_path),
                sess_options=options, providers=["CPUExecutionProvider"])
            self.inputs = {value.name for value in self.session.get_inputs()}
            self.backend = ("ONNX Runtime CPU, official FP32 ONNX" if device == "cpu_fp32"
                            else "ONNX Runtime CPU, official AVX2 UINT8 weights")
            self.runtime_version = ort.__version__

    def encode(self, texts):
        return self.encode_ids([row.ids for row in self.tokenizer.encode_batch(texts)])

    def encode_ids(self, encoded):
        lengths = np.array([len(ids) for ids in encoded])
        if np.any(lengths > MAX_LENGTH) or np.any(lengths == 0):
            raise ValueError("Encoder input length out of bounds")
        order = np.argsort(lengths, kind="stable")
        vectors = np.empty((len(encoded), DIMENSIONS), dtype=np.float32)
        for start in range(0, len(order), self.batch_size):
            indices = order[start:start+self.batch_size]
            width = int(lengths[indices[-1]])
            if self.device == "gpu":
                width = min(MAX_LENGTH, ((width + 15) // 16) * 16)
            inputs = {name: np.zeros((len(indices), width), dtype=np.int64) for name in self.inputs}
            for row, i in enumerate(indices):
                ids = encoded[i]
                inputs["input_ids"][row, :len(ids)] = ids
                inputs["attention_mask"][row, :len(ids)] = 1
            if self.device == "gpu":
                mean = self.session(inputs)[0].astype(np.float32, copy=False)
            else:
                hidden = self.session.run(None, inputs)[0]
                mask = inputs["attention_mask"].astype(np.float32)
                mean = np.einsum("btd,bt->bd", hidden, mask) / mask.sum(axis=1, keepdims=True)
            norm = np.linalg.norm(mean, axis=1, keepdims=True)
            if not np.isfinite(mean).all() or np.any(norm == 0):
                raise ValueError("Invalid encoder output")
            vectors[indices] = mean / norm
        return vectors, int(np.count_nonzero(lengths == MAX_LENGTH))


def build_tfidf(data, output, stopwords, max_features=200000, min_df=2):
    """Fit on Books only; retain the original 50k candidate mask for comparison."""
    settings = dict(min_df=min_df, max_df=.95, max_features=max_features)
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words=stopwords,
                                dtype=np.float32, norm="l2")
    with tempfile.TemporaryDirectory(dir=output) as temp:
        selected, stats = build_vocabulary(data / "books_tfidf_metadata.jsonl",
            Path(temp) / "counts.sqlite", vectorizer.build_analyzer(), settings)
    vocabulary = {term: i for i, (term, _, _) in enumerate(selected)}
    vectorizer.set_params(vocabulary=vocabulary)
    books = vectorizer.fit_transform(text for _, _, text in validated_records(data / "books_tfidf_metadata.jsonl"))
    movies = batched_transform(vectorizer,
        (text for _, _, text in validated_records(data / "movies_tfidf_metadata.jsonl")), 2048)
    # The baseline 50k mask defines the same CF books and movie candidates as the study.
    baseline = sorted(selected, key=lambda r: (-r[1], r[0]))[:min(50000, max_features)]
    columns = [vocabulary[t] for t, _, _ in baseline]
    book_baseline = np.asarray(books[:, columns].getnnz(axis=1)).ravel() > 0
    movie_keep = np.asarray(movies[:, columns].getnnz(axis=1)).ravel() > 0
    book_keep = np.asarray(books.getnnz(axis=1)).ravel() > 0
    book_ids = np.array([r["work_id"] for r in iter_metadata(data / "books_tfidf_metadata.jsonl")])
    movie_ids = np.array([r["work_id"] for r in iter_metadata(data / "movies_tfidf_metadata.jsonl")])
    sparse.save_npz(output / "books_tfidf.npz", books[book_keep])
    sparse.save_npz(output / "movies_tfidf.npz", movies[movie_keep])
    np.savez_compressed(output / "catalog.npz", book_ids=book_ids[book_keep],
        movie_ids=movie_ids[movie_keep], cf_book_mask=book_baseline[book_keep])
    joblib.dump(vectorizer, output / "tfidf.joblib")


def build_minilm(data, output, device):
    encoder = Encoder(4, device)
    with np.load(output / "catalog.npz") as catalog:
        maps = {domain: catalog[key].tolist() for domain, key in
                [("books", "book_ids"), ("movies", "movie_ids")]}
    for domain, ids in maps.items():
        index = {work: i for i, work in enumerate(ids)}
        matrix = np.lib.format.open_memmap(output / f"{domain}_minilm.npy", mode="w+",
                                         dtype=np.float32, shape=(len(ids), DIMENSIONS))
        texts, rows, count = [], [], 0
        for record in iter_metadata(data / f"{domain}_tfidf_metadata.jsonl"):
            if record["work_id"] not in index:
                continue
            texts.append(record["tfidf_text"]); rows.append(index[record["work_id"]])
            if len(texts) == 256:
                matrix[rows] = encoder.encode(texts)[0]
                count += len(rows); texts, rows = [], []
                if count % 8192 == 0:
                    print(f"{domain}: {count}/{len(ids)}", flush=True)
        if texts:
            matrix[rows] = encoder.encode(texts)[0]
        matrix.flush()
