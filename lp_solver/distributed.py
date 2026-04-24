from __future__ import annotations

import json
import logging
import multiprocessing as mp
import socket
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence

from .config import RuntimeConfig
from .scoring import ScoringEngine
from .search import AdaptivePolicy, KeyFactory, key_to_pipeline
from .tensor_ops import TensorEngine

LOG = logging.getLogger(__name__)


@dataclass
class CandidateResult:
    score: float
    key_meta: Dict
    preview: List[int]
    parts: Dict[str, float]


class Worker:
    def __init__(self, worker_id: int, config: RuntimeConfig) -> None:
        self.worker_id = worker_id
        self.config = config
        self.tensor = TensorEngine()
        self.scorer = ScoringEngine()
        self.factory = KeyFactory(config.constraints, seed=config.random_seed + worker_id)

    def run_batch(self, page_tensor, refs: List[Sequence[int]], strategy: str, beam_pool=None) -> List[CandidateResult]:
        candidates = []
        length = page_tensor.shape[-1] if hasattr(page_tensor, "shape") else len(page_tensor)
        if strategy == "grid":
            key_iter = self.factory.grid(length=length, limit=self.config.batch_size)
        elif strategy == "genetic" and beam_pool:
            key_iter = self._genetic_keys(beam_pool, n=self.config.batch_size)
        else:
            key_iter = (self.factory.random_key(length) for _ in range(self.config.batch_size))

        for ck in key_iter:
            pipeline = key_to_pipeline(ck)
            out = pipeline.apply(page_tensor, self.tensor)
            if hasattr(out, "detach"):
                arr = out.detach().cpu().numpy().astype(int).tolist()
            elif hasattr(out, "astype"):
                arr = out.astype(int).tolist()
            else:
                arr = [int(v) for v in out]
            sb = self.scorer.score(arr, refs)
            candidates.append(
                CandidateResult(
                    score=sb.total,
                    key_meta={
                        "a": ck.a,
                        "b": ck.b,
                        "key": ck.key,
                        "block_pattern": ck.block_pattern,
                        "perm_head": ck.permutation[:16],
                    },
                    preview=arr[:128],
                    parts=sb.parts,
                )
            )
        candidates.sort(key=lambda c: c.score, reverse=True)
        return candidates[: self.config.top_n_per_worker]

    def _genetic_keys(self, pool, n: int):
        for _ in range(n):
            p1, p2 = self.factory.rng.choice(pool), self.factory.rng.choice(pool)
            child = self.factory.crossover(p1, p2)
            yield self.factory.mutate(child)


class SocketWorkerServer:
    def __init__(self, host: str, port: int, worker: Worker) -> None:
        self.host, self.port, self.worker = host, port, worker

    @staticmethod
    def _recv_exact(conn: socket.socket, n: int) -> bytes:
        chunks = []
        remaining = n
        while remaining > 0:
            chunk = conn.recv(remaining)
            if not chunk:
                raise ConnectionError("socket closed during payload read")
            chunks.append(chunk)
            remaining -= len(chunk)
        return b"".join(chunks)

    def serve_forever(self) -> None:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.bind((self.host, self.port))
            s.listen()
            LOG.info("socket worker listening at %s:%s", self.host, self.port)
            while True:
                conn, _ = s.accept()
                with conn:
                    header = self._recv_exact(conn, 8)
                    size = int.from_bytes(header, "big")
                    payload = json.loads(self._recv_exact(conn, size).decode("utf-8"))
                    page = payload["page"]
                    refs = payload["refs"]
                    strategy = payload.get("strategy", "random")
                    results = self.worker.run_batch(page, refs, strategy)
                    body = json.dumps([r.__dict__ for r in results]).encode("utf-8")
                    conn.sendall(len(body).to_bytes(8, "big") + body)


def run_worker_process(worker_id: int, config: RuntimeConfig, page_vals: List[int], ref_vals, strategy: str, out_q: mp.Queue, beam_pool):
    wk = Worker(worker_id, config)
    page_tensor = wk.tensor.tensor_1d(page_vals)
    res = wk.run_batch(page_tensor=page_tensor, refs=ref_vals, strategy=strategy, beam_pool=beam_pool)
    out_q.put([r.__dict__ for r in res])


class Coordinator:
    def __init__(self, config: RuntimeConfig) -> None:
        self.config = config
        self.policy = AdaptivePolicy()
        self.best: List[Dict] = []

    def _checkpoint(self, iteration: int) -> None:
        out_dir = Path(self.config.checkpoints_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / f"iter_{iteration:06d}.json").write_text(json.dumps(self.best[:256], indent=2), encoding="utf-8")

    def run(self, page_vals: List[int], ref_vals: List[List[int]]) -> List[Dict]:
        beam_pool = []
        beam_factory = KeyFactory(self.config.constraints, seed=self.config.random_seed + 10_000)
        for it in range(self.config.max_iterations):
            q: mp.Queue = mp.Queue()
            procs = []
            strategy = self.config.strategy
            if strategy == "beam" and self.best:
                strategy = "genetic"
            for w in range(self.config.worker_processes):
                p = mp.Process(
                    target=run_worker_process,
                    args=(w, self.config, page_vals, ref_vals, strategy, q, beam_pool),
                )
                p.start()
                procs.append(p)

            collected = []
            for _ in procs:
                collected.extend(q.get())
            for p in procs:
                p.join()

            collected.sort(key=lambda x: x["score"], reverse=True)
            self.best = (self.best + collected)[: self.config.beam_width]

            # adaptive reward update from top candidate score parts
            if self.best:
                top = self.best[0]
                self.policy.update({
                    "modular_shift": top["parts"].get("ngram", 0.0),
                    "vigenere_shift": top["parts"].get("ioc", 0.0),
                    "atbash_variant": top["parts"].get("symmetry", 0.0),
                    "block_transposition": top["parts"].get("pattern", 0.0),
                    "index_permutation": top["parts"].get("cross_page", 0.0),
                    "unknown_transform_slot": top["parts"].get("lexical", 0.0),
                })

            beam_pool = [
                beam_factory.random_key(length=len(page_vals)) for _ in range(min(64, len(self.best) + 1))
            ]

            if it % 10 == 0:
                self._checkpoint(it)
                LOG.info("iter=%s best_score=%.5f", it, self.best[0]["score"] if self.best else 0.0)

        return self.best
