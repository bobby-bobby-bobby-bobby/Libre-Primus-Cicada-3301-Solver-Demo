from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List


@dataclass
class Constraints:
    modular_primes: List[int] = field(default_factory=lambda: [2, 3, 5, 7, 11, 13, 17, 19, 23, 29])
    offsets: List[int] = field(default_factory=lambda: list(range(-127, 128)))
    vigenere_min_len: int = 2
    vigenere_max_len: int = 12
    transposition_block_sizes: List[int] = field(default_factory=lambda: [2, 3, 4, 5, 6, 8, 9, 12])


@dataclass
class RuntimeConfig:
    data_repo_url: str = "https://github.com/solvers-hub/cicada3301.git"
    data_repo_dir: str = "data/source_repo"
    parsed_pages_dir: str = "data/parsed"
    experiments_dir: str = "experiments"
    checkpoints_dir: str = "experiments/checkpoints"
    logs_dir: str = "experiments/logs"
    top_n_per_worker: int = 32
    worker_processes: int = 4
    strategy: str = "beam"
    batch_size: int = 2048
    beam_width: int = 256
    generations: int = 60
    random_seed: int = 3301
    max_iterations: int = 1000
    constraints: Constraints = field(default_factory=Constraints)
    socket_mode: bool = False
    socket_host: str = "127.0.0.1"
    socket_port: int = 50991

    @staticmethod
    def from_path(path: Path) -> "RuntimeConfig":
        if not path.exists():
            cfg = RuntimeConfig()
            cfg.to_path(path)
            return cfg

        data = json.loads(path.read_text(encoding="utf-8"))
        cons = Constraints(**data.get("constraints", {}))
        fields: Dict[str, Any] = {k: v for k, v in data.items() if k != "constraints"}
        return RuntimeConfig(**fields, constraints=cons)

    def to_path(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        obj = self.__dict__.copy()
        obj["constraints"] = self.constraints.__dict__
        path.write_text(json.dumps(obj, indent=2), encoding="utf-8")
