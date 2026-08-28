"""E_D: the static test environment (the degenerate, non-adaptive case).

Serves a fixed labeled item set with seeded private coins (omega_E). Query-level
labels: z_i is fixed in advance from the item, independent of the response.
Construct from a list of items, or load a JSONL fixture with `from_jsonl`.
"""

from __future__ import annotations

import json
import pathlib
import random

from ct_core.interfaces import EnvironmentTrial, TestEnvironment


class _StaticTrial(EnvironmentTrial):
    def __init__(self, item: dict):
        self._item = item

    def next_query(self, transcript):
        return self._item["x"] if not transcript else None

    def score(self, transcript):
        if len(transcript) != 1:
            raise ValueError("static trial requires exactly one response")
        return self._item["z"]


class StaticEnvironment(TestEnvironment):
    DRIVER = "static/v1"

    def __init__(self, items: list[dict], n: int, seed: int):
        rng = random.Random(seed)  # omega_E, held only by the certifier
        self._items = self._validate_items(rng.sample(items, k=min(n, len(items))))

    @staticmethod
    def _validate_items(items) -> list[dict]:
        if not isinstance(items, list) or not items:
            raise ValueError("static environment requires a nonempty item list")
        normalized = []
        for item in items:
            if (not isinstance(item, dict) or not isinstance(item.get("x"), str)
                    or item.get("z") not in (0, 1)):
                raise ValueError("each item must contain a string x and binary z")
            normalized.append({"x": item["x"], "z": int(item["z"])})
        return normalized

    def package_config(self) -> dict:
        return {"items": self._items}

    @classmethod
    def from_config(cls, config: dict) -> "StaticEnvironment":
        if not isinstance(config, dict) or set(config) != {"items"}:
            raise ValueError("malformed static environment configuration")
        environment = cls.__new__(cls)
        environment._items = cls._validate_items(config["items"])
        return environment

    def runtime_driver(self):
        return StaticEnvironment

    def trial_count(self) -> int:
        return len(self._items)

    def max_steps(self) -> int:
        return 1

    def new_trial(self, index: int, randomness: bytes) -> EnvironmentTrial:
        if index < 1:
            raise ValueError("trial index outside the configured item set")
        try:
            return _StaticTrial(self._items[index - 1])
        except IndexError as error:
            raise ValueError("trial index outside the configured item set") from error

    @classmethod
    def from_jsonl(cls, path: str | pathlib.Path, n: int, seed: int) -> "StaticEnvironment":
        items = [
            json.loads(line)
            for line in pathlib.Path(path).read_text().splitlines()
            if line.strip()
        ]
        return cls(items, n, seed)
