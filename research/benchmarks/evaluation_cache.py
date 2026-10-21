from __future__ import annotations

from collections import OrderedDict
from functools import partial

import numpy as np

from . import classifier, pools
from .common import read_json, write_json


class EvaluationCache:
    def __init__(self, capacity=6):
        self.capacity = capacity
        self.entries = OrderedDict()
        self.hits = 0
        self.misses = 0
        self.functions = {"target": pools.target_pool, "seen": pools.seen_pool}

    def load(self, role, run, job, pool, frozen_sha256, *, output_folder=None):
        if not frozen_sha256:
            raise RuntimeError("Evaluation data requires a frozen classifier")
        folder = output_folder or run / "jobs" / job["id"]
        key = (role, str(run.resolve()), job["track"], job["device"], job["channel"],
               job["target"], job["split"], pool["source_identity"])
        if key in self.entries:
            self.hits += 1
            values, receipt = self.entries.pop(key)
            write_json(folder / (role + "_manifest.json"), {**receipt, "classifier_sha256": frozen_sha256})
        else:
            self.misses += 1
            values = self.functions[role](run, job, pool, frozen_sha256, output_folder=folder)
            receipt = read_json(folder / (role + "_manifest.json"))
        self.entries[key] = (values, receipt)
        while len(self.entries) > self.capacity:
            self.entries.popitem(last=False)
        return values

    def statistics(self):
        return {"hits": self.hits, "misses": self.misses, "capacity": self.capacity,
                "cached_array_bytes": sum(value.nbytes for values, receipt in self.entries.values()
                                          for value in values if isinstance(value, np.ndarray))}

    def install(self):
        classifier.target_pool = partial(self.load, "target")
        classifier.seen_pool = partial(self.load, "seen")
