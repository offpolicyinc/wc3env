"""Load training/replays/dataset.py files as model inputs: chunks of consecutive steps, arrays padded to fixed shapes.

    data = ReplayData("runs/dataset/12505_1.npz")
    chunk = data.chunk(start=100, length=64)          # dict of numpy arrays [64, ...]
    batch = collate([chunk, ...])                     # dict of torch tensors [B, 64, ...]

A chunk carries the map's ingredients, not the map: each step's remembered `trees` and `recent` decision cells,
and the file's `static_map` (walkable, buildable, starts, mines) once. The model draws the map from them and the
unit rows on the GPU (model.draw_map), the same grid features.build_map makes when playing; `step(t, map=True)`
draws it on the CPU with build_map instead, for the checks. Labels gain what the heads need: `aim` (0 none,
1 point, 2 unit), `grouped` (whether same_type and size apply), `region` and `cell` (the point as one of 64
patches and a cell inside it).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ..policy.actions import PRODUCTION, SINGLE, TYPES
from ..policy.features import COLUMN, HISTORY, MAP, MAX_UNITS, PATCH, UNIT_FLOATS, Static, build_map
from ..policy.masks import argument_ids
from ..policy.plan import empty as empty_plan
from ..policy.plan import plan
from ..replays.dataset import ID_FIELDS

UNGROUPED = {TYPES.index(t) for t in ("noop", "buy", *SINGLE, *PRODUCTION)}


class ReplayData:
    """One data set file, or the same arrays in memory (a self-play chunk: training/selfplay/actor.py), which
    may carry each step's `recent` decision cells itself (its earlier steps are not there to recompute them)."""

    def __init__(self, source: str | Path | dict):
        if isinstance(source, dict):
            self.d = source
        else:
            with np.load(source) as d:
                self.d = {k: d[k] for k in d.files}
        self.steps = len(self.d["type"])
        self.offsets = np.concatenate([[0], np.cumsum(self.d["unit_count"].astype(np.int64))])
        grid = lambda name: self.d[f"static_{name}"].astype(np.float32)  # noqa: E731
        self.static = Static(
            tuple(self.d["bounds"].tolist()),
            tuple(self.d["races"].tolist()),
            walkable=grid("walkable") / 255,
            buildable=grid("buildable") / 255,
            mines=grid("mines"),
            starts=grid("starts"),
        )
        self.static_map = np.stack(
            [self.static.walkable, self.static.buildable, self.static.starts, self.static.mines]
        ).astype(np.float32)
        if "plan_heroes" in self.d:  # a self-play chunk: the plan its player was given
            self.plan = {k: self.d[f"plan_{k}"] for k in ("heroes", "order", "mix")}
        else:
            self.plan = plan(self.d["type"], self.d["argument"])
        # the points of the last HISTORY decisions before each step (features' memory.recent)
        if "recent" in self.d:
            self.recent = [[int(c) for c in row if c >= 0] for row in self.d["recent"]]
        else:
            self.recent, last = [], []
            for t in range(self.steps):
                self.recent.append([c for c in last if c >= 0])
                if self.d["type"][t] != 0:
                    last = [int(self.d["point"][t]), *last][:HISTORY]

    def step(self, t: int, map: bool = False) -> dict:
        n = int(self.d["unit_count"][t])
        a, b = self.offsets[t], self.offsets[t + 1]
        out = {"unit_count": np.int64(n)}
        units = np.zeros((MAX_UNITS, UNIT_FLOATS), np.float16)
        units[:n] = self.d["units"][a:b]
        out["units"] = units
        for k in ID_FIELDS:
            v = self.d[k][a:b]
            padded = np.zeros((MAX_UNITS, *v.shape[1:]), np.int32)
            padded[:n] = v
            out[k] = padded
        if map:
            out["map"] = build_map(
                n,
                out["unit_cell"],
                out["unit_owner"],
                out["unit_type"],
                units[:, COLUMN["structure"]].astype(np.float32),
                self.d["trees"][t].astype(np.float32),
                self.static,
                self.recent[t],
            )
        else:
            out["trees"] = self.d["trees"][t]
            recent = np.full(HISTORY, -1, np.int32)
            recent[: len(self.recent[t])] = self.recent[t]
            out["recent"] = recent
        out["scalars"] = self.d["scalars"][t].astype(np.float32)
        out["previous_type"] = self.d["previous_type"][t].astype(np.int64)
        n_args = len(argument_ids()) + 1
        out["type_mask"] = self.d["type_mask"][t]
        out["argument_mask"] = np.unpackbits(self.d["argument_mask"][t])[:n_args].astype(bool)
        out["actor_mask"] = np.unpackbits(self.d["actor_mask"][t])[:MAX_UNITS].astype(bool)
        out["target_mask"] = np.unpackbits(self.d["target_mask"][t])[:MAX_UNITS].astype(bool)
        kind = int(self.d["type"][t])
        point, target = int(self.d["point"][t]), int(self.d["target"][t])
        out["type"] = np.int64(kind)
        out["argument"] = np.int64(self.d["argument"][t])
        out["aim"] = np.int64(1 if point >= 0 else 2 if target >= 0 else 0)
        out["actor"] = np.int64(self.d["actor"][t])
        out["target"] = np.int64(target)
        out["same_type"] = np.int64(self.d["same_type"][t])
        out["queued"] = np.int64(self.d["queued"][t]) if "queued" in self.d else np.int64(0)
        out["size"] = np.int64(self.d["size"][t])
        out["grouped"] = np.bool_(kind not in UNGROUPED)
        py, px = (point // MAP, point % MAP) if point >= 0 else (-1, -1)
        out["region"] = np.int64((py // PATCH) * (MAP // PATCH) + px // PATCH if point >= 0 else -1)
        out["cell"] = np.int64((py % PATCH) * PATCH + px % PATCH if point >= 0 else -1)
        out["result"] = np.int64(self.d["result"])
        return out

    def unit_rows(self, start: int, length: int) -> dict:
        """The unit rows of steps start.. start + length - 1 unpadded: `step` and `row` [R] (where each goes in a
        [length, MAX_UNITS] grid), `units` [R, UNIT_FLOATS] and each of ID_FIELDS [R, ...]."""
        end = min(start + length, self.steps)
        t = max(end - start, 0)
        counts = self.d["unit_count"][start : start + t].astype(np.int64)
        a, b = (int(self.offsets[start]), int(self.offsets[end])) if t else (0, 0)
        out = {"step": np.repeat(np.arange(t), counts), "row": np.arange(b - a) - np.repeat(self.offsets[start : start + t] - a, counts)}
        out["units"] = self.d["units"][a:b]
        for k in ID_FIELDS:
            out[k] = self.d[k][a:b]
        return out

    def chunk(self, start: int, length: int, with_plan: bool | tuple = True, rows: bool = True) -> dict:
        """Steps start.. start + length - 1; past the game's end, zeros with `step_valid` false. Also the file's
        static_map and its plan (the empty plan when not with_plan; a pair (order, made): those parts of it only). The same arrays as stacking step(t) for each
        step, made for the whole chunk at once (step by step it was half of a self-play update's time). With
        rows=False, without the unit rows padded to MAX_UNITS (`units` and ID_FIELDS): unit_rows() gives them
        unpadded, for the learner to place on the GPU."""
        d, end = self.d, min(start + length, self.steps)
        t = max(end - start, 0)
        steps = slice(start, start + t)
        counts = d["unit_count"][steps].astype(np.int64)
        a, b = int(self.offsets[start]) if t else 0, int(self.offsets[end]) if t else 0
        step_of = np.repeat(np.arange(t), counts)
        row_of = np.arange(b - a) - np.repeat(self.offsets[start : start + t] - a, counts)
        out = {"unit_count": counts}
        if rows:
            units = np.zeros((t, MAX_UNITS, UNIT_FLOATS), np.float16)
            units[step_of, row_of] = d["units"][a:b]
            out["units"] = units
            for k in ID_FIELDS:
                v = d[k]
                padded = np.zeros((t, MAX_UNITS, *v.shape[1:]), np.int32)
                padded[step_of, row_of] = v[a:b]
                out[k] = padded
        out["trees"] = d["trees"][steps]
        recent = np.full((t, HISTORY), -1, np.int32)
        for j, cells in enumerate(self.recent[start : start + t]):
            recent[j, : len(cells)] = cells
        out["recent"] = recent
        out["scalars"] = d["scalars"][steps].astype(np.float32)
        out["previous_type"] = d["previous_type"][steps].astype(np.int64)
        n_args = len(argument_ids()) + 1
        out["type_mask"] = d["type_mask"][steps]
        out["argument_mask"] = np.unpackbits(d["argument_mask"][steps], axis=-1)[:, :n_args].astype(bool)
        out["actor_mask"] = np.unpackbits(d["actor_mask"][steps], axis=-1)[:, :MAX_UNITS].astype(bool)
        out["target_mask"] = np.unpackbits(d["target_mask"][steps], axis=-1)[:, :MAX_UNITS].astype(bool)
        kind = d["type"][steps].astype(np.int64)
        point, target = d["point"][steps].astype(np.int64), d["target"][steps].astype(np.int64)
        out["type"] = kind
        out["argument"] = d["argument"][steps].astype(np.int64)
        out["aim"] = np.where(point >= 0, 1, np.where(target >= 0, 2, 0)).astype(np.int64)
        out["actor"] = d["actor"][steps].astype(np.int64)
        out["target"] = target
        out["same_type"] = d["same_type"][steps].astype(np.int64)
        out["queued"] = d["queued"][steps].astype(np.int64) if "queued" in d else np.zeros(t, np.int64)
        out["size"] = d["size"][steps].astype(np.int64)
        out["grouped"] = ~np.isin(kind, list(UNGROUPED))
        py, px = point // MAP, point % MAP
        out["region"] = np.where(point >= 0, (py // PATCH) * (MAP // PATCH) + px // PATCH, -1).astype(np.int64)
        out["cell"] = np.where(point >= 0, (py % PATCH) * PATCH + px % PATCH, -1).astype(np.int64)
        out["result"] = np.full(t, int(d["result"]), np.int64)
        missing = length - t
        if missing > 0:
            out = {k: np.concatenate([v, np.zeros((missing, *v.shape[1:]), v.dtype)]) for k, v in out.items()}
        out["step_valid"] = np.arange(length) < t
        out["static_map"] = self.static_map
        order, made = with_plan if isinstance(with_plan, tuple) else (with_plan, with_plan)
        none = empty_plan()
        for k, v in self.plan.items():
            out[f"plan_{k}"] = v if (order if k == "order" else made) else none[k]
        return out


def collate(chunks: list[dict]):
    """Stack equal-length chunks into a batch of torch tensors [B, T, ...]."""
    import torch

    return {k: torch.from_numpy(np.stack([c[k] for c in chunks])) for k in chunks[0]}


def sample(files: list[ReplayData], batch: int, length: int, rng: np.random.Generator) -> list[dict]:
    """`batch` chunks of `length` steps from random files and starts."""
    out = []
    for _ in range(batch):
        data = files[rng.integers(len(files))]
        start = int(rng.integers(max(1, data.steps - length + 1)))
        out.append(data.chunk(start, length))
    return out
