"""The model's features and masks of an observation, from their one implementation (wc3hook/features.c and
masks.c, docs/training/12-hook-features.md):

- live play: the game hook computes them with each observation; a session asks with
  GameConfig(features={slot: hook_spec(static)}) and sends each step's decisions (hook_decision), and the block
  arrives as BinaryObservation.features;
- observations already recorded (data set conversion, benchmarks, tests): the same C as a library, here.

    state = Features(static)                 # one per player per game
    block = state.step(obs, last_action)     # wc3env.binary.parse_features' dict, as the hook's
    masks = chosen_masks(block, kind, argument)   # the masks a choice was made under, as the data set stores them

The library is built on first use when missing (training/native/build_features.bat on Windows, cc on Linux).
"""

from __future__ import annotations

import ctypes as C
import os
import subprocess
import sys
import threading
from functools import cache
from pathlib import Path

import numpy as np

from wc3env.binary import FEAT_MAX_UNITS, FEAT_N_ARGS, parse_features

from .actions import ANY_ARGUMENT, HAS_ARGUMENT, TYPES, Action
from .features import MAP, RACES, Static

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "training" / "native" / "out"
SOURCES = [ROOT / "wc3hook" / "features.c", ROOT / "wc3hook" / "masks.c"]
WRITE_CAP = 4 << 20  # a features block is at most ~0.3 MB
_LOCK = threading.Lock()  # the C keeps its scratch in static buffers (one game thread in the hook): one call at a time


class _Static(C.Structure):
    _fields_ = [("bounds", C.c_double * 4), ("races", C.c_int * 2), ("trees", C.c_float * (MAP * MAP))]


class _Decision(C.Structure):
    _fields_ = [("type", C.c_int), ("actor", C.c_uint32), ("target", C.c_uint32), ("has_point", C.c_int),
                ("x", C.c_double), ("y", C.c_double)]  # fmt: skip


@cache
def _library() -> C.CDLL:
    path = OUT / ("features.dll" if sys.platform == "win32" else "libwc3features.so")
    if not path.exists() or path.stat().st_mtime < max(s.stat().st_mtime for s in SOURCES):
        _build(path)
    lib = C.CDLL(str(path))
    lib.feat_new.restype, lib.feat_new.argtypes = C.c_void_p, [C.POINTER(_Static)]
    lib.feat_free.argtypes = [C.c_void_p]
    lib.feat_step.restype = C.c_int
    lib.feat_step.argtypes = [C.c_void_p, C.c_char_p, C.c_size_t, C.POINTER(_Decision), C.c_void_p]
    lib.feat_write.restype, lib.feat_write.argtypes = C.c_uint32, [C.c_void_p, C.c_void_p, C.c_uint32]
    lib.feat_out_size.restype = C.c_size_t
    return lib


def _build(path: Path) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    if sys.platform == "win32":
        subprocess.run(["cmd", "/c", str(ROOT / "training" / "native" / "build_features.bat")], check=True,
                       capture_output=True)  # fmt: skip
    else:  # into a file of this process's own, then renamed: several processes may build at once
        part = path.with_name(f"{path.name}.{os.getpid()}")
        cc = os.environ.get("CC", "cc")
        subprocess.run([cc, "-O2", "-shared", "-fPIC", "-DFEAT_DLL", "-o", str(part), *map(str, SOURCES), "-lm"],
                       check=True)  # fmt: skip
        os.replace(part, path)


@cache
def _buffers() -> tuple:
    """One output struct and one write buffer, shared (the C uses static scratch too: one thread at a time)."""
    lib = _library()
    return C.create_string_buffer(lib.feat_out_size()), C.create_string_buffer(WRITE_CAP)


def decision(action: Action | None) -> _Decision | None:
    if action is None:  # a noop: nothing to fold in
        return None
    point = action.x is not None
    return _Decision(TYPES.index(action.type), action.actor or 0, action.target or 0, point,
                     action.x if point else 0.0, action.y if point else 0.0)  # fmt: skip


class Features:
    """One player's features over a game, from the C library: step() with each observation (in order) and the
    action decided on the observation before (None for a noop or the first)."""

    def __init__(self, static: Static):
        s = _Static()
        s.bounds[:] = static.bounds
        s.races[:] = [RACES.index(r) for r in static.races]
        if static.trees is not None:  # else none (a map without its data)
            np.ctypeslib.as_array(s.trees)[:] = static.trees.ravel()
        self.lib = _library()
        self.state = self.lib.feat_new(C.byref(s))

    def step(self, obs, last: Action | None) -> dict:
        out, buffer = _buffers()
        d = decision(last)
        with _LOCK:
            if self.lib.feat_step(self.state, obs.raw, len(obs.raw), C.byref(d) if d else None, out):
                raise ValueError("not a binary observation of this wc3env version")
            size = self.lib.feat_write(out, buffer, WRITE_CAP)
            block = C.string_at(buffer, size)
        return parse_features(block)

    def close(self) -> None:
        if self.state:
            self.lib.feat_free(self.state)
            self.state = None

    def __del__(self):
        self.close()


def hook_spec(static: Static) -> dict:
    """What GameConfig.features takes for a slot (rpc.features): the map's bounds, both races, its starting trees."""
    return {"bounds": list(static.bounds), "races": [RACES.index(r) for r in static.races],
            "trees": np.asarray(static.trees if static.trees is not None else np.zeros((MAP, MAP)), np.uint8).tobytes()}  # fmt: skip


def hook_decision(action: Action | None) -> dict | None:
    """The step's `decisions` entry for a slot (VectorSession.send, GameSession.step), or None for a noop."""
    if action is None:
        return None
    out = {"type": TYPES.index(action.type), "actor": action.actor or 0, "target": action.target or 0}
    if action.x is not None:
        out.update(x=action.x, y=action.y)
    return out


def chosen_masks(block: dict, kind: int, argument: int) -> dict:
    """The masks a choice of type `kind` and argument index `argument` was made under, as the data set stores
    them (bool arrays: type, argument [FEAT_N_ARGS], actor [256], target [256]); a noop's for kind 0."""
    name = TYPES[kind]
    args = np.zeros(FEAT_N_ARGS, bool)
    actor = np.zeros(FEAT_MAX_UNITS, bool)
    if kind == 0 or name not in HAS_ARGUMENT:
        args[0] = True  # no argument
    else:
        where = np.nonzero(block["argument_types"] == kind)[0]
        if len(where):
            args[:] = np.unpackbits(block["argument_bits"][where[0]])[:FEAT_N_ARGS].astype(bool)
    if kind:
        key = kind * FEAT_N_ARGS + (0 if name in ANY_ARGUMENT else argument)
        where = np.nonzero(block["actor_keys"] == key)[0]
        if len(where):
            actor[:] = np.unpackbits(block["actor_rows"][where[0]]).astype(bool)
    target = np.arange(FEAT_MAX_UNITS) < block["unit_count"]
    return {"type": block["type_mask"], "argument": args, "actor": actor, "target": target}


def tables(block: dict) -> dict:
    """The block's masks as decide.pack takes them."""
    return {"type": block["type_mask"], "argument_types": block["argument_types"].astype(np.int8),
            "argument": block["argument_bits"], "actor_keys": block["actor_keys"].astype(np.int64),
            "actor_rows": block["actor_rows"]}  # fmt: skip
