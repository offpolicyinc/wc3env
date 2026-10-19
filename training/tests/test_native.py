"""The hook's features and masks (wc3hook/features.c, masks.c) against the library's, in a live game (WC3_E2E=1): the
blocks the hook writes equal policy/native.Features on the same observations and decisions (the plumbing:
rpc.features, step's decisions, the block in the mapping).

    WC3_E2E=1 python -m unittest training.tests.test_native
"""

from __future__ import annotations

import contextlib
import io
import os
import random
import unittest

import numpy as np

from training.policy.actions import TYPES, Action
from training.policy.features import STEP_MS, Static, env_race
from training.policy.native import Features, hook_decision, hook_spec

MAP = "Maps/FrozenThrone/(2)EchoIsles.w3x"


def random_action(obs, static, rng) -> Action:
    """A decision among what the observation shows, for the features' recent-decision columns."""
    own = [int(u) for u in obs.units["unit_id"][(obs.units["flags"] & 1) != 0]] or [0]
    anyone = [int(u) for u in obs.units["unit_id"]] or [0]
    b = static.bounds
    point = rng.random() < 0.5
    return Action(TYPES[rng.randrange(1, len(TYPES))], actor=rng.choice(own), target=rng.choice(anyone),
                  x=rng.uniform(b[0], b[2]) if point else None, y=rng.uniform(b[1], b[3]) if point else None)  # fmt: skip


@unittest.skipUnless(os.environ.get("WC3_E2E"), "plays a game: set WC3_E2E=1")
class HookTest(unittest.TestCase):
    def test_the_hook_writes_the_librarys_blocks(self):
        from wc3env.session import GameConfig, GameSession, MatchSetup, PlayerConfig

        races = ("undead", "nightelf")
        statics = {0: Static.load(MAP, races), 1: Static.load(MAP, races[::-1])}
        config = GameConfig(map=MAP, players=(PlayerConfig(0, env_race(races[0])), PlayerConfig(1, env_race(races[1]))),
                            setup=MatchSetup(seed=5, randomize_starts=True), render=False, sound=False,
                            step_ms=STEP_MS, observation="binary", features={s: hook_spec(st) for s, st in statics.items()})  # fmt: skip
        library = {s: Features(st) for s, st in statics.items()}
        rng, last = random.Random(0), {0: None, 1: None}
        with contextlib.redirect_stdout(io.StringIO()), GameSession(config) as session:
            observations = session.reset()
            for t in range(300):
                for slot, obs in observations.items():
                    want = library[slot].step(obs, last[slot])
                    for k, v in want.items():
                        np.testing.assert_array_equal(obs.features[k], v, err_msg=f"step {t} slot {slot} {k}")
                last = {
                    s: random_action(o, statics[s], rng) if rng.random() < 0.75 else None
                    for s, o in observations.items()
                }
                taken = {s: hook_decision(a) for s, a in last.items() if a is not None}
                observations, done, _ = session.step({0: [], 1: []}, decisions=taken)
                if done:
                    break


if __name__ == "__main__":
    unittest.main()
