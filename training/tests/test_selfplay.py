"""Self-play end to end, with a game installed (WC3_E2E=1): a short local run (learner, inference process, one actor
against Warcraft's AI handicapped) trains, pays every reward channel, counts its games in the league and saves; and the
learner recomputes exactly the log-probabilities the actor sampled with.

    WC3_E2E=1 python -m unittest training.tests.test_selfplay
"""

from __future__ import annotations

import contextlib
import io
import os
import re
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from training.policy.agent import Agents, Seat
from training.policy.features import Static
from training.policy.model import Policy
from training.policy.native import hook_decision, hook_spec
from training.selfplay.actor import Recorder
from training.selfplay.league import HANDICAPS
from training.selfplay.reward import CHANNELS
from training.train.loader import ReplayData
from wc3env.session import GameConfig, GameSession, MatchSetup, PlayerConfig

MAP = "Maps/FrozenThrone/(2)EchoIsles.w3x"


@unittest.skipUnless(os.environ.get("WC3_E2E"), "plays games: set WC3_E2E=1")
class RunTest(unittest.TestCase):
    def test_a_short_run_trains_on_its_games(self):
        """Two one-minute games at a time against the easy AI at 0.2 of its hit points, until the actor has taken
        1,200 steps: updates made, every channel's reward logged, the games in the league, last.pt saved."""
        with tempfile.TemporaryDirectory() as tmp:
            init, out = Path(tmp) / "init.pt", Path(tmp) / "run"
            torch.manual_seed(0)
            torch.save({"model": Policy().state_dict()}, init)
            env = {k: v for k, v in os.environ.items() if k != "WANDB_API_KEY"}  # no charts for a test
            subprocess.run([sys.executable, "-m", "training.selfplay.run", "--init", str(init), "--out", str(out),
                            "--games", "2", "--ai-games", "2", "--races", "orc", "orc", "--steps", "1200", "--port", "7650",
                            "--", "--minutes", "1", "--batch", "2", "--minibatch", "2", "--warmup", "1", "--save-every", "1",
                            "--log-every", "1", "--stall-minutes", "0", "--no-compile"],
                           env=env, check=True, timeout=1800, capture_output=True)  # fmt: skip
            log = (out / "log.txt").read_text()
            actor = (out / "actor-0.log").read_text()
            self.assertNotIn("Traceback", actor, actor[-3000:])
            updates = re.findall(r"update (\d+) .* games (\d+) ", log)
            self.assertGreaterEqual(len(updates), 2, log[-3000:])
            self.assertGreater(int(updates[-1][1]), 0, log[-3000:])  # games ended and reached the learner
            last = [line for line in log.splitlines() if "reward_win" in line][-1]
            for name in CHANNELS:
                self.assertRegex(last, rf"reward_{name} -?[\d.e+-]+")
            self.assertNotIn("reward_hero", log)
            self.assertRegex(last, r"segments 1(\.0)?\b")  # advantages over the actor's segments (batches.score)
            saved = torch.load(out / "last.pt", map_location="cpu", weights_only=False)
            self.assertEqual(tuple(saved["critic"]), CHANNELS)
            self.assertGreaterEqual(saved["update"], 2)
            self.assertEqual(saved["league"].plan()["rung"], "ai0")
            self.assertIn(saved["league"].plan()["handicap"], HANDICAPS)


@unittest.skipUnless(os.environ.get("WC3_E2E"), "plays a game: set WC3_E2E=1")
class ConsistencyTest(unittest.TestCase):
    def test_learner_recomputes_the_actors_log_probabilities(self):
        """A self-play game's chunk, sampled by the actor's code, scored again by the learner's: equal."""
        torch.manual_seed(0)
        model = Policy().eval()
        seats = [Seat(Static.load(MAP, ("orc", "human"))), Seat(Static.load(MAP, ("human", "orc")))]
        config = GameConfig(map=MAP, players=(PlayerConfig(0, "orc"), PlayerConfig(1, "human")),
                            setup=MatchSetup(seed=3, randomize_starts=True), render=False, sound=False,
                            step_ms=250, observation="binary", features={s: hook_spec(seat.static) for s, seat in enumerate(seats)})  # fmt: skip
        with contextlib.redirect_stdout(io.StringIO()), GameSession(config) as session:
            session.reset()
            obs, _, _ = session.step({0: [], 1: []})
            agents = Agents(model, 1.0, seed=1)
            recorder = Recorder(seats[0].static, None, None, 0)
            for _ in range(40):
                blocks = [obs[0].features, obs[1].features]
                opponents = [blocks[1]["scalars"], blocks[0]["scalars"]]
                decisions = agents.act(seats, [obs[0], obs[1]], blocks, opponents)
                recorder.add(obs[0], decisions[0], opponents[0])
                taken = {s: hook_decision(d.action) for s, d in enumerate(decisions) if d.action is not None}
                obs, _, _ = session.step({0: decisions[0].commands, 1: decisions[1].commands}, decisions=taken)
        chunk = recorder.chunk(bootstrap=0.0)
        rows = ReplayData(chunk).chunk(0, 64)
        batch = {k: torch.from_numpy(v[None]) for k, v in rows.items()}
        with torch.no_grad():
            log_prob, _ = model.log_probs(batch, model(batch))
        n = len(chunk["log_prob"])
        np.testing.assert_allclose(log_prob[0, :n].numpy(), chunk["log_prob"], atol=2e-3)
        self.assertGreater(int((chunk["type"] > 0).sum()), 5)  # actions, not only noops


if __name__ == "__main__":
    unittest.main()
