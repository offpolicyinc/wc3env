"""The decision path against the learner, on recorded games: the inference process's choices are scored the same
by the learner (log-probabilities), and the learner's batch equals the chunks it is built from. Needs recorded observations (python -m training.bench.record_obs) and a
cloned checkpoint (runs/bc10/step8k.pt); skipped without them.

    python -m unittest training.tests.test_inference
"""

from __future__ import annotations

import os
import pickle
import tempfile
import threading
import unittest

import numpy as np
import torch

from training.bench.common import CHECKPOINT, MAP, OBSERVATIONS, play
from training.policy.agent import UNIT_KEYS, Agents, Seat
from training.policy.features import Static
from training.policy.model import load
from training.policy.native import Features
from training.selfplay.actor import Recorder
from training.selfplay.inference import Inference
from training.selfplay.inference import parser as inference_parser
from training.selfplay.learner import Learner, parser
from training.selfplay.transport import Client, Server
from training.train.loader import ReplayData

OBSERVATIONS, CHECKPOINT = str(OBSERVATIONS), str(CHECKPOINT)


def _learner_gap(model, chunks: list[dict]) -> float:
    """The largest |learner's log-probability - the actor's| over the chunks' steps."""
    worst = 0.0
    for c in chunks:
        r = ReplayData(c).chunk(0, 64)
        batch = {k: torch.from_numpy(np.asarray(v))[None] for k, v in r.items()}
        state = None if c["start_state"] is None else [torch.from_numpy(h)[None] for h in c["start_state"]]
        with torch.no_grad():
            lp, _ = model.log_probs(batch, model(batch, state))
        n = len(c["log_prob"])
        worst = max(worst, float((lp[0, :n] - torch.from_numpy(c["log_prob"])).abs().max()))
    return worst


@unittest.skipUnless(os.path.exists(OBSERVATIONS) and os.path.exists(CHECKPOINT), "needs build/obs.pkl and bc10")
class DecisionTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.traces = pickle.load(open(OBSERVATIONS, "rb"))
        cls.static = Static.load(MAP, ("orc", "human"))

    def test_inference_process_choices_score_the_same_in_the_learner(self):
        """A learner and the inference process here; two actors (threads, a connection each) over 200 steps: the
        learner's log-probabilities on every chunk, later ones from the memory the inference process returned,
        equal the actors' (float16 storage and float rounding apart)."""
        out = tempfile.mkdtemp()
        learner = Learner(parser().parse_args(["--init", CHECKPOINT, "--out", out, "--no-compile"]))
        server = Server(("127.0.0.1", 0), b"k", learner.handle)  # free ports
        inference = Inference(inference_parser().parse_args(["--learner", "%s:%d" % server.address, "--listen",
                                                             "127.0.0.1:0", "--authkey", "k"]))  # fmt: skip
        threading.Thread(target=inference.run, daemon=True).start()
        chunks, lock = [], threading.Lock()

        class Remote:  # an actor's link to the inference process (training/selfplay/actor.py's _Inference)
            def __init__(self):
                self.client = Client(inference.address, b"k", compress=False)

            def send(self, inputs):
                self.client.send("act", ("latest", inputs))

            def receive(self):
                reply = self.client.receive()
                reply.pop("version")
                return reply

        def actor(games):
            got = play(Agents(remote=Remote()), games, 200, self.traces, self.static)
            with lock:
                chunks.extend(got)

        threads = [threading.Thread(target=actor, args=(g,)) for g in ([0, 1, 2, 3], [4, 5, 6, 7])]
        for th in threads:
            th.start()
        for th in threads:
            th.join()
        self.assertEqual(len(chunks), 24)  # 8 seats x 3 finished chunks
        self.assertGreater(sum(c["start_state"] is not None for c in chunks), 0)
        self.assertLess(_learner_gap(learner.model.eval(), chunks), 2e-2)

    def test_a_snapshot_runs_with_its_own_weights_beside_the_latest(self):
        """One actor plays a snapshot (the clone's weights moved a little), another the latest, at once, so batches
        mix them and the snapshot's weights are swapped in and out of the one model: each actor's choices score the
        same under its own weights, and the model holds the latest's afterwards."""
        out = tempfile.mkdtemp()
        learner = Learner(parser().parse_args(["--init", CHECKPOINT, "--out", out, "--no-compile"]))
        server = Server(("127.0.0.1", 0), b"k", learner.handle)
        inference = Inference(inference_parser().parse_args(["--learner", "%s:%d" % server.address, "--listen",
                                                             "127.0.0.1:0", "--authkey", "k"]))  # fmt: skip
        g = torch.Generator().manual_seed(0)
        moved = {k: v + 0.05 * v.std() * torch.randn(v.shape, generator=g) if v.is_floating_point() and v.numel() > 1 else v
                 for k, v in learner.model.state_dict().items()}  # fmt: skip
        asked = inference.client.call
        inference.client.call = lambda what, *a: {k: v.clone() for k, v in moved.items()} if what == "snapshot" else asked(what, *a)
        swaps, snapshot_of = [], inference._snapshot
        inference._snapshot = lambda policy: swaps.append(policy) or snapshot_of(policy)
        threading.Thread(target=inference.run, daemon=True).start()
        chunks, lock = {"latest": [], "sp1": []}, threading.Lock()

        class Remote:
            def __init__(self, policy):
                self.client, self.policy = Client(inference.address, b"k", compress=False), policy

            def send(self, inputs):
                self.client.send("act", (self.policy, inputs))

            def receive(self):
                reply = self.client.receive()
                reply.pop("version")
                return reply

        def actor(policy, games):
            got = play(Agents(remote=Remote(policy)), games, 200, self.traces, self.static)
            with lock:
                chunks[policy].extend(got)

        threads = [threading.Thread(target=actor, args=a) for a in (("latest", [0, 1, 2, 3]), ("sp1", [4, 5, 6, 7]))]
        for th in threads:
            th.start()
        for th in threads:
            th.join()
        self.assertGreater(len(swaps), 0)
        reference = learner.model.eval()
        self.assertLess(_learner_gap(reference, chunks["latest"]), 2e-2)
        snapshot = load({"model": moved}).eval()
        self.assertLess(_learner_gap(snapshot, chunks["sp1"]), 2e-2)
        self.assertGreater(_learner_gap(reference, chunks["sp1"]), 5e-2)  # the moved weights did change its choices
        for k, t in zip(inference.names, inference.live):
            self.assertTrue(torch.equal(t.cpu(), learner.model.state_dict()[k].cpu()), k)

    def test_learner_batch_equals_its_chunks(self):
        """Learner._batch (unit rows placed on the device) equals the chunks stacked as ReplayData pads them, a
        short chunk (padding past its end) included."""
        agents = Agents(load(CHECKPOINT), 1.0, 0)
        chunks = play(agents, list(range(4)), 129, self.traces, self.static, start=100)
        seat, state = Seat(self.static), Features(self.static)
        short = Recorder(self.static, None, None, 0)
        for t in range(20):
            obs = self.traces[5][t]
            block = state.step(obs, seat.last)
            short.add(obs, agents.act([seat], [obs], [block], [block["scalars"]])[0], block["scalars"])
        chunks.append(short.chunk(bootstrap=0.0))
        learner = Learner(parser().parse_args(["--init", CHECKPOINT, "--out", tempfile.mkdtemp(), "--no-compile"]))
        batch, _ = learner._batch(chunks)
        rows = [ReplayData(c).chunk(0, 64) for c in chunks]
        for k in rows[0]:
            want = torch.from_numpy(np.stack([r[k] for r in rows]))
            self.assertEqual(want.dtype, batch[k].dtype, k)
            self.assertTrue(torch.equal(want, batch[k].cpu()), k)

    def test_a_seat_is_the_same_alone_and_in_a_padded_batch(self):
        """policy/agent.py's Model pads a batch (seats to SEAT_SIZES, the padding seats on row 0 of its memory
        tables; unit rows to UNIT_ROWS): one seat's values over 70 steps are the same alone as among 69 other seats
        with its unit rows padded to 128, its memory and window carried across the steps either way."""
        blocks = {}  # each seat's blocks, made once without decisions: the values then depend on the model alone

        def step_blocks(g):
            if g not in blocks:
                state = Features(self.static)
                blocks[g] = [state.step(self.traces[g][t], None) for t in range(70)]
            return blocks[g]

        class Padded:  # the model, its requests' unit rows padded to 128 first
            def __init__(self, agents):
                self.model, self.out = agents.model, None

            def send(self, x):
                for k in UNIT_KEYS:
                    v = x["features"][k]
                    x["features"][k] = np.concatenate([v, np.zeros((v.shape[0], 128 - v.shape[1], *v.shape[2:]), v.dtype)], 1)
                self.out = self.model.run(x)

            def receive(self):
                return self.out

        def values(games, pad):
            agents = Agents(load(CHECKPOINT), 1.0, 0)
            if pad:
                agents.backend = Padded(agents)
            seats = [Seat(self.static) for _ in games]
            out = []
            for t in range(70):
                b = [step_blocks(g)[t] for g in games]
                d = agents.act(seats, [self.traces[g][t] for g in games], b, [x["scalars"] for x in b])
                out.append(d[0].value)
            return np.array(out)

        alone = values([0], pad=False)
        among = values([0] + [g % len(self.traces) for g in range(1, 70)], pad=True)
        np.testing.assert_allclose(among, alone, atol=1e-4)

    def test_chunk_at_once_equals_step_by_step(self):
        """ReplayData.chunk (the whole chunk at once) equals stacking ReplayData.step for each step."""
        agents = Agents(load(CHECKPOINT), 1.0, 0)
        for c in play(agents, [0, 3], 65, self.traces, self.static):
            r = ReplayData(c)
            for start in (0, 10):
                got = r.chunk(start, 64)
                steps = [r.step(t) for t in range(start, min(start + 64, r.steps))]
                for k, v in steps[0].items():
                    want = np.stack([s[k] for s in steps])
                    np.testing.assert_array_equal(got[k][: len(steps)], want, err_msg=k)
                    self.assertEqual(got[k].dtype, want.dtype, k)


if __name__ == "__main__":
    unittest.main()
