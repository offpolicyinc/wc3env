"""An inference process's start with the step compiled (training/bench/compile_cost.py runs it in fresh processes).

python -c "from training.bench.compile_start import start; start()"
"""

from __future__ import annotations

from training.bench.common import MAP


def start() -> None:
    """A Model(compile=True) making its shapes from one batch of the recorded games' layout."""
    import pickle

    import torch

    from training.policy.agent import Agents, Model, Seat
    from training.policy.features import Static
    from training.policy.model import load
    from training.policy.native import Features

    traces = pickle.load(open("/root/obs.pkl", "rb"))
    static = Static.load(MAP, ("orc", "human"))
    model = Model(load("/root/model.pt", "cuda"), 1.0, 0, compile=True)

    class Here:
        def send(self, x):
            self.out = model.run(x)

        def receive(self):
            return self.out

    seats = [Seat(static) for _ in range(40)]
    states = [Features(static) for _ in range(40)]
    obs = [traces[i % len(traces)][0] for i in range(40)]
    blocks = [s.step(o, None) for s, o in zip(states, obs)]
    Agents(remote=Here()).act(seats, obs, blocks, [b["scalars"] for b in blocks])
    torch.cuda.synchronize()
    print(model.stats(), flush=True)
