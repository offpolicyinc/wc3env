"""Real observations for the benchmarks and tests: 8 local games of bc10 against itself for 400 steps, each game's
player-0 observations in order, pickled to build/obs.pkl (a seat's features need the game's earlier steps).

    python -m training.bench.record_obs
"""

from __future__ import annotations

import pickle

from training.bench.common import CHECKPOINT, MAP, OBSERVATIONS
from training.policy.agent import Agents, Seat
from training.policy.features import STEP_MS, Static, env_race
from training.policy.model import load
from training.policy.native import hook_decision, hook_spec
from wc3env.session import GameConfig, MatchSetup, PlayerConfig
from wc3env.vector import VectorSession, local_workers

GAMES, STEPS = 8, 400


def main() -> None:
    static = Static.load(MAP, ("orc", "human"))
    configs = [GameConfig(map=MAP, players=(PlayerConfig(0, env_race("orc")), PlayerConfig(1, env_race("human"))),
                          setup=MatchSetup(seed=k, randomize_starts=True), render=False, sound=False, step_ms=STEP_MS,
                          observation="binary", features={0: hook_spec(static)}) for k in range(GAMES)]  # fmt: skip
    agents = Agents(load(str(CHECKPOINT)), 1.0, 0)
    seats = [Seat(static) for _ in range(GAMES)]
    traces = [[] for _ in range(GAMES)]
    with VectorSession(configs, group_size=4, launch=local_workers) as session:
        observations = dict(session.reset())
        for _ in range(STEPS):
            obs = [observations[i][0] for i in range(GAMES)]
            for i in range(GAMES):
                traces[i].append(obs[i])
            for i, d in enumerate(agents.act(seats, obs, [o.features for o in obs])):
                taken = {0: hook_decision(d.action)} if d.action is not None else None
                session.send(i, {0: d.commands, 1: []}, taken)
            for _ in range(GAMES):
                i, ob, _, _ = session.recv()
                observations[i] = ob
    OBSERVATIONS.parent.mkdir(exist_ok=True)
    pickle.dump(traces, open(OBSERVATIONS, "wb"))
    print(f"{sum(map(len, traces))} observations in {OBSERVATIONS}")


if __name__ == "__main__":
    main()
