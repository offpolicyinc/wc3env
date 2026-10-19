"""Watch a saved game (training/play/vs_ai.py --replays) in a Warcraft window, at a chosen speed.

    python -m training.play.watch runs/games/bc4-orc-vs-human-0.w3g [--speed 2]

By default the game runs on its own clock (realtime), so Warcraft's replay controls work: scroll the camera, change
the playback speed, pause. --speed N instead drives it step by step at N times real time (no controls: the game
waits between steps).

The game was started with a match setup (start positions, AI difficulty) that Warcraft's own replay menu does
not know, so it plays back through wc3env, which applies the setup saved next to the replay (<name>.w3g.json).
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from ..policy.features import STEP_MS, env_race


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("replay", type=Path)
    p.add_argument(
        "--speed",
        type=float,
        default=0.0,
        help="drive it step by step at this many times real time (0: Warcraft's own controls)",
    )
    p.add_argument("--races", nargs=2, default=("orc", "human"), help="the agent's, then the AI's, as when played")
    args = p.parse_args()
    from wc3env.session import GameConfig, GameSession, PlayerConfig

    config = GameConfig(
        map=str(args.replay.absolute()),
        players=(
            PlayerConfig(0, env_race(args.races[0])),
            PlayerConfig(1, env_race(args.races[1]), control="computer"),
        ),
        render=True,
        window_mode="interactive",
        sound=False,
        step_ms=STEP_MS,
        mode="stepping" if args.speed else "realtime",
    )
    with GameSession(config) as session:
        session.reset()
        done = False
        while not done:
            started = time.time()
            _, done, _ = session.step({0: []})
            time.sleep(max(0.0, STEP_MS / 1000 / args.speed - (time.time() - started)) if args.speed else 1.0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
