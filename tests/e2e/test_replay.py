"""Replay playback: our own recordings and a 1.29 ladder replay play as recorded while observed.

Fixtures hold native replay data, no map assets, on the pinned build's stock Echo Isles. self_play.w3g
(1.3 KiB) has two agents moving, no setup/debug mutations; its digest covers both players at every second
from 1 through 51. ladder_hvh.w3g is a Human-vs-Human ladder game recorded by an observer.
"""

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from wc3env.game import launch
from wc3env.rpc import RpcError
from wc3env.session import GameConfig, GameSession, MatchSetup, PlayerConfig

from .support import state

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"


class ReplayTest(unittest.TestCase):
    def play(self, replay, slots):
        """A replay's game, created for `slots` and running fast; closed after the test."""
        game = launch(map=replay, agents=slots, render=False)
        self.addCleanup(game.close)
        game.rpc.create_game(str(replay), [{"slot": p, "control": "agent"} for p in slots])
        game.rpc.debug("speed", factor=512)
        game.rpc.debug("waitfloor", ms=1)
        return game.rpc

    def test_recorded_two_player_actions_and_eof(self):
        replay = FIXTURES / "self_play.w3g"
        game = launch(map=replay, agents=(0, 1), render=False)
        self.addCleanup(game.close)
        rpc = game.rpc
        players = [{"slot": p, "control": "agent"} for p in (0, 1)]
        with self.assertRaisesRegex(RpcError, "requires stepping mode"):
            rpc.create_game(str(replay), players, "realtime")
        rpc.create_game(str(replay), players)
        rpc.debug("speed", factor=64)
        rpc.debug("waitfloor", ms=1)
        trace = [state({p: rpc.observe(p) for p in (0, 1)})]
        with self.assertRaisesRegex(RpcError, "replay actions"):
            rpc.act(0, [])
        for tick in range(50):
            result = rpc.step(1000)
            self.assertEqual(result["elapsed_ms"], 1000)
            self.assertEqual(result["reason"], "replay_end" if tick == 49 else "target")
            trace.append(state({p: rpc.observe(p) for p in (0, 1)}))
        digest = hashlib.sha256(json.dumps(trace, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        self.assertEqual(digest, "362f4b20a9254551b9d0b29ed9bfdcfcebf7c3470381147294538e1d9ac09649")
        self.assertEqual(rpc.status, "ended")
        with self.assertRaises(RpcError):
            rpc.step(1000)
        with self.assertRaisesRegex(RpcError, "replay reset requires a new process"):
            rpc.reset()
        self.assertEqual(state({p: rpc.observe(p) for p in (0, 1)}), trace[-1])

    def test_ladder_replay_plays_as_recorded_while_observed(self):
        """A 1.29 ladder replay recorded by an observer (Human vs Human, Echo Isles): players 1 and 2 read as left
        at load. Observing and capturing events must not create game objects, or the engine reuses object ids
        differently and the recorded orders miss their units: the players then never get past 12 food."""
        rpc = self.play(FIXTURES / "ladder_hvh.w3g", (1, 2))
        for k in range(480 // 5):  # 8 minutes
            rpc.step(5000, observe=[1, 2])
            if k % 12 == 0:
                rpc.observe(1)  # the JSON observation reads map bounds too
        o = rpc.observe(1)
        types = {u["type_id"] for u in o["units"]}
        self.assertTrue({"Hamg", "Hmkg", "hfoo", "hkee"} <= types, sorted(types))
        self.assertGreaterEqual(o["player"]["food_used"], 30)

    def test_a_staged_episode_replays_to_the_same_units(self):
        """Orc (the insane AI playing our slot) against a human computer, from a fixed setup, with events and
        observations every step; the replay, started from the setup save_replay keeps beside it, reaches the
        same units, ids and hp."""
        config = GameConfig(
            map="(2)EchoIsles.w3x",
            players=(PlayerConfig(0, "orc"), PlayerConfig(1, "human", control="computer")),
            ai_agents=(0,),
            ai_difficulty=2,
            setup=MatchSetup(seed=3),
            render=False,
            sound=False,
            step_ms=250,
            observation="binary",
        )
        replay = Path(self.enterContext(tempfile.TemporaryDirectory())) / "episode.w3g"
        with GameSession(config) as session:
            session.reset()
            session.debug("speed", factor=256)
            for _ in range(4 * 180):  # 3 minutes
                session.step({0: []})
            live = session.game.rpc.observe(0)
            session.save_replay(replay)
        rpc = self.play(replay, (0,))
        while rpc.observe(0)["game_time_seconds"] < live["game_time_seconds"]:
            rpc.step(250, observe=[0])
        back = rpc.observe(0)

        def units(o):
            return sorted((u["unit_id"], u["type_id"], u["hp"]) for u in o["units"] + o["inside"])

        self.assertEqual(units(back), units(live))
        self.assertGreater(len(units(live)), 15)

    def test_session_step_stops_at_replay_end_without_inventing_results(self):
        replay = FIXTURES / "self_play.w3g"
        with GameSession(
            GameConfig(
                map=str(replay),
                players=(PlayerConfig(0), PlayerConfig(1)),
                render=False,
                step_ms=60000,
                max_episodes_per_process=None,
            )
        ) as session:
            pids = []
            starts = []
            for _ in range(2):
                starts.append(state(session.reset()))
                pids.append(session.game.pid)
                session.game.rpc.debug("speed", factor=64)
                session.game.rpc.debug("waitfloor", ms=1)
                observations, done, info = session.step({0: [], 1: []})
                self.assertTrue(done)
                self.assertEqual((info["elapsed_ms"], info["step_reason"]), (50000, "replay_end"))
                self.assertEqual([observations[p]["result"] for p in (0, 1)], ["", ""])
            self.assertNotEqual(*pids)
            self.assertEqual(*starts)


if __name__ == "__main__":
    unittest.main()
