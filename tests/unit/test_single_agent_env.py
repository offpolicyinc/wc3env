"""Single-agent episode behavior over the in-process game."""

import threading
import unittest
from unittest.mock import patch

from tests.fakes import FakeGame
from wc3env.env import WC3Env, run_episode
from wc3env.fake_server import FakeServer
from wc3env.protocol import Action
from wc3env.session import GameConfig


class EnvTest(unittest.TestCase):
    def test_reset_step_done_loop(self):
        env = WC3Env(GameConfig(map="m"), game_factory=FakeGame)
        with self.assertRaises(RuntimeError):
            env.step([])
        obs = env.reset()
        self.assertEqual(obs["ticks_skipped"], 0)
        pe = obs["units"][1]["unit_id"]
        obs2, done, info = env.step([Action(pe, "move", {"x": -4500.0, "y": 2800.0})])
        self.assertFalse(done)
        self.assertEqual(info["rejected"], [])
        self.assertEqual(obs2["game_time_seconds"], 1.0)
        self.assertEqual(env.steps, 1)
        env.close()

    def test_realtime_mode_never_steps_and_counts_skipped_ticks(self):
        server = FakeServer()
        env = WC3Env(GameConfig(map="m", mode="realtime"), game_factory=lambda cfg: FakeGame(cfg, server))
        env.reset()
        server.world.advance(3000)  # the game moved on by itself
        obs, _, info = env.step([])
        self.assertEqual(obs["game_time_seconds"], 3.0)
        self.assertEqual(info["ticks_skipped"], 2)
        self.assertNotIn("step", [r["method"] for r in server.log])

    def test_episode_ends_with_the_result(self):
        server = FakeServer()
        env = WC3Env(GameConfig(map="m"), game_factory=lambda cfg: FakeGame(cfg, server))

        class Idle:
            def act(self, observation):
                if observation.sequence == 3:
                    server.world.result = "victory"  # the game ends during the next step
                return []

        records = run_episode(env, Idle(), max_steps=100)
        self.assertTrue(env.done)
        self.assertEqual(records[-1]["observation"]["result"], "victory")
        self.assertEqual(len(records), 4)

    def test_run_episode_records_agent_protocol_errors_and_continues(self):
        class Bad:
            def __init__(self):
                self.n = 0

            def act(self, observation):
                self.n += 1
                return [Action(999, "stop", {})] if self.n == 1 else []

        env = WC3Env(GameConfig(map="m"), game_factory=FakeGame)
        records = run_episode(env, Bad(), max_steps=3)
        self.assertIn("error", records[0])
        self.assertEqual(env.steps, 3)  # the failed decision still costs one (empty) step

    def test_episode_accepts_and_records_dictionary_actions(self):
        class Agent:
            def act(self, obs):
                return [{"unit_id": min(obs.unit_ids), "command": "stop"}]

        with WC3Env(GameConfig(map="m"), game_factory=FakeGame) as env:
            records = run_episode(env, Agent(), 2)
        self.assertEqual(len(records), 2)
        self.assertEqual(records[0]["actions"][0]["arguments"], {})

    def test_short_step_faults_session_and_reset_replaces_process(self):
        with WC3Env(GameConfig(map="m"), game_factory=FakeGame) as env:
            env.reset()
            game = env.session.game
            with patch.object(game.rpc, "step", return_value={"game_time_ms": 0, "reason": "target"}):
                with self.assertRaisesRegex(RuntimeError, "advanced 0 ms"):
                    env.step([])
            with self.assertRaisesRegex(RuntimeError, "reset"):
                env.step([])
            env.reset()
            self.assertTrue(game.closed)
            self.assertIsNot(env.session.game, game)

    def test_result_between_calls_returns_final_observation(self):
        with WC3Env(GameConfig(map="m"), game_factory=FakeGame) as env:
            env.reset()
            env.session.game.rpc.debug("end")
            obs, done, info = env.step([])
            self.assertTrue(done)
            self.assertEqual(obs["result"], "victory")
            self.assertEqual((info["elapsed_ms"], info["step_reason"]), (0, "game_over"))

    def test_close_wakes_a_blocked_step(self):
        entered, cancelled = threading.Event(), threading.Event()

        class BlockingGame(FakeGame):
            def close(self):
                self.closed = True
                cancelled.set()

        with WC3Env(GameConfig(map="m"), game_factory=BlockingGame) as env:
            env.reset()

            def blocked(ms, **_):
                entered.set()
                if not cancelled.wait(2):
                    raise AssertionError("close did not cancel I/O")
                raise ConnectionError("closed")

            errors = []

            def step():
                try:
                    env.step([])
                except Exception as exc:
                    errors.append(exc)

            with patch.object(env.session.game.rpc, "step", side_effect=blocked):
                thread = threading.Thread(target=step)
                thread.start()
                self.assertTrue(entered.wait(1))
                env.close()
                thread.join(1)
            self.assertFalse(thread.is_alive())
            self.assertIsInstance(errors[0], ConnectionError)
