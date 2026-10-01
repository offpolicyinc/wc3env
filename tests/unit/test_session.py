"""Shared rounds, process recycling and lifecycle ownership without a game process."""

import threading
import unittest
from unittest.mock import patch

from tests.fakes import FakeGame
from wc3env.protocol import Action, ProtocolError
from wc3env.session import GameConfig, GameSession, MatchSetup, PlayerConfig


class SessionTest(unittest.TestCase):
    def setUp(self):
        self.games = []

        def launch(config):
            game = FakeGame(config)
            self.games.append(game)
            return game

        self.config = GameConfig(map="test.w3x", players=(PlayerConfig(0), PlayerConfig(1)))
        self.session = GameSession(self.config, game_factory=launch)
        self.addCleanup(self.session.close)

    def test_two_players_submit_before_one_shared_advance(self):
        observations = self.session.reset()
        actions = {p: [Action(o["units"][0]["unit_id"], "stop", {})] for p, o in observations.items()}
        server = self.games[-1].server
        server.log.clear()
        observations, done, info = self.session.step(actions)
        self.assertEqual([r["method"] for r in server.log], ["step", "observe", "observe"])  # actions ride the step
        self.assertEqual(list(server.log[0]["params"]["actions"]), ["0", "1"])
        self.assertEqual({o["game_time_seconds"] for o in observations.values()}, {1.0})
        self.assertEqual({o["sequence"] for o in observations.values()}, {1})
        self.assertEqual(info["rejected"], {0: [], 1: []})
        self.assertEqual(info["elapsed_ms"], 1000)
        self.assertFalse(done)

    def test_a_step_may_name_its_own_length(self):
        self.session.reset()
        observations, _, info = self.session.step({0: [Action(1001, "stop", {})] * 200, 1: []}, 2500)
        self.assertEqual(info["rejected"], {0: [], 1: []})
        self.assertEqual(info["elapsed_ms"], 2500)
        self.assertEqual({o["game_time_seconds"] for o in observations.values()}, {2.5})
        with self.assertRaises(ValueError):
            self.session.step({0: [], 1: []}, 30)

    def test_all_batches_are_validated_before_any_rpc(self):
        self.session.reset()
        server = self.games[-1].server
        server.log.clear()
        valid = Action(1001, "stop", {})
        for batches in (
            {0: [valid]},
            {0: [valid], 1: [Action(1001, "stop", {})]},
            {0: [valid], 1: [{}]},
            {0: [valid], 1: [Action(2000, 5, {})]},
            {0: [valid], 1: [Action(2000, "stop", [])]},
            {0: [valid], 1: [Action(True, "stop", {})]},
            {0: [valid], 1: [Action(2000, "move", {"x": float("nan"), "y": 0})]},
            {0: [valid], 1: [Action(2000, "move", {"x": object(), "y": 0})]},
            {0: [valid], 1: [Action(2000, "build", {"auto_place": 1})]},
            {0: [valid], 1: [Action(2000, "move", {"auto_place": False})]},
        ):
            with self.assertRaises(ProtocolError):
                self.session.step(batches)
        self.assertEqual(server.log, [])
        self.session.step({0: [], 1: []})
        self.assertEqual(self.session.steps, 1)

    def test_placements_are_returned_per_player_without_mutating_actions(self):
        self.session.reset()
        actions = {0: [Action(1001, "build", {"type_id": "hhou", "x": 0, "y": 0, "auto_place": True})], 1: []}
        rpc = self.session.game.rpc
        original = rpc.step

        def placed(ms, observe=None, actions=None):
            result = original(ms, observe=observe, actions=actions)
            result["acts"]["0"]["placements"] = [{"index": 0, "x": 128, "y": 256}]
            return result

        with patch.object(rpc, "step", side_effect=placed):
            _, _, info = self.session.step(actions)
        self.assertEqual(info["placements"], {0: [{"index": 0, "x": 128, "y": 256}], 1: []})
        self.assertEqual((actions[0][0].arguments["x"], actions[0][0].arguments["y"]), (0, 0))

    def test_views_share_cached_observations_without_draining_events(self):
        view = self.session.view(1)
        self.assertIsNone(view.observation)
        first = self.session.reset()[1]
        count = len(self.games[-1].server.log)
        self.assertIs(view.observation, first)
        self.assertIs(self.session.view(1).observation, first)
        self.assertEqual(len(self.games[-1].server.log), count)
        self.assertFalse(any(hasattr(view, method) for method in ("step", "reset", "close")))
        self.session.step({0: [], 1: []})
        self.assertEqual(view.observation["sequence"], 1)

    def test_reset_reuses_process_and_discards_episode_state(self):
        self.session.reset()
        self.session.step({0: [], 1: []})
        previous = self.games[-1]
        observations = self.session.reset()
        self.assertFalse(previous.closed)
        self.assertIs(self.session.game, previous)
        self.assertEqual(self.session.steps, 0)
        self.assertEqual({o["sequence"] for o in observations.values()}, {0})
        self.assertEqual({o["game_time_seconds"] for o in observations.values()}, {0.0})
        self.assertEqual(self.session.setup["map"], self.config.map)

    def test_partial_round_failure_requires_reset(self):
        self.session.reset()
        rpc = self.session.game.rpc

        def lost(ms, observe=None, actions=None):
            raise TimeoutError("lost the step's reply")

        with patch.object(rpc, "step", side_effect=lost):
            with self.assertRaises(TimeoutError):
                self.session.step({0: [Action(1001, "stop", {})], 1: [Action(2000, "stop", {})]})
        with self.assertRaisesRegex(RuntimeError, "reset"):
            self.session.step({0: [], 1: []})
        self.assertEqual(self.session.steps, 0)
        old = self.session.game
        self.session.reset()
        self.assertTrue(old.closed)
        self.assertIsNot(self.session.game, old)
        self.session.step({0: [], 1: []})

    def test_failed_reset_or_setup_closes_process_and_next_reset_relaunches(self):
        self.session.reset()
        old = self.session.game
        with patch.object(old.rpc, "reset", side_effect=TimeoutError("reload failed")):
            with self.assertRaises(TimeoutError):
                self.session.reset()
        self.assertTrue(old.closed)
        self.assertEqual(self.session.observations, {})
        self.assertIsNone(self.session.setup)
        with patch("wc3env.rpc.RpcClient.create_game", side_effect=ValueError("unsupported")):
            with self.assertRaises(ValueError):
                self.session.reset()
        self.assertTrue(self.games[-1].closed)
        self.assertIsNone(self.session.setup)
        with self.assertRaises(RuntimeError):
            _ = self.session.game
        self.session.reset()
        self.assertIsNot(self.session.game, old)

    def test_close_is_final_and_idempotent(self):
        self.session.reset()
        self.session.close()
        self.session.close()
        self.assertTrue(self.games[-1].closed)
        self.assertEqual([r["method"] for r in self.games[-1].server.log].count("quit"), 1)
        with self.assertRaisesRegex(RuntimeError, "closed"):
            self.session.reset()

    def test_close_during_launch_disposes_unpublished_process_once(self):
        entered, finish = threading.Event(), threading.Event()
        errors = []
        game = FakeGame(self.config)

        def launch(config):
            entered.set()
            if not finish.wait(2):
                raise AssertionError("launch was not released")
            return game

        session = GameSession(self.config, game_factory=launch)
        self.addCleanup(session.close)

        def reset():
            try:
                session.reset()
            except Exception as exc:
                errors.append(exc)

        thread = threading.Thread(target=reset)
        with patch.object(game, "close", wraps=game.close) as close:
            thread.start()
            try:
                self.assertTrue(entered.wait(1))
                session.close()
                self.assertFalse(game.closed)  # launch still owns it
            finally:
                finish.set()
                thread.join(2)
            self.assertFalse(thread.is_alive())
            self.assertTrue(game.closed)
            close.assert_called_once()
            self.assertRegex(str(errors[0]), "closed during launch")
            self.assertNotIn("create_game", [r["method"] for r in game.server.log])

    def test_concurrent_close_detaches_process_once(self):
        self.session.reset()
        game = self.session.game
        entered, finish = threading.Event(), threading.Event()

        def dispose():
            entered.set()
            if not finish.wait(2):
                raise AssertionError("dispose was not released")

        with patch.object(game, "close", side_effect=dispose) as close:
            thread = threading.Thread(target=self.session.close)
            thread.start()
            try:
                self.assertTrue(entered.wait(1))
                self.session.close()
                close.assert_called_once()
            finally:
                finish.set()
                thread.join(2)
        game.close()
        self.assertFalse(thread.is_alive())


class ProcessRecyclingTest(unittest.TestCase):
    def test_recycling_closes_old_process_before_launch_and_clears_groups(self):
        session = GameSession(GameConfig(map="m", max_episodes_per_process=2), game_factory=FakeGame)
        self.addCleanup(session.close)
        session.reset()
        first = session.game
        session.reset()
        self.assertIs(session.game, first)
        session.groups(0).assign("workers", [1001])
        session.reset()
        self.assertTrue(first.closed)
        self.assertIsNot(session.game, first)
        self.assertEqual(session.steps, 0)
        with self.assertRaises(KeyError):
            session.groups(0).members("workers")

    def test_replay_reset_always_replaces_the_process(self):
        with GameSession(
            GameConfig(map="recording.w3g", max_episodes_per_process=None), game_factory=FakeGame
        ) as session:
            session.reset()
            first = session.game
            session.reset()
            self.assertTrue(first.closed)
            self.assertIsNot(session.game, first)

    def test_tuning_survives_a_failed_replacement(self):
        with GameSession(GameConfig(map="m", max_episodes_per_process=1), game_factory=FakeGame) as session:
            session.reset()
            rpc = session.game.rpc
            # The fake does not implement clock tuning; emulate a successful reply.
            with patch.object(rpc, "_reply", return_value={}):
                rpc.debug("speed", factor=64)
            self.assertEqual(rpc.tuning, {"speed": {"factor": 64}})
            with patch("wc3env.rpc.RpcClient.create_game", side_effect=ValueError("unsupported")):
                with self.assertRaises(ValueError):
                    session.reset()
            with patch("wc3env.rpc.RpcClient.debug", return_value={}) as debug:
                session.reset()
                debug.assert_called_once_with("speed", factor=64)

    def test_generated_match_seed_survives_process_recycling(self):
        launched = []

        def launch(config):
            launched.append(config)
            return FakeGame(config)

        with patch("wc3env.session.secrets.randbelow", return_value=123) as random:
            with GameSession(
                GameConfig(map="m", setup=MatchSetup(), max_episodes_per_process=1), game_factory=launch
            ) as s:
                for _ in range(3):
                    s.reset()
            random.assert_called_once()
        self.assertEqual([c.setup.seed for c in launched], [123, 123, 123])


if __name__ == "__main__":
    unittest.main()
