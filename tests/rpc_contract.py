"""The RPC conformance suite (docs/specs/protocol.md): every method in every lifecycle
status, malformed input, exact error codes, exactness of `step`, the closed set of `act`
rejection reasons. The DLL runs it in tests/e2e/test_rpc_contract_dll.py.

Subclass with `make_client()` returning a fresh RpcClient against a server in `launched`
status, and `players()` returning a create_game player list the server accepts.
"""

from __future__ import annotations

from wc3env.protocol import PROTOCOL_VERSION, SCORE_FIELDS
from wc3env.rpc import ERROR_CODES, REJECT_REASONS, STATUSES, RpcClient, RpcError

MAP = "(2)EchoIsles.w3x"


class RpcContract:
    """Mix into a unittest.TestCase."""

    def test_auto_place_is_a_boolean_build_argument(self):
        c = self.start()
        unit = next(u for u in c.observe(0)["units"] if not u["structure"])
        build = {
            "unit_id": unit["unit_id"],
            "command": "build",
            "arguments": {"type_id": "hhou", "x": unit["x"], "y": unit["y"]},
        }
        for flag in (1, 0, "true", None, [], {}):
            with self.subTest(flag=flag):
                bad = {**build, "arguments": {**build["arguments"], "auto_place": flag}}
                self.assertEqual(c.act(0, [bad])["rejected"], [{"index": 0, "reason": "bad_arguments"}])
        for flag in (True, False):
            bad = {**build, "command": "move", "arguments": {"x": 0, "y": 0, "auto_place": flag}}
            self.assertEqual(c.act(0, [bad])["rejected"], [{"index": 0, "reason": "bad_arguments"}])

    def test_queued_orders_allow_explicit_repeated_units_and_reject_invalid_flags(self):
        c = self.start()
        unit = next(u for u in c.observe(0)["units"] if not u["structure"])
        move = {"unit_id": unit["unit_id"], "command": "move", "arguments": {"x": unit["x"] + 100, "y": unit["y"]}}
        queued = {**move, "arguments": {**move["arguments"], "queued": True, "y": unit["y"] + 100}}
        self.assertEqual(c.act(0, [move, queued])["rejected"], [])
        for flag in (1, 0, "true", None, [], {}):
            with self.subTest(flag=flag):
                bad = {**move, "arguments": {**move["arguments"], "queued": flag}}
                self.assertEqual(c.act(0, [bad])["rejected"], [{"index": 0, "reason": "bad_arguments"}])
        bad = {"unit_id": unit["unit_id"], "command": "select", "arguments": {"queued": True}}
        self.assertEqual(c.act(0, [bad])["rejected"], [{"index": 0, "reason": "bad_arguments"}])

    def make_client(self) -> RpcClient:
        raise NotImplementedError

    def players(self) -> list[dict]:
        return [{"slot": 0, "race": "human", "control": "agent"}, {"slot": 1, "control": "computer"}]

    def end_game(self, c: RpcClient) -> None:
        """Bring the running game to a result. The fake has a debug op; the DLL razes the enemy."""
        c.debug("end")

    def start(self, mode="stepping") -> RpcClient:
        c = self.make_client()
        c.create_game(MAP, self.players(), mode)
        return c

    def assertFault(self, code, fn, *a, **kw):
        with self.assertRaises(RpcError) as ctx:
            fn(*a, **kw)
        self.assertEqual(ctx.exception.code, code, ctx.exception.detail)
        self.assertIn(ctx.exception.code, ERROR_CODES)
        return ctx.exception

    # ---- envelope ---------------------------------------------------------------------------
    def test_every_reply_carries_version_id_status(self):
        c = self.make_client()
        r = c.call_raw({"protocol_version": PROTOCOL_VERSION, "id": 42, "method": "info"})
        self.assertEqual((r["protocol_version"], r["id"], r["ok"]), (PROTOCOL_VERSION, 42, True))
        self.assertIn(r["status"], STATUSES)
        self.assertEqual(r["status"], "launched")

    def test_bad_version_bad_request_unknown_method(self):
        c = self.make_client()
        r = c.call_raw({"protocol_version": 2, "id": 1, "method": "info"})
        self.assertEqual((r["ok"], r["error"]), (False, "bad_version"))
        r = c.call_raw("this is not json")
        self.assertEqual((r["ok"], r["error"]), (False, "bad_request"))
        r = c.call_raw({"protocol_version": 1, "id": "seven", "method": "info"})
        self.assertEqual((r["ok"], r["error"]), (False, "bad_request"))
        r = c.call_raw({"protocol_version": 1, "id": 3, "params": {}})
        self.assertEqual((r["ok"], r["error"]), (False, "bad_request"))
        self.assertFault("unknown_method", c.call, "teleport")
        r = c.call_raw({"protocol_version": 1, "id": 4, "method": "info", "params": []})
        self.assertEqual((r["ok"], r["error"]), (False, "bad_params"))

    def test_info_in_every_status(self):
        c = self.make_client()
        i = c.info()
        self.assertEqual(c.status, "launched")
        self.assertIsNone(i["mode"])
        for k in ("exe_hash", "dll_version", "game_time_ms", "instance"):
            self.assertIn(k, i)
        c.create_game(MAP, self.players(), "stepping")
        self.assertEqual(c.info()["mode"], "stepping")
        self.assertEqual(c.status, "in_game")

    # ---- lifecycle --------------------------------------------------------------------------
    def test_methods_outside_their_status_are_bad_status(self):
        c = self.make_client()
        self.assertFault("bad_status", c.step, 25)
        self.assertFault("bad_status", c.observe, 0)
        self.assertFault("bad_status", c.act, 0, [])
        self.assertFault("bad_status", c.debug, "end")
        self.assertFault("bad_status", c.reset)
        self.assertFault("bad_status", c.save_replay, "unused.w3g")
        c.create_game(MAP, self.players(), "stepping")
        self.assertFault("bad_status", c.create_game, MAP, self.players(), "stepping")

    def test_reset_discards_a_running_episode_and_holds_for_create(self):
        c = self.start()
        initial = c.observe(0)
        c.step(250)
        c.reset()
        self.assertEqual(c.status, "launched")
        self.assertIsNone(c.info()["mode"])
        self.assertFault("bad_status", c.observe, 0)
        c.create_game(MAP, self.players())
        fresh = c.observe(0)
        self.assertEqual(fresh["sequence"], 0)
        self.assertEqual(fresh["game_time_seconds"], initial["game_time_seconds"])
        self.assertEqual(fresh["result"], "")

    def test_create_game_params(self):
        c = self.make_client()
        self.assertFault("bad_params", c.create_game, 7, self.players(), "stepping")
        self.assertFault("bad_params", c.create_game, MAP, "everyone", "stepping")
        self.assertFault("bad_params", c.create_game, MAP, [{"slot": "a"}], "stepping")
        self.assertFault("bad_params", c.create_game, MAP, self.players(), "sideways")
        self.assertFault("bad_params", c.create_game, "", self.players(), "stepping")
        for bad in (
            [],
            [self.players()[0]] * 2,
            [7],
            [{"slot": 0, "race": "elf", "control": "agent"}],
            [{"slot": 0, "race": None, "control": "agent"}],
            [{"slot": 0, "control": "human"}],
        ):
            self.assertFault("bad_params", c.create_game, MAP, bad, "stepping")
        for slot in (-1, 16, True):
            self.assertFault("bad_params", c.create_game, MAP, [{"slot": slot, "control": "agent"}], "stepping")
        self.assertEqual(c.status, "launched")
        r = c.create_game(MAP, self.players(), "stepping")
        self.assertIn("game_time_ms", r)
        self.assertTrue(r["map"].endswith(MAP))
        self.assertEqual(r["players"][0], self.players()[0])
        self.assertIn(r["players"][1]["race"], ("human", "orc", "undead", "night_elf"))
        self.assertEqual(c.status, "in_game")

    def test_observation_metadata(self):
        c = self.start()
        obs = c.observe(0)
        own = next(p for p in obs["players"] if p["id"] == 0)
        self.assertEqual((own["kind"], own["relation"], own["controller"]), ("player", "self", "agent"))
        self.assertEqual(set(obs["score"]), set(SCORE_FIELDS))
        bounds = obs["map"]["bounds"]
        self.assertLess(bounds["min_x"], bounds["max_x"])
        self.assertLess(bounds["min_y"], bounds["max_y"])
        following = c.observe(0)
        self.assertEqual(following["sequence"], obs["sequence"] + 1)
        self.assertEqual(following["game_time_seconds"], obs["game_time_seconds"])

    def test_release_numeric_limits(self):
        c = self.make_client()
        for version in (True, 1.0):
            r = c.call_raw({"protocol_version": version, "id": 1, "method": "info"})
            self.assertEqual((r["ok"], r["error"]), (False, "bad_version"))
        for value in ("NaN", "Infinity", "1e999", "01", "1e"):
            r = c.call_raw('{"protocol_version":1,"id":1,"method":"info","params":{"ignored":' + value + "}}")
            self.assertEqual((r["ok"], r["error"]), (False, "bad_request"))
        c.create_game(MAP, self.players(), "stepping")
        for ms in (60025, 2**32, True):
            self.assertFault("bad_params", c.step, ms)
        self.assertFault("bad_params", c.act, 0, [{"unit_id": True, "command": "stop"}])

    def test_observed_abilities_are_private(self):
        obs = self.start().observe(0)
        for unit in obs["units"]:
            self.assertIsInstance(unit["abilities"], list)
        for unit in obs["visible_enemies"]:
            self.assertNotIn("abilities", unit)

    def test_json_string_escapes(self):
        c = self.make_client()
        reply = c.call_raw(r'{"protocol_version":1,"id":9,"method":"\u0069nfo"}')
        self.assertTrue(reply["ok"], reply)
        for method in ("snow\u2603", "smile\U0001f642", 'quote" slash\\ tab\t'):
            error = self.assertFault("unknown_method", c.call, method)
            self.assertEqual(error.detail, method)

    def test_ended_allows_observe_and_create_game_but_not_step_or_act(self):
        c = self.start()
        self.end_game(c)
        self.assertEqual(c.status, "ended")
        final = c.observe(0)
        self.assertTrue(final["result"])
        self.assertFault("bad_status", c.step, 25)
        self.assertFault("bad_status", c.act, 0, [])
        c.create_game(MAP, self.players(), "stepping")
        self.assertEqual(c.status, "in_game")
        self.assertEqual(c.observe(0)["result"], "")

    def test_reset_from_realtime_starts_a_stepping_episode(self):
        c = self.start("realtime")
        c.observe(0)
        c.reset()
        c.create_game(MAP, self.players(), "stepping")
        self.assertEqual(c.observe(0)["sequence"], 0)
        before = c.info()["game_time_ms"]
        self.assertEqual(c.step(250)["game_time_ms"], before + 250)

    def test_quit_is_valid_anywhere(self):
        c = self.make_client()
        c.quit()
        c = self.start()
        c.quit()

    # ---- step -------------------------------------------------------------------------------
    def test_step_is_exact_and_frames_reported(self):
        c = self.start()
        t0 = c.info()["game_time_ms"]
        for ms in (25, 250, 1000, 25):
            r = c.step(ms)
            self.assertEqual(r["game_time_ms"], t0 + ms, f"step {ms}")
            self.assertGreaterEqual(r["frames"], 1)
            t0 = r["game_time_ms"]
        self.assertAlmostEqual(c.observe(0)["game_time_seconds"], t0 / 1000)

    def test_step_params(self):
        c = self.start()
        for bad in (0, -25, 30, 12.5, "250", None):
            e = self.assertFault("bad_params", c.step, bad)
            self.assertIn("25", e.detail)

    def test_step_in_realtime_is_not_stepping(self):
        c = self.start("realtime")
        self.assertFault("not_stepping", c.step, 25)

    # ---- observe ----------------------------------------------------------------------------
    def test_observe_shape_and_sequence(self):
        c = self.start()
        o1, o2 = c.observe(0), c.observe(0)
        for o in (o1, o2):
            self.assertEqual(o["protocol_version"], PROTOCOL_VERSION)
            for k in (
                "sequence",
                "game_time_seconds",
                "player",
                "units",
                "inside",
                "visible_enemies",
                "items",
                "inventory",
                "events",
                "result",
            ):
                self.assertIn(k, o)
            for k in ("gold", "lumber", "food_used", "food_cap"):
                self.assertIsInstance(o["player"][k], int)
            for u in o["units"] + o["visible_enemies"]:
                for k in (
                    "unit_id",
                    "type_id",
                    "owner",
                    "x",
                    "y",
                    "hp",
                    "max_hp",
                    "mana",
                    "max_mana",
                    "structure",
                    "hero",
                    "level",
                ):
                    self.assertIn(k, u)
                self.assertEqual(len(u["type_id"]), 4)
            self.assertTrue(all(u["owner"] == 0 for u in o["units"]))
            self.assertTrue(all(u["owner"] != 0 for u in o["visible_enemies"]))
            for u in o["units"]:  # own units also carry their order, own structures their production
                self.assertIn("order", u)
                for k in ("state", "state_seconds", "queue", "queue_seconds"):
                    self.assertEqual(k in u, u["structure"])
            self.assertTrue(all("order" not in u and "queue" not in u for u in o["visible_enemies"]))
        self.assertEqual(o2["sequence"], o1["sequence"] + 1)
        self.assertFault("bad_params", c.observe, 99)
        self.assertFault("bad_params", c.observe, "me")

    def test_observation_is_a_function_of_game_time(self):
        c = self.start()
        a = c.observe(0)
        c.step(0 + 25 * 4)
        b = c.observe(0)
        self.assertEqual(b["game_time_seconds"], a["game_time_seconds"] + 0.1)

    def test_observation_sequences_are_per_player(self):
        c = self.start()
        self.assertEqual(c.observe(1)["sequence"], 0)
        self.assertEqual(c.observe(0)["sequence"], 0)
        self.assertEqual(c.observe(0)["sequence"], 1)
        self.assertEqual(c.observe(1)["sequence"], 1)

    # ---- act --------------------------------------------------------------------------------
    def test_act_params_and_rejection_reasons(self):
        c = self.start()
        o = c.observe(0)
        mine = o["units"][0]["unit_id"]
        theirs = o["visible_enemies"][0]["unit_id"] if o["visible_enemies"] else 999999
        self.assertFault("bad_params", c.act, "me", [])
        self.assertFault("bad_params", c.act, 0, "move")
        self.assertFault("bad_params", c.act, 0, [{"command": "move"}])
        self.assertFault("bad_params", c.act, 0, [{"unit_id": mine, "command": 5}])
        r = c.act(
            0,
            [
                {"unit_id": mine, "command": "stop", "arguments": {}},
                {"unit_id": mine, "command": "stop", "arguments": {}},
                {"unit_id": theirs, "command": "stop", "arguments": {}},
                {"unit_id": 123456789, "command": "stop", "arguments": {}},
                {"unit_id": mine, "command": "fly", "arguments": {}},
            ],
        )
        by_index = {j["index"]: j["reason"] for j in r["rejected"]}
        self.assertNotIn(0, by_index)
        self.assertNotIn(1, by_index)
        self.assertEqual(by_index[2], "not_your_unit")
        self.assertEqual(by_index[3], "unknown_unit")
        self.assertEqual(by_index[4], "unknown_command")
        for j in r["rejected"]:
            self.assertIn(j["reason"], REJECT_REASONS)

    def test_large_batches_and_repeated_units_keep_command_order(self):
        c = self.start()
        obs = c.observe(0)
        u = next(u for u in obs["units"] if not u["structure"])
        stop = {"unit_id": u["unit_id"], "command": "stop"}
        self.assertEqual(c.act(0, [stop] * 200)["rejected"], [])
        moves = [
            {"unit_id": u["unit_id"], "command": "move", "arguments": {"x": u["x"] + dx, "y": u["y"]}}
            for dx in (-300, 300)
        ]
        self.assertEqual(c.act(0, moves)["rejected"], [])
        self.assertEqual(c.observe(0)["game_time_seconds"], obs["game_time_seconds"])
        c.step(2000)
        moved = next(w for w in c.observe(0)["units"] if w["unit_id"] == u["unit_id"])
        self.assertGreater(moved["x"], u["x"] + 100)

    def test_a_move_order_moves_the_unit(self):
        c = self.start()
        o = c.observe(0)
        u = next(u for u in o["units"] if not u["structure"])
        r = c.act(0, [{"unit_id": u["unit_id"], "command": "move", "arguments": {"x": u["x"] + 300, "y": u["y"]}}])
        self.assertEqual(r["rejected"], [])
        c.step(2000)
        v = next(w for w in c.observe(0)["units"] if w["unit_id"] == u["unit_id"])
        self.assertGreater(v["x"], u["x"] + 100)

    def test_bad_arguments_is_a_rejection_not_a_fault(self):
        c = self.start()
        u = next(u for u in c.observe(0)["units"] if not u["structure"])
        r = c.act(0, [{"unit_id": u["unit_id"], "command": "move", "arguments": {"x": "far"}}])
        self.assertEqual(r["rejected"], [{"index": 0, "reason": "bad_arguments"}])

    # ---- determinism ------------------------------------------------------------------------
    def test_same_inputs_same_observations(self):
        def run():
            c = self.start()
            u = next(u for u in c.observe(0)["units"] if not u["structure"])
            out = []
            for i in range(4):
                c.act(
                    0,
                    [
                        {
                            "unit_id": u["unit_id"],
                            "command": "move",
                            "arguments": {"x": u["x"] + 200 * (i + 1), "y": u["y"]},
                        }
                    ],
                )
                c.step(250)
                o = c.observe(0)
                out.append(
                    [(w["unit_id"], w["type_id"], round(w["x"], 1), round(w["y"], 1), w["hp"]) for w in o["units"]]
                    + [o["player"]]
                )
            return out

        self.assertEqual(run(), run())
