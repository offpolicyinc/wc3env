"""Every order given to a unit reaches its owner's observation, tagged with who gave it."""

import unittest

from wc3env.binary import IMMEDIATE, ORIGINS, OWN, POINT, TARGET, SharedObservations, fourcc
from wc3env.session import GameConfig, GameSession, PlayerConfig

MOVE, SMART, STOP, RETURN_RESOURCES = 851986, 851971, 851972, 852017
CONFIG = GameConfig(
    map="(2)EchoIsles.w3x",
    players=(PlayerConfig(0, "human"), PlayerConfig(1, "undead", control="computer")),
    render=False,
    sound=False,
    step_ms=250,
    observation="binary",
)


def by_origin(obs, origin):
    """(unit, order, kind, target, x, y, queued) for each order of that origin in the observation."""
    fields = ("unit_id", "order_id", "kind", "target_id", "x", "y", "queued")
    return [tuple(round(o[f].item()) for f in fields) for o in obs.orders if ORIGINS[o["origin"]] == origin]


class OrdersTest(unittest.TestCase):
    def test_commands_come_back_as_player_orders_and_the_rest_is_tagged(self):
        with GameSession(CONFIG) as session:
            obs = session.reset()[0]
            own = [u for u in obs.units if u["flags"] & OWN]
            peasants = [int(u["unit_id"]) for u in own if fourcc(u["type_id"]) == "hpea"]
            hall = next(int(u["unit_id"]) for u in own if fourcc(u["type_id"]) == "htow")
            mine = min(
                (u for u in obs.units if fourcc(u["type_id"]) == "ngol"),
                key=lambda u: (u["x"] - own[0]["x"]) ** 2 + (u["y"] - own[0]["y"]) ** 2,
            )
            mine_id, mine_x, mine_y = int(mine["unit_id"]), round(float(mine["x"])), round(float(mine["y"]))
            x, y = round(float(own[0]["x"])) + 300, round(float(own[0]["y"]))
            obs = session.step(
                {
                    0: [
                        {"unit_id": peasants[0], "command": "move", "arguments": {"x": x, "y": y}},
                        {
                            "unit_id": peasants[0],
                            "command": "move",
                            "arguments": {"x": x, "y": y + 200, "queued": True},
                        },
                        {"unit_id": peasants[1], "command": "smart", "arguments": {"target_id": mine_id}},
                        {"unit_id": peasants[2], "command": "stop"},
                        {"unit_id": hall, "command": "train", "arguments": {"type_id": "hpea"}},
                    ]
                }
            )[0][0]
            none = 0xFFFFFFFF
            self.assertEqual(
                sorted(by_origin(obs, "player")),
                sorted(
                    [
                        (peasants[0], MOVE, POINT, none, x, y, 0),
                        (peasants[0], MOVE, POINT, none, x, y + 200, 1),
                        (peasants[1], SMART, TARGET, mine_id, mine_x, mine_y, 0),  # where the target is
                        (peasants[2], STOP, IMMEDIATE, none, 0, 0, 0),
                        (hall, int.from_bytes(b"hpea", "big"), IMMEDIATE, none, 0, 0, 0),
                    ]
                ),
            )
            self.assertEqual(obs.orders_lost, 0)

            engine = []
            for _ in range(160):  # 40 s: the smart peasant returns its first load on its own
                obs = session.step({0: []})[0][0]
                engine += by_origin(obs, "engine")
                self.assertEqual(by_origin(obs, "player"), [])
                self.assertEqual(by_origin(obs, "script"), [])  # no JASS gives the agent's units orders
            self.assertIn(RETURN_RESOURCES, [o[1] for o in engine if o[0] == peasants[1]])

            rpc = session.game.rpc
            located = rpc.observe_binary(1)["observations"][0]
            shared = SharedObservations(session.game.pid)
            self.addCleanup(shared.close)
            computer = shared.read(located["offset"], located["size"])
            script = by_origin(computer, "script")
            self.assertTrue(script, "the computer's AI script orders its units")
            self.assertEqual(by_origin(computer, "player"), [])


if __name__ == "__main__":
    unittest.main()
