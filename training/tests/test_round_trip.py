"""Each action type survives the round trip: translate it, play it, capture the orders, convert them back.

    python -m unittest training.tests.test_round_trip

The converter must give back the action the translator was given (same type, argument, target or point, and a
selection that picks the same units), which is what makes replay labels mean what the policy's actions do.
"""

import os
import unittest
from dataclasses import replace

from training.policy.actions import Action, fourcc_id, select, translate
from training.replays.convert import convert
from wc3env.binary import LUMBER, OWN, fourcc
from wc3env.session import GameConfig, GameSession, PlayerConfig

CONFIG = GameConfig(
    map="(2)EchoIsles.w3x",
    players=(PlayerConfig(0, "human"), PlayerConfig(1, "undead", control="computer")),
    render=False,
    sound=False,
    step_ms=250,
    observation="binary",
)


@unittest.skipUnless(os.environ.get("WC3_E2E"), "plays a game: set WC3_E2E=1")
class RoundTripTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.session = GameSession(CONFIG)
        obs = cls.session.reset()[0]
        c = cls.rpc = cls.session.game.rpc
        c.debug("ai", player=1, paused=1)
        c.debug("resources", player=0, gold=20000, lumber=20000)
        hall = next(u for u in obs.units if u["flags"] & OWN and fourcc(u["type_id"]) == "htow")
        cls.home = float(hall["x"]), float(hall["y"])

        def spawn(kind, dx, dy, n=1, player=0):
            x, y = cls.home[0] + dx, cls.home[1] + dy
            return c.debug("spawn", type_id=kind, player=player, x=x, y=y, n=n)["unit_ids"]

        spawn("hhou", 900, 900, n=6)  # food for the staged army
        cls.footmen = spawn("hfoo", -600, -600, n=8)
        cls.paladin = spawn("Hpal", -300, -700)[0]
        cls.archmage = spawn("Hamg", -200, -700)[0]
        cls.barracks = spawn("hbar", 700, -500)[0]
        cls.altar = spawn("halt", 700, 300)[0]
        cls.shop = spawn("ngme", -500, 200, player=15)[0]
        for hero in (cls.paladin, cls.archmage):
            c.debug("level", unit_id=hero, level=10)
        c.debug("give", unit_id=cls.paladin, type_id="phea")
        cls.hall = int(hall["unit_id"])
        cls.spawn = staticmethod(spawn)
        cls.obs = cls.step([])

    @classmethod
    def tearDownClass(cls):
        cls.session.close()

    @classmethod
    def step(cls, commands):
        obs, _, info = cls.session.step({0: commands})
        cls.placements = info.get("placements", {}).get(0, [])
        return obs[0]

    def round_trip(self, action: Action, settle: int = 0) -> None:
        for _ in range(settle):
            type(self).obs = self.step([])
        before = self.obs
        commands = translate(action, before)
        self.assertTrue(commands, f"{action.type}: nothing to command")
        after, got, seen = self.step(commands), [], []
        if action.type == "build" and action.x is not None:  # the site the placement search picked near the point
            site = self.placements[0]
            action = replace(action, x=float(site["x"]), y=float(site["y"]))
        for _ in range(3):  # commands go out with the next network turn: this step or the one after
            converted = convert(before, after)
            seen += converted
            got = [c for c in converted if c.action is not None and c.action.type == action.type]
            if got:
                break
            before, after = after, self.step([])
        type(self).obs = after
        self.assertEqual(len(got), 1, f"{action.type}: {seen}")
        a, reason = got[0].action, got[0].reason
        self.assertEqual(reason, "", action.type)
        self.assertEqual((a.argument, a.target), (action.argument, action.target), action.type)
        if action.x is not None:
            self.assertLess(abs(a.x - action.x) + abs(a.y - action.y), 1, action.type)
        if action.actor:
            self.assertEqual(set(select(a, before)), set(select(action, before)), action.type)

    def point(self, dx, dy):
        return self.home[0] + dx, self.home[1] + dy

    def nearest(self, rows, x, y):
        return min(rows, key=lambda r: (float(r["x"]) - x) ** 2 + (float(r["y"]) - y) ** 2)

    def tree_near(self, x, y) -> int:
        d = self.obs.destructables
        return int(self.nearest(d[(d["flags"] & LUMBER) != 0], x, y)["id"])

    def mines(self):
        u = self.obs.units
        return sorted(
            u[u["type_id"] == fourcc_id("ngol")],
            key=lambda m: (float(m["x"]) - self.home[0]) ** 2 + (float(m["y"]) - self.home[1]) ** 2,
        )

    def settle(self, n):
        for _ in range(n):
            type(self).obs = self.step([])

    def test_undead(self):
        ghoul, acolyte = self.spawn("ugho", 300, -300)[0], self.spawn("uaco", 350, -300)[0]
        ziggurat = self.spawn("uzig", 900, -900)[0]
        self.settle(1)
        self.round_trip(Action("smart", actor=ghoul, size=1, target=self.tree_near(*self.point(300, -300))))
        mine = int(self.mines()[0]["unit_id"])
        self.round_trip(Action("build", argument=fourcc_id("ugol"), actor=acolyte, target=mine))
        unsummon = 852210
        self.round_trip(
            Action("cast_unit", argument=unsummon, actor=self.spawn("uaco", 850, -850)[0], target=ziggurat), settle=1
        )

    def test_night_elf(self):
        wisp, ancient = self.spawn("ewsp", 300, 300)[0], self.spawn("eaom", -900, 900)[0]
        far_mine = self.mines()[0]
        mine_x, mine_y = float(far_mine["x"]) - self.home[0] + 2500, float(far_mine["y"]) - self.home[1]
        far_mine = {
            "unit_id": self.spawn("ngol", mine_x, mine_y, player=15)[0],
            "x": self.home[0] + mine_x,
            "y": self.home[1] + mine_y,
        }
        tree_of_life = self.spawn(
            "etol", float(far_mine["x"]) - self.home[0] + 450, float(far_mine["y"]) - self.home[1]
        )[0]
        self.settle(1)
        self.round_trip(Action("smart", actor=wisp, size=1, target=self.tree_near(*self.point(300, 300))))
        entangle, unroot, eat_tree, root = 852147, 852166, 852146, 852165
        self.round_trip(Action("cast_unit", argument=entangle, actor=tree_of_life, target=int(far_mine["unit_id"])))
        self.round_trip(Action("cast", argument=unroot, actor=ancient))
        self.settle(12)  # uprooting takes a few seconds
        self.round_trip(
            Action("cast_unit", argument=eat_tree, actor=ancient, target=self.tree_near(*self.point(-900, 900)))
        )
        x, y = self.point(-1200, 600)
        self.round_trip(Action("cast_point", argument=root, actor=ancient, x=x, y=y), settle=8)

    def test_orc(self):
        peon, burrow = self.spawn("opeo", 600, 600)[0], self.spawn("otrb", 750, 600)[0]
        self.settle(1)
        self.round_trip(Action("smart", actor=peon, size=1, target=burrow))
        battle_stations, stand_down = 852099, 852113
        self.round_trip(Action("cast", argument=battle_stations, actor=burrow), settle=4)
        self.round_trip(Action("cast", argument=stand_down, actor=burrow), settle=4)

    def test_item_pickup_and_aimed_use(self):
        x, y = self.point(-150, -700)
        item = self.rpc.debug("item", type_id="wlsd", x=x, y=y)["item_id"]
        self.settle(1)
        self.round_trip(Action("smart", actor=self.archmage, size=1, target=item))
        self.settle(8)
        self.round_trip(Action("use_item", argument=fourcc_id("wlsd"), actor=self.archmage, target=self.footmen[2]))

    def test_unit_orders(self):
        x, y = self.point(-900, -300)
        footman = self.footmen[0]
        self.round_trip(Action("move", actor=footman, same_type=True, size=4, x=x, y=y))
        self.round_trip(Action("attack_move", actor=footman, same_type=True, size=8, x=x + 200, y=y))
        self.round_trip(Action("stop", actor=footman, same_type=True, size=2))
        self.round_trip(Action("hold", actor=footman, same_type=True, size=1))
        self.round_trip(Action("attack", actor=footman, same_type=True, size=2, target=self.footmen[-1]))
        self.round_trip(Action("smart", actor=self.paladin, size=1, target=self.footmen[-1]))

    def test_production(self):
        self.round_trip(Action("train", argument=fourcc_id("hpea")))
        self.round_trip(Action("research", argument=fourcc_id("Rhde")))
        self.round_trip(Action("cancel", actor=self.barracks, size=1))
        x, y = self.point(-200, 900)
        self.round_trip(Action("rally", actor=self.hall, size=1, x=x, y=y))
        peasant = next(int(u["unit_id"]) for u in self.obs.units if u["flags"] & OWN and fourcc(u["type_id"]) == "hpea")
        self.round_trip(Action("build", argument=fourcc_id("hhou"), actor=peasant, x=x, y=y))

    def test_heroes(self):
        self.round_trip(Action("learn", argument=fourcc_id("AHhb"), actor=self.paladin))
        self.round_trip(Action("learn", argument=fourcc_id("AHds"), actor=self.paladin))
        self.round_trip(Action("learn", argument=fourcc_id("AHbz"), actor=self.archmage))
        self.rpc.debug("hp", unit_id=self.footmen[1], value=100)
        self.rpc.debug("mana", unit_id=self.paladin, value=1000)
        self.rpc.debug("mana", unit_id=self.archmage, value=1000)
        holy_light, divine_shield, blizzard = 852092, 852090, 852089
        self.round_trip(Action("cast_unit", argument=holy_light, actor=self.paladin, target=self.footmen[1]), settle=2)
        self.round_trip(Action("cast", argument=divine_shield, actor=self.paladin))
        x, y = self.point(-700, -900)
        self.round_trip(Action("cast_point", argument=blizzard, actor=self.archmage, x=x, y=y))
        self.rpc.debug("hp", unit_id=self.paladin, value=200)
        self.round_trip(Action("use_item", argument=fourcc_id("phea"), actor=self.paladin), settle=1)

    def test_items(self):
        for item in ("rde1", "ratc", "rst1"):
            self.rpc.debug("give", unit_id=self.paladin, type_id=item)
        type(self).obs = self.step([])
        x, y = self.point(-400, -500)
        self.round_trip(Action("drop_item", argument=fourcc_id("rde1"), actor=self.paladin, x=x, y=y))
        self.round_trip(Action("drop_item", argument=fourcc_id("ratc"), actor=self.paladin, target=self.archmage))
        shop = self.obs.units[self.obs.units["unit_id"] == self.shop][0]
        self.round_trip(Action("move", actor=self.paladin, size=1, x=float(shop["x"]) + 150, y=float(shop["y"]) - 150))
        self.round_trip(Action("sell", argument=fourcc_id("rst1"), actor=self.paladin, target=self.shop))

    def test_shop(self):
        shop = self.obs.units[self.obs.units["unit_id"] == self.shop][0]
        x, y = float(shop["x"]) + 150, float(shop["y"]) - 150
        self.round_trip(Action("move", actor=self.paladin, size=1, x=x, y=y))
        dust = Action("buy", argument=fourcc_id("dust"), actor=self.paladin, target=self.shop)
        for _ in range(100):  # neutral stock appears after a delay: repeat the order until the purchase lands
            commands = translate(dust, self.obs)
            for k in range(4):
                before = self.obs
                type(self).obs = self.step(commands if k == 0 else [])
                if dust in [c.action for c in convert(before, self.obs)]:
                    return
        self.fail("the purchase never landed")

    def test_revive(self):
        self.rpc.debug("kill", unit_id=self.archmage)
        self.round_trip(Action("revive", argument=fourcc_id("Hamg")), settle=100)  # a death resolves in ~20 s


if __name__ == "__main__":
    unittest.main()
