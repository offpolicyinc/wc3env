"""The .w3g reader against a 1.29 ladder replay (tests/fixtures/ladder_hvh.w3g) and our own recording."""

import struct
import unittest
from pathlib import Path

from wc3env import w3g

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"


class W3gTest(unittest.TestCase):
    def test_a_ladder_replay_recorded_by_an_observer(self):
        replay = w3g.read(FIXTURES / "ladder_hvh.w3g")
        self.assertEqual((replay.version, replay.build, replay.duration_ms), (29, 6060, 553440))
        self.assertEqual((replay.map, replay.map_checksum), ("Maps\\FrozenThrone\\(2)EchoIsles.w3x", 0xCCD6B5CA))
        self.assertEqual(
            [(p.slot, p.name, p.race) for p in replay.opponents],
            [(1, "Deathnot!", "human"), (2, "vankov.stas", "human")],
        )
        self.assertTrue(any(p.observer for p in replay.players))
        commands = list(replay.commands())
        self.assertEqual([t for t, _, _ in commands], sorted(t for t, _, _ in commands))
        self.assertLessEqual(commands[-1][0], replay.duration_ms)
        # the first command: player 3 (slot 1) selects its town hall, the object the game created for it
        t, pid, ids = next(replay.selections())
        self.assertEqual((t, pid, ids), (720, 3, [15387]))
        # then trains a peasant (flags 0x42, the order id is the type) and right-clicks the gold mine
        orders = [(t, pid, a, *struct.unpack_from("<HI", b)) for t, pid, a, b in replay.actions() if 0x10 <= a <= 0x14]
        self.assertEqual(orders[:2], [(720, 3, 0x10, 0x42, int.from_bytes(b"hpea", "big")), (720, 3, 0x12, 0, 851971)])
        self.assertEqual(len(orders), 1353)

    def test_our_own_recording(self):
        replay = w3g.read(FIXTURES / "self_play.w3g")
        self.assertEqual([p.slot for p in replay.opponents], [0, 1])
        self.assertEqual(replay.map_checksum, 0xCCD6B5CA)  # the same stock map and patch as the ladder game
        self.assertTrue(list(replay.commands()))

    def test_paused_time_is_not_game_time(self):
        def turn(dt, *actions):  # one turn: player 3's command block with these actions
            block = b"".join(actions)
            body = struct.pack("<H", dt) + struct.pack("<BH", 3, len(block)) + block
            return b"\x1f" + struct.pack("<H", len(body)) + body

        # 1 s of play, a pause at 1.1 s, 5 s of turns while paused, a resume, then a command at 6.3 s
        data = turn(1000) + turn(100, b"\x01") + turn(5000) + turn(100, b"\x02") + turn(100, b"\x10" + bytes(14))
        replay = w3g.Replay(29, 6060, 7000, "", 0, (), data, 0)
        self.assertEqual(replay.pauses, ((1100, 6200),))
        self.assertEqual([t for t, _, a, _ in replay.actions() if a == 0x10], [6300])
        self.assertEqual((replay.game_ms(1000), replay.game_ms(3000), replay.game_ms(6300)), (1000, 1100, 1200))

    def test_not_a_replay(self):
        with self.assertRaisesRegex(ValueError, "not a Warcraft III replay"):
            w3g.read(Path(__file__))


if __name__ == "__main__":
    unittest.main()
