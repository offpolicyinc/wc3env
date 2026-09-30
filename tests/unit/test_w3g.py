"""The .w3g reader against a 1.29 ladder replay (tests/fixtures/ladder_hvh.w3g) and our own recording."""

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

    def test_our_own_recording(self):
        replay = w3g.read(FIXTURES / "self_play.w3g")
        self.assertEqual([p.slot for p in replay.opponents], [0, 1])
        self.assertEqual(replay.map_checksum, 0xCCD6B5CA)  # the same stock map and patch as the ladder game
        self.assertTrue(list(replay.commands()))

    def test_not_a_replay(self):
        with self.assertRaisesRegex(ValueError, "not a Warcraft III replay"):
            w3g.read(Path(__file__))


if __name__ == "__main__":
    unittest.main()
