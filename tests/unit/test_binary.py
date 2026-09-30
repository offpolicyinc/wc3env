"""Binary observations parse as the DLL lays them out (wc3hook/obsbin.h)."""

import unittest

import numpy as np

from wc3env.binary import DEAD, DTYPES, HEADER, HERO, INSIDE, MAGIC, OWN, POINT, TABLES, VERSION, fourcc, parse
from wc3env.protocol import own_unit_ids


def encode(**tables):
    """An observation as obs_write_binary writes it: the header, then each table's records."""
    offset, body, entries = HEADER.size, b"", []
    for name in TABLES:
        records = np.array(tables.get(name, []), DTYPES[name])
        entries += [offset, len(records), DTYPES[name].itemsize]
        offset += records.nbytes
        body += records.tobytes()
    score = [0] * 24 + [7]
    # magic, version, size, player, sequence, time; gold, lumber, food used, food cap; result, events lost,
    # orders lost; time of day
    head = [MAGIC, VERSION, offset, 1, 0, 61250, 500, 0, 0, 0, 2, 0, 3, 8.5, *score, len(TABLES), *entries]
    return HEADER.pack(*head) + body


class BinaryObservationTest(unittest.TestCase):
    def test_record_sizes_match_the_dll(self):
        sizes = {name: dtype.itemsize for name, dtype in DTYPES.items()}
        self.assertEqual(HEADER.size, 292)
        self.assertEqual(
            sizes,
            {
                "units": 100,
                "abilities": 24,
                "buffs": 8,
                "queue": 12,
                "inventory": 16,
                "items": 16,
                "destructables": 24,
                "events": 20,
                "heroes": 24,
                "research": 8,
                "orders": 40,
            },
        )

    def test_tables_and_scalars_parse(self):
        peasant, paladin = int.from_bytes(b"hpea", "big"), int.from_bytes(b"Hpal", "big")
        combat = (0, 5, 6, 2.0, 190, 270, 0)  # armor, damage min/max, attack period, move speed, facing, resource
        units = [
            (15419, peasant, 1, 4404.2, 3010.1, 220, 220, 0, 0, OWN, 0, 0, 0xFFFFFFFF, 0, 0, 0, 0, 0, *combat),
            (15420, peasant, 1, 0, 0, 220, 220, 0, 0, OWN | INSIDE, 0, 0, 0xFFFFFFFF, 0, 0, 0, 0, 0, *combat),
            (15500, peasant, 0, 10, 20, 100, 220, 0, 0, 0, 0, 0, 0xFFFFFFFF, 0, 0, 0, 0, 0, *combat),
            (15600, paladin, 1, 0, 0, 0, 650, 0, 255, OWN | HERO | DEAD, 2, 0, 0xFFFFFFFF, 0, 0, 0, 0, 0, *combat),
        ]
        obs = parse(
            encode(
                units=units,
                events=[(20, 15500, 0, peasant, 0)],
                heroes=[(15600, 250, 1, 25, 15, 17)],
                research=[(int.from_bytes(b"Rhde", "big"), 1)],
                orders=[(15419, 851986, POINT, 0xFFFFFFFF, 0, 100.0, -200.0, 1, 0, 61000)],
            )
        )
        self.assertEqual((obs.player, obs.game_time_seconds, obs.gold, obs.result), (1, 61.25, 500, "defeat"))
        self.assertEqual(obs.time_of_day, 8.5)
        self.assertEqual((obs.orders_lost, obs.orders["y"].tolist(), obs.orders["origin"].tolist()), (3, [-200.0], [1]))
        self.assertEqual(obs.heroes["xp"].tolist(), [250])
        self.assertEqual(fourcc(obs.research["type_id"][0]), "Rhde")
        self.assertEqual(obs.units["damage_max"].tolist(), [6] * 4)
        self.assertEqual(obs.score["total"], 7)
        self.assertEqual(fourcc(obs.units["type_id"][0]), "hpea")
        self.assertAlmostEqual(float(obs.units["x"][0]), 4404.2, places=3)
        self.assertEqual(own_unit_ids(obs), frozenset({15419, 15420}))
        self.assertEqual(obs.events["kind"].tolist(), [20])
        self.assertEqual(len(obs.destructables), 0)

    def test_mismatched_layout_is_rejected(self):
        data = bytearray(encode())
        data[4] = 99  # version
        with self.assertRaisesRegex(ValueError, "does not match"):
            parse(bytes(data))
        with self.assertRaisesRegex(ValueError, "shorter"):
            parse(b"\0" * 8)


if __name__ == "__main__":
    unittest.main()
