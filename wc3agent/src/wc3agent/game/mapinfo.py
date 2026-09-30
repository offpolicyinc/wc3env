"""What the map places before the game starts, from `tools.prepare map`: starts, mines, camps, shops."""

import json
from math import hypot
from pathlib import Path


class MapInfo:
    def __init__(self, data):
        if data.get("schema_version") not in (1, 2):  # 2 adds bounds, pathing and trees
            raise ValueError("Unsupported map description; regenerate with tools.prepare map")
        self.name = data["map"]
        self.starts = data["start_locations"]
        self.mines = data["gold_mines"]
        self.buildings = data["neutral_buildings"]
        # Camps keep the file's order (weakest first), so "camp 3" means the same thing all game.
        self.camps = [{"number": i + 1, **camp} for i, camp in enumerate(data["creep_camps"])]

    @classmethod
    def load(cls, path):
        try:
            return cls(json.loads(Path(path).read_text(encoding="utf-8")))
        except FileNotFoundError:
            raise FileNotFoundError(f"Map description missing: {path}. Run python -m tools.prepare map") from None

    @staticmethod
    def distance(a, b):
        return hypot(a["x"] - b["x"], a["y"] - b["y"])

    def nearest(self, point, places):
        return min(places, key=lambda place: self.distance(point, place), default=None)

    def home(self, point):
        """The start location a player's first hall stands on."""
        return self.nearest(point, self.starts)

    def enemy_starts(self, home):
        return [s for s in self.starts if s is not home]

    def camp(self, number):
        return next((c for c in self.camps if c["number"] == number), None)
