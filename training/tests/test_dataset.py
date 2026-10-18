"""The data path end to end on a ladder replay: write its data set, load it back, train on it.

python -m unittest training.tests.test_dataset      (plays the replay: about a minute)
"""

import tempfile
import unittest
from pathlib import Path

import numpy as np

from training.replays.dataset import write
from training.train.loader import ReplayData, collate

FIXTURE = Path(__file__).resolve().parents[2] / "tests" / "fixtures" / "ladder_hvh.w3g"


class DatasetTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dir = tempfile.TemporaryDirectory()
        cls.row = write(str(FIXTURE), Path(cls.dir.name))
        cls.data = [ReplayData(p) for p in sorted(Path(cls.dir.name).glob("*.npz"))]

    @classmethod
    def tearDownClass(cls):
        cls.dir.cleanup()

    def test_written_for_both_players(self):
        self.assertNotIn("error", self.row)
        self.assertEqual(len(self.data), 2)
        for d in self.data:
            self.assertGreater(d.steps, 2000)  # a 9-minute game at 4 steps a second
            self.assertGreater(int((d.d["type"] != 0).sum()), 400)  # hundreds of actions each

    def test_loader_and_model_draw_the_same_map(self):
        """The loader's map (features.build_map, on the CPU) and the model's (model.draw_map, on the GPU when
        training and playing) from the same stored rows."""
        import torch

        from training.policy.model import draw_map

        for d in self.data:
            for t in range(0, d.steps, 97):
                cpu = d.step(t, map=True)["map"]
                s = {k: torch.from_numpy(np.asarray(v))[None] for k, v in d.step(t).items()}
                gpu = draw_map(s, torch.from_numpy(d.static_map)[None])[0].numpy()
                self.assertLess(float(np.abs(cpu - gpu).max()), 1e-6)

    def test_labels_fit_their_masks(self):
        for d in self.data:
            allowed = checked = 0
            for t in range(d.steps):
                s = d.step(t)
                self.assertTrue(s["type_mask"][s["type"]])
                for head, mask in (("argument", "argument_mask"), ("actor", "actor_mask"), ("target", "target_mask")):
                    if (head == "argument" and s[head] > 0) or (head != "argument" and s[head] >= 0):
                        checked += 1
                        allowed += bool(s[mask][s[head]])
            self.assertGreaterEqual(allowed / checked, 0.99)  # a carried-over label can meet a changed step

    def test_a_small_model_learns_a_batch(self):
        import torch

        from training.policy.model import Policy

        torch.manual_seed(0)
        batch = collate([self.data[0].chunk(200, 16), self.data[1].chunk(400, 16)])
        model = Policy(d=64, layers=1, heads=4, memory=1)
        model(batch)
        optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3)
        losses = []
        for _ in range(30):
            loss, _, _ = model.cloning_loss(batch, model(batch))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(float(loss))
        self.assertLess(losses[-1], losses[0] / 3)


if __name__ == "__main__":
    unittest.main()
