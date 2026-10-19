"""What the benchmarks share: the recorded observations they replay, the checkpoint that decides, and the Modal
image they run in (this checkout's training/ and wc3env, the observations and the checkpoint)."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OBSERVATIONS = ROOT / "build" / "obs.pkl"  # python -m training.bench.record_obs
CHECKPOINT = ROOT / "runs" / "bc10" / "step8k.pt"
MAP = "Maps/FrozenThrone/(2)EchoIsles.w3x"
# the hook's feature code, which policy/native.py builds into a library
FEATURE_SOURCES = {"features.c", "features.h", "features_obs.h", "features_table.h", "masks.c", "obsbin.h"}


# FlashAttention-2's prebuilt wheel for this PyTorch (the learner's packed step transformer)
FLASH_ATTN_WHEEL = "https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.8cxx11abiTRUE-cp311-cp311-linux_x86_64.whl"


def image(*commands: str):
    """A GPU image with PyTorch and this checkout; the observations at /root/obs.pkl, the checkpoint at
    /root/model.pt; a C compiler and the hook's feature code for policy/native.py's library. `commands`: shell
    commands run after PyTorch's install (e.g. another package)."""
    import modal

    built = modal.Image.debian_slim(python_version="3.11").apt_install("gcc").pip_install("numpy", "torch==2.8.0")
    if commands:
        built = built.run_commands(*commands)
    return (
        built.add_local_dir(ROOT / "wc3hook", "/root/wc3hook", ignore=lambda p: p.name not in FEATURE_SOURCES)
        .add_local_dir(ROOT / "src" / "wc3env", "/root/wc3env", ignore=["**/*.dll", "**/__pycache__/**"])
        .add_local_dir(ROOT / "training", "/root/training", ignore=["**/__pycache__/**", "tests/**"])
        .add_local_file(OBSERVATIONS, "/root/obs.pkl")
        .add_local_file(CHECKPOINT, "/root/model.pt")
    )


def play(agents, games: list[int], steps: int, traces, static, start: int = 0) -> list[dict]:
    """Seats deciding over recorded games (seat j replays game games[j]'s observations from `start`): the chunks
    they finished, as an actor sends them to the learner."""
    from training.policy.agent import Seat
    from training.policy.native import Features
    from training.selfplay.actor import Recorder

    seats = [Seat(static) for _ in games]
    states = [Features(static) for _ in games]
    recorders, chunks = [None] * len(games), []
    for t in range(start, start + steps):
        obs = [traces[g][t] for g in games]
        blocks = [state.step(o, seat.last) for state, o, seat in zip(states, obs, seats)]
        opponents = [b["scalars"] for b in blocks]
        for j, d in enumerate(agents.act(seats, obs, blocks, opponents)):
            if d.chunk_start:
                if recorders[j] is not None:
                    chunks.append(recorders[j].chunk(bootstrap=d.value))
                recorders[j] = Recorder(static, None, d.start_state, 0)
            recorders[j].add(obs[j], d, opponents[j])
    return chunks
