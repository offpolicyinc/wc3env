# Contributing

## Build and test

```powershell
wc3hook/build.bat                                # x86 hook + native tests; needs MSVC Build Tools
python -m pip install -e . -e wc3agent
python -m unittest tests.test_w3g                # the replay reader on recorded games; CI runs it
python -m ruff check . && python -m ruff format .   # CI enforces both
```

C sources under `wc3hook/` follow `.clang-format` (vendored `minhook/` and `yyjson/` are left untouched).

Tests are end to end: a real game, real replays, the real DLL. No unit tests against fakes, and none
for configuration values. CI has no game, so run the real-game tests locally before changing `wc3hook/`, the RPC or
session/pool lifecycle. They skip when `WC3_GAME_DIR` has no supported installation. Every test
module and scenario owns its game processes, so the runner executes four at a time (`--jobs`).

The test harness disables sound automatically. Set `WC3_SOUND=0` for ad hoc game
probes and other test scripts too; public launches keep sound on by default.

```powershell
python -m tests.e2e                  # focused native regressions and core scenarios
python -m tests.e2e worker_loop      # one scenario
python -m unittest tests.e2e.test_pool   # one module, the ordinary unittest way
```

`tests/e2e/extended/` holds broader fixtures that stay out of the everyday run: use them on request,
after hook or protocol changes, and before tagging. They cover Undead and Night Elf economy and
openings, hero casts at a unit, a point and with no target (a summon), an 8-player map, shops and
taverns, production and build queues, orders and casts the engine refuses, hero revival, rarely
seen events, full-match determinism across resets and process replacement, allocator stress,
extra seeds and a recorded combat replay that playback must reproduce. A new fixture is
usually just a JSON config for an existing scenario class.

```powershell
python -m tests.e2e.extended         # ~4 min, bounded by the seeded full matches
python -m tests.e2e.extended --scenarios-only   # ~1 min
```

`tests/rpc_contract.py` is the RPC conformance suite, run against the real DLL
(`tests/e2e/test_rpc_contract_dll.py`). Extend it when the protocol changes, and update
[docs/specs/](docs/specs/) in the same change.

Keep tests beside the behavior they exercise: protocol, session, clock, actions, observations,
orders, memory outcomes, or model coordination. Avoid catchall release/regression files.
Shared fixtures live in `tests/e2e/support.py`; test modules should not import one another.

## Guidelines

- Commit generated reference data, including `src/wc3env/data/reference.json`,
  with the code that generates or uses it. Do not gitignore generated references.
- Keep game binaries, maps, raw asset dumps, activation files and run output local.
  `assets/`, `runs/` and `sessions/` remain ignored.
- Executable offsets belong in `wc3hook/wc3hook.h` with a comment saying what they address.
  Supporting another game build means verifying every offset, not changing the hash.
- Dependencies run one way: `wc3env` never imports `wc3agent` or `tools`. Inside `wc3agent`, only `play.py`
  imports `wc3env`; everything else takes plain data and stays importable without a game.
- Tests assert what a real game does. Do not pin prompt prose, display wording, prices, scenario counts
  or configuration values.
- Tests should not pin one machine's timing or memory numbers. Measurements
  go in [docs/compatibility.md](docs/compatibility.md) via `tools/check_compatibility.py`.
- Docs: [design](docs/design.md) describes implemented behavior, [specs](docs/specs/) define
  contracts, the [roadmap](docs/roadmap.md) holds unfinished work. Keep each fact in one place.
- This project is for offline research. Changes aimed at online play will not be accepted.
