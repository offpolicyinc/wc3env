"""One hooked game process, driven over the RPC (docs/specs/protocol.md).

    from wc3env.game import launch
    game = launch()
    game.rpc.create_game("(2)EchoIsles.w3x", players, "stepping")
    ...
    game.close()

`launch(map=...)` selects the map, defaulting to settings.map, with wc3hook.dll injected (`inject.py`:
the game is created suspended, the DLL loaded by a remote thread, the main thread resumed once
the DLL signals ready) and connects the pipe. The DLL holds the game at a fixed game time in
step mode; `create_game` takes it from `launched` to `in_game`.
"""

from __future__ import annotations

import ctypes
import json
import uuid
from ctypes import wintypes
from pathlib import Path

from .inject import k32, launch_with_dll
from .pipe import HookPipe
from .rpc import RpcClient
from .settings import settings

ROOT = Path(__file__).resolve().parents[2]
HOOK_OUT = ROOT / "wc3hook" / "out"
DLL = HOOK_OUT / "wc3hook.dll"
user32 = ctypes.WinDLL("user32", use_last_error=True)
_EnumWindowProc = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)
user32.EnumWindows.argtypes = [_EnumWindowProc, wintypes.LPARAM]
user32.GetWindowThreadProcessId.argtypes = [wintypes.HWND, wintypes.LPDWORD]
user32.IsWindowVisible.argtypes = [wintypes.HWND]
user32.GetWindowRect.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.RECT)]
user32.SetWindowPos.argtypes = [
    wintypes.HWND,
    wintypes.HWND,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    wintypes.UINT,
]


def hook_dll() -> Path:
    override = getattr(settings(), "hook_dll", None)
    if override:
        return override
    bundled = Path(__file__).with_name("native") / "wc3hook.dll"
    return bundled if bundled.is_file() else DLL


class Game:
    def __init__(self, pid: int, pipe: HookPipe, map: Path, docs: Path | None = None, *, process_handle: int):
        self.pid = pid
        self._process_handle = process_handle
        self.pipe = pipe
        self.rpc = RpcClient(pipe)
        self.map = map
        self._closed = False
        self.data_dir = docs
        self.startup: dict = {}  # match setup, AI difficulty and AI slots this process started with

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            # Keep the original handle: a crashed child's PID can be reused by another process.
            k32.TerminateProcess(self._process_handle, 0)
            k32.WaitForSingleObject(self._process_handle, 5000)
        finally:
            k32.CloseHandle(self._process_handle)
            self.pipe.close()

    def log(self) -> str:
        """The DLL's log in this launch's data directory."""
        return ((self.data_dir or HOOK_OUT) / f"hook-{self.pid}.log").read_text(errors="replace")

    # ---- the game window --------------------------------------------------------------------
    def windows(self) -> list[int]:
        found = []

        @_EnumWindowProc
        def cb(hwnd, _):
            pid = wintypes.DWORD()
            user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
            if pid.value == self.pid and user32.IsWindowVisible(hwnd):
                found.append(hwnd)
            return True

        user32.EnumWindows(cb, 0)
        return found

    def resize(self, w: int = 640, h: int = 480) -> bool:
        """Request a window size; Warcraft may clamp it to its minimum dimensions.
        True means a request was posted, not that the exact size was applied.
        SWP_ASYNCWINDOWPOS avoids deadlock while the game thread is frozen."""
        if any(type(v) is not int or not 0 < v <= 0x7FFFFFFF for v in (w, h)):
            raise ValueError("window dimensions must be positive 32-bit integers")
        flags = 0x0004 | 0x0010 | 0x0040 | 0x4000  # NOZORDER | NOACTIVATE | SHOWWINDOW | ASYNCWINDOWPOS
        for hwnd in self.windows():
            if not user32.SetWindowPos(hwnd, 0, 0, 0, w, h, flags):
                raise ctypes.WinError(ctypes.get_last_error())
        return bool(self.windows())

    def offscreen(self) -> bool:
        """Park the window off the desktop: still 'visible' to the game (so no auto-pause), but
        nobody can click or close it by accident. Posted, not sent (see resize)."""
        flags = (
            0x0004 | 0x0001 | 0x0010 | 0x0040 | 0x4000
        )  # NOZORDER | NOSIZE | NOACTIVATE | SHOWWINDOW | ASYNCWINDOWPOS
        for hwnd in self.windows():
            rect = wintypes.RECT()
            if not user32.GetWindowRect(hwnd, ctypes.byref(rect)):
                raise ctypes.WinError(ctypes.get_last_error())
            x = user32.GetSystemMetrics(76) - (rect.right - rect.left) - 1
            y = user32.GetSystemMetrics(77) - (rect.bottom - rect.top) - 1
            if not user32.SetWindowPos(hwnd, 0, x, y, 0, 0, flags):
                raise ctypes.WinError(ctypes.get_last_error())
        return bool(self.windows())


def user_dir(instance: int | None = None, *, output_dir: str | Path | None = None) -> Path:
    """Per-instance 'Documents' handed to the game through the DLL (WC3HOOK_DOCS), so instances
    do not share Warcraft III\\Replay, CustomMapData, or telemetry files."""
    cfg = settings()
    root = output_dir if output_dir is not None else getattr(cfg, "output_dir", None)
    root = Path(root).expanduser().absolute() if root is not None else cfg.user_dir.parent / "wc3env"
    label = "game" if instance is None else f"inst{instance}"
    return root / f"{label}-{uuid.uuid4().hex}"


def replay_startup_path(replay: Path) -> Path:
    """Where save_replay keeps a replay's startup options (match setup, AI difficulty, AI slots)."""
    return replay.with_name(replay.name + ".json")


def resolve_map(map: str | Path | None = None) -> Path:
    """An absolute path, a path under the install, or a unique filename under Maps/."""
    cfg = settings()
    if map is None:
        path = cfg.map
    else:
        path = Path(map)
        if not path.is_absolute():
            if len(path.parts) == 1:
                matches = list((cfg.game_dir / "Maps").rglob(path.name))
                if len(matches) != 1:
                    raise ValueError(f"map name {map!r} matched {len(matches)} files; use an explicit path")
                path = matches[0]
            else:
                path = cfg.game_dir / path
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"No map at {path}; check WC3_GAME_DIR and WC3_MAP (see .env.example)")
    return path


def launch(
    instance: int | None = None,
    window: bool = True,
    connect_timeout: float = 60.0,
    *,
    map: str | Path | None = None,
    agents: tuple[int, ...] = (0,),
    window_mode: str = "background",
    render: bool = True,
    sound: bool | None = None,
    background_visible: bool = False,
    ai_difficulty: int | None = None,
    ai_agents: tuple[int, ...] = (),
    output_dir: str | Path | None = None,
    _setup: dict | None = None,
) -> Game:
    """Start offscreen for automation, or use interactive mode for normal window/input behavior.

    Rendering controls presentation only; both modes still create a game window. See docs/design.md#rendering-and-input.
    Sound defaults to settings.sound (on); sound=False disables it from startup.
    background_visible shows an input-isolated background window. ai_difficulty
    optionally sets unclaimed melee AI to 0 (easy), 1 (normal), or 2 (insane).
    ai_agents are agent slots whose melee AI still runs; step() orders act alongside it.
    output_dir chooses the parent for unique per-process profiles and logs. It
    overrides WC3_OUTPUT_DIR and is independent of any agent or run-directory layout.
    """
    if window_mode not in ("interactive", "background"):
        raise ValueError("window_mode must be interactive or background")
    if window_mode == "background" and not window:
        raise ValueError("background mode requires a window; use interactive mode for fullscreen")
    if type(render) is not bool:
        raise ValueError("render must be a bool")
    if sound is not None and type(sound) is not bool:
        raise ValueError("sound must be a bool or None")
    if sound is None:
        sound = settings().sound
    if not agents or any(type(p) is not int or not 0 <= p < 16 for p in agents) or len(set(agents)) != len(agents):
        raise ValueError("agents must contain unique player slots in 0..15")
    if type(background_visible) is not bool or (background_visible and window_mode != "background"):
        raise ValueError("background_visible requires background window mode and a bool")
    if ai_difficulty is not None and (type(ai_difficulty) is not int or ai_difficulty not in (0, 1, 2)):
        raise ValueError("ai_difficulty must be None, 0 (easy), 1 (normal), or 2 (insane)")
    map_path = resolve_map(map)
    if map_path.suffix.lower() == ".w3g":
        # A staged game replays only from the same start: the setup shapes the map's player configuration and
        # the AI difficulty its melee AI, neither of which the .w3g records; save_replay keeps them beside it.
        if _setup is not None or ai_difficulty is not None:
            raise ValueError("replay playback takes match setup and AI difficulty from the replay's saved startup")
        saved = replay_startup_path(map_path)
        if saved.is_file():
            startup = json.loads(saved.read_text(encoding="utf-8"))
            _setup, ai_difficulty = startup.get("setup"), startup.get("ai_difficulty")
            ai_agents = tuple(startup.get("ai_agents", ())) or ai_agents
    if not set(ai_agents) <= set(agents):
        raise ValueError("ai_agents must be agent slots")
    try:
        # Warcraft reads -loadfile through its ANSI command line, even though injection is Unicode.
        if str(map_path).encode("mbcs", errors="strict").decode("mbcs") != str(map_path):
            raise UnicodeError("map path cannot round-trip through the Windows code page")
    except UnicodeError as exc:
        raise ValueError(
            "Warcraft cannot read this map path in the active Windows code page; "
            "move the map to a path using representable characters (ASCII is portable)"
        ) from exc
    docs = user_dir(instance, output_dir=output_dir)
    # This legacy engine uses MAX_PATH buffers, including generated replay filenames.
    if len(str(map_path)) >= 260 or len(str(docs / "Warcraft III" / "Replay" / "TempReplay.w3g")) >= 260:
        raise ValueError("Warcraft III requires paths shorter than 260 characters; choose shorter map/output paths")
    docs.mkdir(parents=True, exist_ok=True)
    args = (["-window"] if window else []) + ["-loadfile", str(map_path)]
    # WC3HOOK_DOCS is read by wc3hook.dll; passed to the child only, so concurrent launches do not race
    pid, process_handle = launch_with_dll(
        settings().game_exe,
        hook_dll(),
        args,
        env={
            "WC3HOOK_DOCS": str(docs),
            "WC3HOOK_MAP": str(map_path),
            "WC3HOOK_AGENTS": str(sum(1 << p for p in agents)),
            "WC3HOOK_LOG_DIR": str(docs),
            "WC3HOOK_WINDOW_MODE": window_mode,
            "WC3HOOK_RENDER": str(int(render)),
            "WC3HOOK_SOUND": str(int(sound)),
            "WC3HOOK_BACKGROUND_VISIBLE": str(int(background_visible)),
            "WC3HOOK_AGENT_AI": str(sum(1 << p for p in ai_agents)),
            "WC3HOOK_AI_DIFFICULTY": "" if ai_difficulty is None else str(ai_difficulty),
            "WC3HOOK_SETUP": json.dumps(_setup, allow_nan=False) if _setup is not None else "",
        },
    )
    game = Game(pid, HookPipe(pid), map_path, docs, process_handle=process_handle)
    game.startup = {
        k: v for k, v in {"setup": _setup, "ai_difficulty": ai_difficulty, "ai_agents": list(ai_agents)}.items() if v
    }
    try:
        game.pipe.connect(connect_timeout)
        return game
    except Exception:
        game.close()
        raise
