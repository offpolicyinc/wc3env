"""Stop everything of ours on Modal: every running app named wc3-selfplay, every sandbox in it, every container of
it, repeated until none is left (deep-sports and other apps are never touched). Prints what it stopped."""

import subprocess
import sys
import time

import modal


def ours() -> tuple[list[str], list]:
    containers = (
        [c.task_id for c in modal.container.list() if getattr(c, "app_name", "") == "wc3-selfplay"]
        if hasattr(modal, "container")
        else []
    )
    return containers, []


def main() -> int:
    for attempt in range(10):
        apps = subprocess.run([sys.executable, "-m", "modal", "app", "list", "--json"], capture_output=True, text=True)
        import json

        running = [
            a["app_id"]
            for a in json.loads(apps.stdout or "[]")
            if a.get("description") == "wc3-selfplay" and a.get("state") not in ("stopped", "deployed")
        ]
        for app_id in running:
            subprocess.run([sys.executable, "-m", "modal", "app", "stop", app_id, "--yes"], capture_output=True)
        deployed = modal.App.lookup("wc3-selfplay")
        boxes = list(modal.Sandbox.list(app_id=deployed.app_id))
        for box in boxes:
            box.terminate()
        listed = subprocess.run(
            [sys.executable, "-m", "modal", "container", "list", "--json"], capture_output=True, text=True
        )
        containers = [
            c["container_id"] for c in json.loads(listed.stdout or "[]") if c.get("app_name") == "wc3-selfplay"
        ]
        for c in containers:
            subprocess.run([sys.executable, "-m", "modal", "container", "stop", c, "--yes"], capture_output=True)
        print(
            f"pass {attempt}: stopped {len(running)} apps, {len(boxes)} sandboxes, {len(containers)} containers",
            flush=True,
        )
        if not running and not boxes and not containers:
            return 0
        time.sleep(15)
    return 1


if __name__ == "__main__":
    sys.exit(main())
