"""Write every unit's combat role into wc3agent's unit_roles.json.

Hand-picked roles ("assigned") win; every other non-building unit in reference.json gets one from its
stats (wc3agent.game.roles.stats_role). Run after regenerating reference.json:

    python -m tools.prepare.unit_roles
"""

import json
from collections import Counter

from wc3agent.game.roles import ROLES, TABLE, stats_role

from wc3env.data import REFERENCE


def main():
    units = json.loads(REFERENCE.read_text(encoding="utf-8"))["units"]
    table = json.loads(TABLE.read_text(encoding="utf-8"))
    assigned = table["assigned"]
    unknown = {raw: kind for raw, kind in assigned.items() if kind not in ROLES}
    if unknown:
        raise ValueError(f"unknown roles: {unknown}")
    table["roles"] = {
        raw: assigned.get(raw) or stats_role(unit) for raw, unit in sorted(units.items()) if not unit.get("structure")
    }
    TABLE.write_text(json.dumps(table, indent=1) + "\n", encoding="utf-8")
    print(f"{len(table['roles'])} units:", dict(Counter(table["roles"].values())))


if __name__ == "__main__":
    main()
