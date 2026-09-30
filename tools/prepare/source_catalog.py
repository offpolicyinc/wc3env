"""Compile static definitions from installed game archives; preparation only."""

from __future__ import annotations

import csv
import json
import re

from wc3env.data import DIR

from .gamedata import RACES, order_ids, profiles, read_all, slk

# Game facts kept by hand or by measurement sit with the generated data, in the environment package.
GAME_DATA = DIR


# Abilities whose game files name no order: the Hippogryph's Pick up Archer and the Archer's Mount Hippogryph.
MISSING_ORDERS = {"Acoi": "coupleinstant"}


def number(value, default=0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def identifiers(value):
    return [v.strip(' "') for v in (value or "").split(",") if v.strip(' "') not in ("", "_", "-")]


def field(profile, name, default=""):
    return next((v for k, v in profile.items() if k.casefold() == name.casefold()), default)


def damage_multipliers(text):
    """Read class names and matrix values from the installed gameplay constants."""
    header = re.search(r"^\s*// Damage bonus lists:\s*(.+)$", text, re.MULTILINE)
    if not header:
        return {}  # Do not guess column order for unsupported game data.
    armor_classes = [v.strip().lower() for v in header[1].split(",")]
    result = {}
    for attack, values in re.findall(r"^DamageBonus(\w+)=(.+)$", text, re.MULTILINE):
        multipliers = [float(v.strip()) for v in values.split(",")]
        if len(multipliers) != len(armor_classes):
            raise ValueError(f"DamageBonus{attack}: unexpected armor column count")
        result[attack.lower()] = dict(zip(armor_classes, multipliers))
    return result


class SourceCatalog:
    def __init__(self, files, orders, forms):
        self.orders = orders
        self.forms = forms
        self.units = slk(files[r"Units\UnitData.slk"])
        self.balance = slk(files[r"Units\UnitBalance.slk"])
        self.weapons = slk(files[r"Units\UnitWeapons.slk"])
        self.unit_abilities = slk(files[r"Units\UnitAbilities.slk"])
        self.abilities = slk(files[r"Units\AbilityData.slk"])
        self.buffs = slk(files.get(r"Units\AbilityBuffData.slk", ""))
        self.items = slk(files[r"Units\ItemData.slk"])
        self.item_strings = profiles(files.get(r"Units\ItemStrings.txt", ""))
        self.item_functions = profiles(files.get(r"Units\ItemFunc.txt", ""))
        self.damage_multipliers = damage_multipliers(files.get(r"Units\MiscGame.txt", ""))
        self.ability_functions, self.ability_strings, self.unit_strings, self.upgrade_strings = {}, {}, {}, {}
        self.unit_functions, self.upgrade_functions = {}, {}
        self.upgrades = slk(files[r"Units\UpgradeData.slk"])
        for path, content in files.items():
            if path.endswith("UnitFunc.txt"):
                self.unit_functions.update(profiles(content))
            elif path.endswith("UpgradeFunc.txt"):
                self.upgrade_functions.update(profiles(content))
            elif path.endswith("AbilityFunc.txt"):
                self.ability_functions.update(profiles(content))
            elif path.endswith("AbilityStrings.txt"):
                self.ability_strings.update(profiles(content))
            elif path.endswith("UnitStrings.txt"):
                self.unit_strings.update(profiles(content))
            elif path.endswith("UpgradeStrings.txt"):
                self.upgrade_strings.update(profiles(content))
        self.tables = {**self.units, **self.balance, **self.weapons, **self.abilities, **self.items}
        # Some tooltip references mix fields from balance and weapons for the same unit.
        for raw in self.units:
            self.tables[raw] = {**self.units[raw], **self.balance.get(raw, {}), **self.weapons.get(raw, {})}

    @classmethod
    def load(cls, game_dir=None):
        tables = ("UnitData", "UnitBalance", "UnitWeapons", "UnitAbilities", "AbilityData", "ItemData")
        names = [rf"Units\{name}.slk" for name in tables]
        names.append(r"Units\ItemStrings.txt")
        names.append(r"Units\AbilityBuffData.slk")
        names.append(r"Units\ItemFunc.txt")
        names.append(r"Units\MiscGame.txt")
        names.append(r"Units\UpgradeData.slk")
        for race in (*RACES, "Campaign"):
            names += [rf"Units\{race}{kind}.txt" for kind in ("UnitFunc", "UpgradeFunc")]
        for race in (*RACES, "Common", "Item", "Campaign"):
            names += [
                rf"Units\{race}{kind}.txt"
                for kind in ("AbilityFunc", "AbilityStrings", "UnitStrings", "UpgradeStrings")
            ]
        files = read_all(names, game_dir)
        orders = order_ids()
        forms = json.loads((GAME_DATA / "order_targets.json").read_text())["forms"]
        catalog = cls(files, orders, forms)
        measured = GAME_DATA / "item_targets.json"  # from tools/scripts/measure_item_targets.py
        if measured.is_file():
            catalog.item_forms = {raw: v["form"] for raw, v in json.loads(measured.read_text())["items"].items()}
        return catalog

    def text(self, value):
        def substitute(match):
            raw, field, percentage = match.group(1), match.group(2), match.group(3)
            found = self.tables.get(raw, {}).get(field)
            if found is None:
                return match.group(0)  # Preserve unresolved data references explicitly.
            return f"{number(found) * 100:g}" if percentage else found

        value = re.sub(r"<([^,<>]+),([^,<>]+)(,%)*>", substitute, value or "")
        value = re.sub(r"\|c[0-9a-fA-F]{8}|\|r", "", value).replace("|n", " ")
        return " ".join(value.strip('"').split())

    def unit(self, raw):
        strings, balance, weapons = self.unit_strings.get(raw, {}), self.balance.get(raw, {}), self.weapons.get(raw, {})
        data = self.units.get(raw, {})
        return {
            "type_id": raw,
            "name": self.text(strings.get("Name", data.get("comment(s)", raw))),
            # "(Bear Form)", "(Storm Crow Form)": which form of a shape-shifting unit this type is
            "form": suffix if "Form" in (suffix := self.text(field(strings, "EditorSuffix"))) else "",
            "description": self.text(field(strings, "Ubertip")),
            "base_move_speed": number(balance.get("spd")),
            "base_attack_range": number(weapons.get("rangeN1")),
            "base_attack_type": weapons.get("atkType1"),
            "base_armor_class": balance.get("defType"),
            "attack_targets": sorted({tag for index in (1, 2) for tag in identifiers(weapons.get(f"targs{index}"))}),
            "base_attack_period": number(weapons.get("cool1")),
            "base_attack_point": number(weapons.get("dmgpt1")),
            "base_cast_point": number(weapons.get("castpt")),
            "movement_type": data.get("movetp", ""),
            "collision_radius": number(balance.get("collision")),
            "classifications": identifiers(balance.get("type")),
            "potential_hero_abilities": identifiers(self.unit_abilities.get(raw, {}).get("heroAbilList")),
            "source": "Installed game tables; base values can differ after buffs/upgrades.",
            **self.economy(raw),
        }

    @staticmethod
    def requirement_tiers(prof):
        """`Requires`, `Requires1`, ...: a unit's one list, a hero's per hero owned, an upgrade's per level."""
        tiers = [identifiers(prof.get("Requires"))]
        while f"Requires{len(tiers)}" in prof:
            tiers.append(identifiers(prof[f"Requires{len(tiers)}"]))
        return tiers

    def economy(self, raw):
        """What a unit costs, what it needs, what it makes, and its base combat numbers."""
        balance, weapons, prof = self.balance.get(raw, {}), self.weapons.get(raw, {}), self.unit_functions.get(raw, {})
        armed = weapons.get("weapsOn") not in (None, "", "0", "-", "_")
        dice, plus = number(weapons.get("dice1")), number(weapons.get("dmgplus1"))
        tiers = self.requirement_tiers(prof)
        hero = raw[:1].isupper()
        hp = number(balance.get("HP"))
        if hero:  # level 1: 25 hit points per strength, one damage per point of the primary attribute
            hp += 25 * number(balance.get("STR"))
            plus += number(balance.get(str(balance.get("Primary", "")).upper()))
        return {
            "race": self.units.get(raw, {}).get("race", ""),
            "structure": balance.get("isbldg") == "1",
            "hero": hero,
            "gold": int(number(balance.get("goldcost"))),
            "lumber": int(number(balance.get("lumbercost"))),
            "food": int(number(balance.get("fused"))),
            "food_made": int(number(balance.get("fmade"))),
            "build_seconds": number(balance.get("bldtm")),
            "hp": int(hp),
            "mana": int(number(balance.get("manaN"))),
            "armor": number(balance.get("def")),
            "level": int(number(balance.get("level"))),
            "damage": [plus + dice, plus + dice * number(weapons.get("sides1"))] if armed else None,
            # Level 1 attributes and their growth per level; `hp` and `damage` above already include level 1.
            "attributes": {
                "primary": str(balance.get("Primary", "")).lower(),
                **{k.lower(): number(balance.get(k)) for k in ("STR", "AGI", "INT")},
                **{k.lower() + "_per_level": number(balance.get(k + "plus")) for k in ("STR", "AGI", "INT")},
            }
            if hero
            else None,
            "requires": tiers[0],
            # A hero's list per heroes already owned: the first is free, the second needs tier two.
            "requires_by_count": tiers if len(tiers) > 1 else None,
            # UnitFunc can list types UnitData lacks (the Peon's `orbr`); keep only real units.
            "trains": [r for r in identifiers(prof.get("Trains")) if r in self.units],
            "builds": [r for r in identifiers(prof.get("Builds")) if r in self.units],
            "researches": identifiers(prof.get("Researches")),
            "upgrades_to": [r for r in identifiers(prof.get("Upgrade")) if r in self.units],
            "sells_items": identifiers(prof.get("Sellitems")) + identifiers(prof.get("Makeitems")),
            "sells_units": identifiers(prof.get("Sellunits")),
        }

    def upgrade(self, raw):
        row, prof = self.upgrades.get(raw, {}), self.upgrade_functions.get(raw, {})
        strings = {key.lower(): value for key, value in self.upgrade_strings.get(raw, {}).items()}
        names = next(csv.reader([strings.get("name", "")]), [])
        tips = next(csv.reader([strings.get("ubertip", "")]), [])
        tiers = self.requirement_tiers(prof)
        levels = [
            {
                "level": i + 1,
                "name": self.text(names[min(i, len(names) - 1)]) if names else raw,
                "gold": int(number(row.get("goldbase")) + i * number(row.get("goldmod"))),
                "lumber": int(number(row.get("lumberbase")) + i * number(row.get("lumbermod"))),
                "seconds": number(row.get("timebase")) + i * number(row.get("timemod")),
                "requires": tiers[min(i, len(tiers) - 1)],
                "description": self.text(tips[min(i, len(tips) - 1)]) if tips else "",
            }
            for i in range(int(number(row.get("maxlevel"), 1)))
        ]
        return {"type_id": raw, "race": row.get("race", ""), "levels": levels}

    def ability(self, raw, level=1):
        row = self.abilities.get(raw, {})
        base = row.get("code", raw)
        prof = self.ability_functions.get(raw, self.ability_functions.get(base, {}))
        strings = self.ability_strings.get(raw, self.ability_strings.get(base, {}))
        tips = next(csv.reader([field(strings, "Ubertip")]), [])
        description = self.text(tips[min(level - 1, len(tips) - 1)]) if tips else ""
        # Key case differs between the race files (OrderOn in Orc, order in Night Elf): match any case.
        keys = {k.lower(): v for k, v in prof.items()}
        if base in MISSING_ORDERS and "order" not in keys:
            keys["order"] = MISSING_ORDERS[base]
        passive = "PassiveButtons" in prof.get("Art", "")
        if not passive and not any(k in keys for k in ("order", "orderon", "orderoff", "unorder")):
            # A variant that lists no orders (Disenchant on Dispel Magic) is commanded with its base ability's.
            keys.update(
                {
                    k.lower(): v
                    for k, v in self.ability_functions.get(base, {}).items()
                    if k.lower().endswith("order") or k.lower().startswith("order")
                }
            )
        if keys.get("order") and f"{keys['order']}targ" in self.orders:
            # Searing Arrows: 'flamingarrows' switches autocast on, 'unflamingarrows' off, and the one aimed
            # shot is 'flamingarrowstarg'.
            keys = {
                "order": f"{keys['order']}targ",
                "orderon": keys["order"],
                **({"orderoff": keys["unorder"]} if keys.get("unorder") else {}),
            }
        orders = [
            {
                "name": keys[key],
                "order_id": self.orders.get(keys[key]),
                "target_form": self.forms.get(base) if key == "order" else "none",
                "kind": {"order": "cast", "orderon": "enable_autocast", "orderoff": "disable_autocast"}.get(
                    key, "deactivate"
                ),
            }
            for key in ("order", "orderon", "orderoff", "unorder")  # unorder: switch a toggle off (Defend, forms)
            if keys.get(key)
        ]
        return {
            "ability_id": raw,
            "name": self.text(strings.get("Name", row.get("comments", raw))),
            # These engine components expose basic orders/hero/inventory UI,
            # rather than combat skills. Keep them in native diagnostics.
            "interface_ability": base in {"Aatk", "Amov", "AHer", "AInv", "Aihn"},
            "hero_ability": row.get("hero") == "1",
            "item_ability": row.get("item") == "1",
            "level": level,
            "description": description,
            "passive": passive,
            "orders": orders,
            "requires": identifiers(prof.get("Requires")),
            "requirement_counts": {
                r: int(number(n, 1))
                for r, n in zip(
                    identifiers(prof.get("Requires")),
                    identifiers(prof.get("Requiresamount")) or ["1"] * len(identifiers(prof.get("Requires"))),
                )
            },
            "range": number(row.get(f"Rng{level}")),
            "area": number(row.get(f"Area{level}")),
            "duration_seconds": number(row.get(f"Dur{level}")),
            "hero_duration_seconds": number(row.get(f"HeroDur{level}")),
            "targets": identifiers(row.get(f"targs{level}")),
            "effects": {
                k[: -len(str(level))]: v
                for k, v in row.items()
                if re.fullmatch(rf"Data[A-I]{level}|UnitID{level}|BuffID{level}|EfctID{level}", k)
                and v not in ("-", "_", "")
            },
            "command_support": "mapped"
            if any(o["order_id"] and o["target_form"] for o in orders)
            else "passive"
            if passive
            else "unmapped",
            "source": "Installed game ability tables and tooltips; native readback supplies current rank, mana and cooldown.",
        }

    def buff(self, raw):
        """A buff's name and tooltip ('Purge': 'This unit is Purged; ...'); the caller adds the abilities that apply it."""
        strings = self.ability_strings.get(raw, {})
        return {
            "buff_id": raw,
            "name": self.text(field(strings, "Bufftip")) or self.buffs[raw].get("comments", raw).strip(),
            "description": self.text(field(strings, "Buffubertip")),
        }

    def requirement_name(self, raw, level=1):
        if raw in self.units:
            return self.unit(raw)["name"]
        names = next(csv.reader([field(self.upgrade_strings.get(raw, {}), "Name")]), [])
        return self.text(names[min(level - 1, len(names) - 1)]) if names else "Unidentified research"

    def item(self, raw):
        row = self.items.get(raw, {})
        strings = self.item_strings.get(raw, {})
        abilities = [self.ability(aid) for aid in identifiers(row.get("abilList"))]
        forms = {
            self.forms.get(self.abilities.get(aid, {}).get("code", aid)) for aid in identifiers(row.get("abilList"))
        }
        # AIhe is the engine's immediate healing-item class; DataA is restored HP.
        # Recognize all items using that mechanic, independently of their item IDs.
        self_healing = forms == {"none"} and any(
            self.abilities.get(aid, {}).get("code") == "AIhe" and number(self.abilities[aid].get("DataA1")) > 0
            for aid in identifiers(row.get("abilList"))
        )
        name = self.text(strings.get("Name", row.get("comment", raw)))
        written = getattr(self, "item_forms", {}).get(raw)
        return {
            "type_id": raw,
            "name": name,
            "description": self.text(field(strings, "Ubertip")),
            "self_healing": self_healing,
            "usable": row.get("usable") == "1",
            "perishable": row.get("perishable") == "1",
            "abilities": abilities,
            "target_form": next(iter(forms)) if len(forms) == 1 and None not in forms else written,
            "gold": int(number(row.get("goldcost"))),
            "lumber": int(number(row.get("lumbercost"))),
            "level": int(number(row.get("Level"))),
            "class": row.get("class", ""),
            "stock_max": int(number(row.get("stockMax"))),
            "stock_seconds": number(row.get("stockRegen")),
            "stock_start_seconds": number(row.get("stockStart")),
            # What a player's own shop needs before it stocks the item: TWN2/TWN3 mean any tier 2/3 hall.
            "requires": identifiers(self.item_functions.get(raw, {}).get("Requires")),
        }
