/* The observation: unit and item accessors copied from the natives, fog, inventory, ground items, destructables, the
 * result. */
#include "wc3hook.h"
#include "orders_table.h"
#include "upgrades_table.h"

/* Enumerate units without a group handle. 0x3f4ba0(class, cb, ctx, 0) is the game's own
 * "for every live object of this class" (GroupEnumUnitsOfPlayer ends in it); the unit class
 * id is 0x2b773375. The callback gets the unit object and must return nonzero to continue.
 * Fields are read as the natives read them after their handle->object step:
 *   type id     [obj+0x34]                       (GetUnitTypeId)
 *   position    obj->vtbl[0xb8/4](&v3) then 0x3fcc50 -> float x,y   (GetUnitX/Y)
 *   owner       0x292340(obj) -> player object, byte +0x30 = index (GetOwningPlayer, Player)
 *   life etc.   0x2935c0(obj, &out, state) -> &float, state 0..3 = life, max life, mana, max mana */
typedef void(__cdecl *EnumObjectsFn)(int cls, void *cb, void *ctx, int zero);
BYTE *g_fn_owner, *g_fn_state, *g_fn_unpack; /* g_base + RVA_*, for the naked bodies */
/* the accessors copy the natives' instruction sequences (they rely on caller-cleanup quirks) */
__declspec(naked) int __cdecl unit_xy_bits(BYTE *u, int idx) { /* float bits of x (idx 0) or y (4) */
    __asm {
        push ebp
        mov ebp, esp
        sub esp, 0xc
        mov ecx, [ebp + 8]
        lea eax, [ebp - 0xc]
        push eax
        mov eax, [ecx]
        call dword ptr [eax + 0xb8]
        mov ecx, eax
        call dword ptr [g_fn_unpack]
        add eax, [ebp + 0xc]
        mov eax, [eax]
        mov esp, ebp
        pop ebp
        ret
    }
}
__declspec(naked) int __cdecl unit_state_bits(BYTE *u, int state) { /* 0 life 1 max 2 mana 3 max */
    __asm {
        push ebp
        mov ebp, esp
        push esi
        mov esi, [ebp + 0xc]
        push esi
        lea ecx, [ebp + 0xc]
        push ecx
        mov ecx, [ebp + 8]
        call dword ptr [g_fn_state]
        pop esi
        mov eax, [eax]
        pop ebp
        ret
    }
}
__declspec(naked) BYTE *__cdecl unit_owner(BYTE *u) {
    __asm {
        mov ecx, [esp + 4]
        jmp dword ptr [g_fn_owner]
    }
}
/* GetPlayerId's body after its handle step: the internal slot (byte +0x30), mapped 24..27 ->
 * 12..15 for maps whose script version is below 0x17ac (the neutral players on legacy maps).
 * The version is the loaded map's, read once per map (obs_clear forgets it on reload): every unit an
 * observation lists, every event and every order asks, and the read goes through the JASS VM. */
BYTE *g_vm_global, *g_fn_ctx, *g_fn_ver;
static int g_script_version = -1;
static int map_script_version(void) {
    if (!g_game)
        return script_version(); /* no map yet */
    if (g_script_version < 0)
        g_script_version = script_version();
    return g_script_version;
}
void obs_clear(void) {
    g_script_version = -1;
}
int __cdecl player_jass_id(BYTE *pl) {
    int slot = pl[0x30];
    return (unsigned)(slot - 24) <= 3 && (unsigned)map_script_version() < 0x17ac ? slot - 12 : slot;
}
/* IsUnitVisible(u, p): unit->vtbl[0xfc/4](player slot, 0, 4), callee cleans (thiscall, 3 args) */
__declspec(naked) int __cdecl unit_visible(BYTE *u, int player_slot) {
    __asm {
        push 4
        push 0
        push dword ptr [esp + 12 + 4]
        mov ecx, [esp + 12 + 4]
        mov eax, [ecx]
        call dword ptr [eax + 0xfc]
        ret
    }
}
/* UnitItemInSlot: inventory ability [u+0x404] -> 0x291820(u, slot), ret 4 */
BYTE *g_fn_slot;
BYTE *g_fn_resolve_pair; /* 0x3fc7e0: (ecx = &ref pair) -> object or 0 */
__declspec(naked) BYTE *__cdecl unit_item_in_slot(BYTE *u, int slot) {
    __asm {
        push dword ptr [esp + 8]
        mov ecx, [esp + 4 + 4]
        call dword ptr [g_fn_slot]
        ret
    }
}
/* IsItemOwned (0x990f0): ref pair +0x194/+0x198 == -1 -> not owned; otherwise owned only if the
 * pair still resolves through 0x3fc7e0 (a dropped item can keep a stale pair) */
__declspec(naked) int __cdecl item_owned(BYTE *it) {
    __asm {
        mov ecx, [esp + 4]
        mov eax, [ecx + 0x198]
        lea ecx, [ecx + 0x194]
        and eax, [ecx]
        cmp eax, -1
        je none
        call dword ptr [g_fn_resolve_pair]
        test eax, eax
        setne al
        movzx eax, al
        ret
    none:
        xor eax, eax
        ret
    }
}
float bits_to_f(int i) {
    union {
        float f;
        int i;
    } v;
    v.i = i;
    return v.f;
}
/* Observation contents (docs/specs/observations.md; implementation: docs/design.md):
 *   fog        IsUnitVisible's check on the unit object (vtable slot 0xfc, args slot,0,4)
 *   inventory  UnitItemInSlot: [u+0x404] inventory ability -> 0x291820(u, slot); item type [+0x34],
 *              charges [+0x18c]
 *   items      objects of registry class 'item' (0x6974656d) that are not hidden ([+0x20] & 1, IsItemVisible),
 *              not owned (IsItemOwned), and alive (GetWidgetLife's item getter reads ITEM_LIFE).
 *              Consumed power-ups can linger unowned and unhidden, with zero life.
 * IsUnitType(STRUCTURE): flags [u+0x164] & 0x10000. IsUnitType(HERO): type id's first letter is
 * upper case ([u+0x37]) and flags bit 0x40000000 (illusion) is clear; observations call an illusion of a
 * hero a hero, as a player sees it, and flag only the observer's own. GetHeroLevel: hero record [u+0x3fc],
 * its +0x70 read through the game's protected-int accessor 0x3fb2b0 (thiscall, no args), + 1. */
/* unit flag bits at [u+0x164], from the natives: 0x10 IsUnitLoaded (in a transport; a worker inside
 * a mine sets it too), 0x10000 STRUCTURE, 0x200000 IsUnitPaused, 0x40000000 IsUnitIllusion.
 * [u+0x20] & 1 is IsUnitHidden (ShowUnit false); the registry enumerator still lists those. */
#define UF_LOADED 0x10
#define UF_STRUCTURE 0x10000
#define UF_PAUSED 0x200000
#define UF_ILLUSION 0x40000000
#define UF_FLYING 0x20000000 /* every flyer has it (chimaera, hippogryph, gryphon, storm crow) */

static DWORD unit_flags(BYTE *u) {
    return *(DWORD *)(u + 0x164);
}
static int unit_is_hero(BYTE *u) { /* a hero type, illusions included */
    BYTE c = u[0x37];
    return c >= 'A' && c <= 'Z';
}
static int unit_is_structure(BYTE *u) {
    return (unit_flags(u) & UF_STRUCTURE) != 0;
}
int unit_hero_level(BYTE *u) {
    BYTE *h = *(BYTE **)(u + 0x3fc);
    int r;
    BYTE *fn = g_base + RVA_PROTECTED_INT;
    if (!unit_is_hero(u) || !h)
        return 0;
    h += 0x70;
    __asm { mov ecx, h }
    __asm {
        call fn
    }
    __asm { mov r, eax }
    return r + 1;
}
/* The RPC names a unit or item by the game's own object id ([o+0xc]: allocated in creation order,
 * stable for the object's life, identical across runs with the same inputs), never by address. */
unsigned obs_id(BYTE *o) {
    return *(DWORD *)(o + 0xc);
}

/* One observation in the making: the tables of obsbin.h, filled once by the enumeration callbacks. The
 * binary observation copies them out; the JSON observation is formatted from them. */
typedef struct {
    int player, slot; /* JASS id, internal fog slot */
    BB t[OBS_TABLES];
} Obs;
static const DWORD RECORD_SIZE[OBS_TABLES] = {sizeof(BinUnit),         sizeof(BinAbility), sizeof(BinBuff),
                                              sizeof(BinQueue),        sizeof(BinInventory), sizeof(BinItem),
                                              sizeof(BinDestructable), sizeof(BinEvent),   sizeof(BinHero),
                                              sizeof(BinResearch)};
#define RECORDS(o, table, type) ((const type *)(o)->t[table].p)
#define COUNT(o, table) ((o)->t[table].n / RECORD_SIZE[table])

/* IsVisibleToPlayer (0x9a4a0) after its handle step: the fog map's point query with the player's
 * bit. Observations test every destructable, so skipping the Player native and handle lookup
 * per test matters. Slots past 15 (legacy neutral ids) get no bit, as the native's 16-bit shift. */
static int point_visible(float x, float y, int slot) {
    typedef int(__fastcall * FogQueryFn)(BYTE *, void *, float, float, float, int);
    BYTE *world = *(BYTE **)(g_base + RVA_FOG_WORLD);
    BYTE *fog = world ? *(BYTE **)(world + 0x34) : NULL;
    if (!fog)
        return 0;
    float z = *(float *)(g_base + RVA_FOG_QUERY_Z);
    return ((FogQueryFn)(g_base + RVA_FOG_QUERY))(fog, NULL, x, y, z, slot < 16 ? 1 << slot : 0);
}

/* GetDestructableLife: dead if vtbl[0x13c](d) is nonzero, else life from vtbl[0x12c](d, &out) */
static __declspec(naked) int __cdecl destr_dead(BYTE *d) {
    __asm {
        mov ecx, [esp + 4]
        mov eax, [ecx]
        call dword ptr [eax + 0x13c]
        ret
    }
}
static __declspec(naked) int __cdecl destr_life_bits(BYTE *d) {
    __asm {
        push ebp
        mov ebp, esp
        sub esp, 4
        lea eax, [ebp - 4]
        push eax
        mov ecx, [ebp + 8]
        mov eax, [ecx]
        call dword ptr [eax + 0x12c]
        mov eax, [ebp - 4]
        mov esp, ebp
        pop ebp
        ret
    }
}
static int item_on_ground(BYTE *it) {
    return *(DWORD *)(it + 0x34) && !(*(DWORD *)(it + 0x20) & 1) && bits_to_f(*(int *)(it + ITEM_LIFE)) > 0 &&
           !item_owned(it);
}
int object_visible(BYTE *object, int cls, int player) {
    if (*(DWORD *)(object + 0x20) & 1)
        return 0;
    if (cls == UNIT_CLASS)
        return unit_visible(object, player_slot(player));
    if (cls == ITEM_CLASS && !item_on_ground(object))
        return 0;
    float x = bits_to_f(unit_xy_bits(object, 0)), y = bits_to_f(unit_xy_bits(object, 4));
    return point_visible(x, y, player_slot(player));
}
static int __cdecl obs_destructable_cb(BYTE *d, void *ctx) {
    Obs *o = (Obs *)ctx;
    float hp;
    if (destr_dead(d) || (hp = bits_to_f(destr_life_bits(d))) <= 0 || !object_visible(d, DESTRUCTABLE_CLASS, o->player))
        return 1;
    /* The engine's destructable target-mask getter reads the loaded object-data row (including custom
     * types). 0x40 is the tree target class, not a type-ID list. */
    typedef DWORD(__fastcall * TargetMaskFn)(BYTE *, void *);
    DWORD targets = ((TargetMaskFn)(g_base + 0x2ce930))(d, NULL);
    BinDestructable r = {obs_id(d), *(DWORD *)(d + 0x34), bits_to_f(unit_xy_bits(d, 0)), bits_to_f(unit_xy_bits(d, 4)),
                         hp, ((targets & 0x40) ? BD_LUMBER : 0) | ((*(DWORD *)(d + 0x20) & 8) ? BD_INVULNERABLE : 0)};
    bb_push(&o->t[T_DESTRUCTABLES], &r, sizeof r);
    return 1;
}
static int __cdecl obs_item_cb(BYTE *it, void *ctx) {
    Obs *o = (Obs *)ctx;
    if (!item_on_ground(it))
        return 1;
    BinItem r = {obs_id(it), *(DWORD *)(it + 0x34), bits_to_f(unit_xy_bits(it, 0)), bits_to_f(unit_xy_bits(it, 4))};
    if (point_visible(r.x, r.y, o->slot))
        bb_push(&o->t[T_ITEMS], &r, sizeof r);
    return 1;
}

/* a ref pair at an address -> its object, or NULL for the empty pair (-1, -1) */
static __declspec(naked) BYTE *__cdecl pair_object(BYTE *pair) {
    __asm {
        mov ecx, [esp + 4]
        mov eax, [ecx + 4]
        and eax, [ecx]
        cmp eax, -1
        je none
        jmp dword ptr [g_fn_resolve_pair]
    none:
        xor eax, eax
        ret
    }
}
/* an ability object's class tag: vtbl[0x1c]() ('Aque', 'ABnP', ...) */
static __declspec(naked) DWORD __cdecl ability_tag(BYTE *ability) {
    __asm {
        mov ecx, [esp + 4]
        mov eax, [ecx]
        jmp dword ptr [eax + 0x1c]
    }
}
/* The unit's current order, as GetUnitCurrentOrder (0x973a0) reads it: the pair at u+0x3a8 -> the order
 * object, id at +0x24, or NULL when idle. The same object holds the point at +0x48/+0x50 and the target's
 * pair at +0x58, whose first word is the target's object id (measured on harvest gold/tree, move and
 * build). A build order's id is the structure's type id. */
static BYTE *current_order(BYTE *u) {
    BYTE *ord = pair_object(u + 0x3a8);
    return ord && *(DWORD *)(ord + 0x24) ? ord : NULL;
}
int unit_order_point(BYTE *u, unsigned *id, float *x, float *y) {
    BYTE *ord = current_order(u);
    if (!ord)
        return 0;
    *id = *(DWORD *)(ord + 0x24);
    *x = *(float *)(ord + 0x48);
    *y = *(float *)(ord + 0x50);
    return 1;
}

/* Abilities and buffs: every ability the unit has learned, with the live numbers the natives give, and
 * the buffs on any unit in view. Both sit on the unit's ability chain (u+0x3e8, next at +0x24), buff ids
 * starting with B ('Bprg' purged, 'Bblo' bloodlust). Internal abilities (movement, inventory, production)
 * are on the chain too; the client tells them apart by id. Nothing here says whether a cast would succeed,
 * and a buff's remaining time is not read. */
/* The class tag is the ability's base class ('Aprg'); a variant the unit actually has (the Shaman's 'Apg2',
 * the Spirit Walker's 'Adcn' on class 'Adis') reads level 0 under it. The object also holds its own id:
 * find the four-character code in it that the unit has a level of. Returns that level (0 if none). */
static int variant_id(BYTE *a, int handle, DWORD *aid, int buff) {
    static int offsets[2] = {-1, -1}; /* found once each for abilities and buffs, then read directly */
    int offset = offsets[buff];
    if (offset >= 0) {
        DWORD id = *(DWORD *)(a + offset);
        int level = NATIVE(RVA_N_UNITABILITYLEVEL, NativeI_II)(handle, (int)id);
        if (level > 0) {
            *aid = id;
            return level;
        }
    }
    for (int off = 0x28; off <= 0x100; off += 4) {
        DWORD id = *(DWORD *)(a + off);
        BYTE first = (BYTE)(id >> 24);
        if (id == *aid || (buff ? first != 'B' : first != 'A' && first != 'S'))
            continue; /* ability ids start with A (or S for some specials), buff ids with B */
        int level = NATIVE(RVA_N_UNITABILITYLEVEL, NativeI_II)(handle, (int)id);
        if (level > 0) {
            if (offset < 0) {
                offsets[buff] = off;
                hook_log("%s: variant ids at +0x%x", buff ? "buffs" : "abilities", off);
            }
            *aid = id;
            return level;
        }
    }
    return 0;
}
/* One walk of the chain: buffs of any unit, and for the observer's own units also abilities and a
 * structure's production: 'ABnP' while under construction and 'AUnP' while upgrading to another structure,
 * each with the total seconds at +0x7c; 'Aque' holds the queued unit and research type ids from +0xa8, front
 * first, zero after the last, and the front item's total seconds at +0x7c (the walk
 * UnitSetConstructionProgress 0x2d8bc0 does). */
static void gather_chain(Obs *o, BYTE *u, BinUnit *r, int handle) {
    int own = (r->flags & BU_OWN) != 0, n = 0;
    BYTE *queue = NULL;
    for (BYTE *a = pair_object(u + 0x3e8); a && n < 64; a = pair_object(a + 0x24), n++) {
        DWORD id = ability_tag(a);
        int buff = (BYTE)(id >> 24) == 'B';
        if (!buff && !own)
            continue;
        if (own && (r->flags & BU_STRUCTURE) && (id == 'Aque' || id == 'ABnP' || id == 'AUnP')) {
            if (id == 'Aque')
                queue = a;
            else {
                r->state = id == 'ABnP' ? 1 : 2;
                r->state_seconds = *(float *)(a + 0x7c);
            }
        }
        if (!handle)
            break;
        int level = NATIVE(RVA_N_UNITABILITYLEVEL, NativeI_II)(handle, (int)id);
        if (level <= 0 && (level = variant_id(a, handle, &id, buff)) <= 0)
            continue;
        if (buff) {
            BinBuff b = {r->unit_id, id};
            bb_push(&o->t[T_BUFFS], &b, sizeof b);
        } else {
            BinAbility b = {r->unit_id, id, level, NATIVE(RVA_N_ABILITYMANACOST, NativeI_III)(handle, (int)id, level),
                            bits_to_f(NATIVE(RVA_N_ABILITYCOOLDOWN, NativeI_III)(handle, (int)id, level)),
                            bits_to_f(NATIVE(RVA_N_ABILITYCOOLDOWNLEFT, NativeI_II)(handle, (int)id))};
            bb_push(&o->t[T_ABILITIES], &b, sizeof b);
        }
    }
    for (DWORD i = 0; queue && i < 7 && *(DWORD *)(queue + 0xa8 + 4 * i); i++) {
        BinQueue q = {r->unit_id, i, *(DWORD *)(queue + 0xa8 + 4 * i)};
        bb_push(&o->t[T_QUEUE], &q, sizeof q);
    }
    r->queue_seconds = queue && *(DWORD *)(queue + 0xa8) ? *(float *)(queue + 0x7c) : 0;
}

/* obsbin.h BinHero: attributes for any visible hero, XP and skill points for the observer's own */
static void obs_hero(Obs *o, DWORD unit_id, int handle, int own) {
    BinHero h = {unit_id};
    if (own) {
        h.xp = NATIVE(RVA_N_HEROXP, NativeI_I)(handle);
        h.skill_points = NATIVE(RVA_N_HEROSKILLPOINTS, NativeI_I)(handle);
    }
    h.strength = NATIVE(RVA_N_HEROSTR, NativeI_II)(handle, 1);
    h.agility = NATIVE(RVA_N_HEROAGI, NativeI_II)(handle, 1);
    h.intelligence = NATIVE(RVA_N_HEROINT, NativeI_II)(handle, 1);
    bb_push(&o->t[T_HEROES], &h, sizeof h);
}
/* UnitItemInSlot: inventory ability [u+0x404]; item type [+0x34], charges [+0x18c] */
static void obs_inventory(Obs *o, BYTE *u, DWORD unit_id) {
    for (DWORD slot = 0; *(BYTE **)(u + 0x404) && slot < 6; slot++) {
        BYTE *it = unit_item_in_slot(u, slot);
        if (it) {
            BinInventory v = {unit_id, slot, *(DWORD *)(it + 0x34), *(int *)(it + 0x18c)};
            bb_push(&o->t[T_INVENTORY], &v, sizeof v);
        }
    }
}
/* obsbin.h BinUnit's info-panel numbers; this build's weapon natives count weapons from 1 */
static void obs_combat(BinUnit *r, int handle) {
    r->armor = bits_to_f(NATIVE(RVA_N_UNITARMOR, NativeI_I)(handle));
    int base = NATIVE(RVA_N_UNITBASEDAMAGE, NativeI_II)(handle, 1);
    int dice = NATIVE(RVA_N_UNITDICENUMBER, NativeI_II)(handle, 1);
    int sides = NATIVE(RVA_N_UNITDICESIDES, NativeI_II)(handle, 1);
    if (dice > 0 && sides > 0) {
        r->damage_min = base + dice;
        r->damage_max = base + dice * sides;
    }
    r->attack_period = bits_to_f(NATIVE(RVA_N_UNITATTACKPERIOD, NativeI_II)(handle, 1));
    r->move_speed = bits_to_f(NATIVE(RVA_N_UNITMOVESPEED, NativeI_I)(handle));
    r->facing = bits_to_f(NATIVE(RVA_N_UNITFACING, NativeI_I)(handle));
}

static int __cdecl obs_unit_cb(BYTE *u, void *ctx) {
    Obs *o = (Obs *)ctx;
    float life = bits_to_f(unit_state_bits(u, 0));
    BYTE *pl = unit_owner(u);
    int owner = pl ? player_jass_id(pl) : -1, own = owner == o->player, inside = 0;
    /* every row is alive but the observer's dead heroes (BU_DEAD), listed for revival */
    int dead = life <= 0;
    if (dead && !(own && unit_is_hero(u) && !(unit_flags(u) & UF_ILLUSION)))
        return 1;
    if (!dead) {
        /* what GroupEnumUnitsInRect leaves out: hidden ([u+0x20] & 1; a worker in a mine is), in a transport
         * (UF_LOADED), and a locust ([u+0x20] bit 2 clear: 0x064c on uloc against 0x...e on every other unit
         * surveyed, 25 kinds; the flag word cannot tell a locust from a gryphon, both are 0x20001001). A wisp in
         * an entangled mine also has bit 2 clear, but is loaded (0x191405 / 0x1018); a locust is not. The
         * observer's own hidden or loaded units (in a mine, a Burrow, a building under construction, a transport)
         * are kept, flagged BU_INSIDE (the JSON observation's `inside`). */
        DWORD f20 = *(DWORD *)(u + 0x20);
        int loaded = (unit_flags(u) & UF_LOADED) != 0;
        if (!(f20 & 2) && !loaded)
            return 1;
        inside = (f20 & 1) || loaded;
        if (inside ? !own : !unit_visible(u, o->slot))
            return 1;
    }
    /* a handle for an existing unit creates no game object, so replays keep their object ids (design.md) */
    int handle = handle_of(u);
    BinUnit r = {obs_id(u), *(DWORD *)(u + 0x34), owner, bits_to_f(unit_xy_bits(u, 0)), bits_to_f(unit_xy_bits(u, 4)),
                 dead ? 0 : life, bits_to_f(unit_state_bits(u, 1)), dead ? 0 : bits_to_f(unit_state_bits(u, 2)),
                 bits_to_f(unit_state_bits(u, 3))};
    r.flags = (own ? BU_OWN : 0) | (unit_is_structure(u) ? BU_STRUCTURE : 0) | (unit_is_hero(u) ? BU_HERO : 0) |
              (inside ? BU_INSIDE : 0) | (own && (unit_flags(u) & UF_ILLUSION) ? BU_ILLUSION : 0) |
              (dead ? BU_DEAD : 0);
    r.level = unit_hero_level(u);
    r.order_target = 0xffffffff;
    if (!dead) {
        gather_chain(o, u, &r, handle);
        BYTE *ord = own ? current_order(u) : NULL;
        if (ord) {
            r.order_id = *(DWORD *)(ord + 0x24);
            r.order_target = *(DWORD *)(ord + 0x58);
            r.order_x = *(float *)(ord + 0x48);
            r.order_y = *(float *)(ord + 0x50);
        }
        if (handle)
            obs_combat(&r, handle);
        if (handle && (r.flags & BU_STRUCTURE))
            r.resource = NATIVE(RVA_N_RESOURCEAMOUNT, NativeI_I)(handle);
    }
    if (handle && (r.flags & BU_HERO))
        obs_hero(o, r.unit_id, handle, own);
    obs_inventory(o, u, r.unit_id);
    bb_push(&o->t[T_UNITS], &r, sizeof r);
    return 1;
}

/* the internal slot of a JASS player id, as the Player native (0xa4240) maps it: on a map whose
 * script version is below 0x17ac the neutral ids 12..15 are slots 24..27, otherwise identity.
 * The same version read as player_jass_id, which is this mapping's inverse. */
__declspec(naked) int __cdecl script_version(void) {
    __asm {
        mov eax, dword ptr [g_vm_global]
        mov eax, [eax]
        mov ecx, [eax + 0x30]
        lea ecx, [ecx + 0x24]
        call dword ptr [g_fn_ctx]
        push eax
        call dword ptr [g_fn_ver]
        add esp, 4
        ret
    }
}
int player_slot(int jass_id) {
    if (jass_id >= 12 && jass_id <= 15 && (unsigned)map_script_version() < 0x17ac)
        return jass_id + 12;
    return jass_id;
}

/* the raw function pointers the naked accessors jump through; once, after g_base is known */
void accessors_init(void) {
    g_fn_owner = g_base + RVA_UNIT_OWNER;
    g_fn_state = g_base + RVA_UNIT_STATE;
    g_fn_unpack = g_base + RVA_POS_UNPACK;
    g_vm_global = g_base + RVA_VM_GLOBAL;
    g_fn_ctx = g_base + RVA_VM_STR_CTX;
    g_fn_ver = g_base + RVA_SCRIPT_VERSION;
    g_fn_slot = g_base + RVA_ITEM_IN_SLOT;
    g_fn_resolve_pair = g_base + RVA_RESOLVE_PAIR;
}

/* Everything one observation reports for `player`, into the static tables; fills the header too. The
 * player's unread events are consumed. Game thread. */
static Obs *gather(int player, int seq, BinHeader *h) {
    static Obs o;
    for (int i = 0; i < OBS_TABLES; i++)
        o.t[i].n = 0;
    o.player = player;
    o.slot = player_slot(player);
    ((EnumObjectsFn)(g_base + RVA_ENUM_OBJECTS))(UNIT_CLASS, (void *)obs_unit_cb, &o, 0);
    ((EnumObjectsFn)(g_base + RVA_ENUM_OBJECTS))(ITEM_CLASS, (void *)obs_item_cb, &o, 0);
    ((EnumObjectsFn)(g_base + RVA_ENUM_OBJECTS))(DESTRUCTABLE_CLASS, (void *)obs_destructable_cb, &o, 0);
    memset(h, 0, sizeof *h);
    h->magic = OBS_MAGIC;
    h->version = OBS_VERSION;
    h->player = (DWORD)player;
    h->sequence = (DWORD)seq;
    h->game_time_ms = game_time();
    h->events_lost = events_gather(&o.t[T_EVENTS], player);
    int pj = NATIVE(RVA_N_PLAYER, NativeI_I)(player);
    for (size_t i = 0; i < sizeof UPGRADES / sizeof UPGRADES[0]; i++) {
        BinResearch t = {UPGRADES[i], NATIVE(RVA_N_TECHCOUNT, NativeI_III)(pj, (int)UPGRADES[i], 1)};
        if (t.level > 0)
            bb_push(&o.t[T_RESEARCH], &t, sizeof t);
    }
    h->time_of_day = bits_to_f(NATIVE(RVA_N_FLOATGAMESTATE, NativeI_I)(2)); /* GAME_STATE_TIME_OF_DAY */
    NativeI_II ps = NATIVE(RVA_N_GETPLAYERSTATE, NativeI_II);
    h->gold = ps(pj, 1);
    h->lumber = ps(pj, 2);
    h->food_used = ps(pj, 5);
    h->food_cap = ps(pj, 4);
    const char *result = player_result(player);
    h->result = !strcmp(result, "victory") ? 1 : !strcmp(result, "defeat") ? 2 : !strcmp(result, "draw") ? 3 : 0;
    metadata_scores(player, h->score);
    h->tables = OBS_TABLES;
    DWORD size = sizeof *h;
    for (int i = 0; i < OBS_TABLES; i++) {
        h->table[i] = (BinTable){size, (DWORD)COUNT(&o, i), RECORD_SIZE[i]};
        size += (DWORD)o.t[i].n;
    }
    h->size = size;
    return &o;
}

/* The binary observation (obsbin.h) into dst, at most cap bytes: the header, then each table's records.
 * Returns its size, or 0 when it does not fit. Game thread. */
DWORD obs_write_binary(BYTE *dst, DWORD cap, int player, int seq) {
    BinHeader h;
    Obs *o = gather(player, seq, &h);
    if (h.size > cap)
        return 0;
    memcpy(dst, &h, sizeof h);
    for (int i = 0; i < OBS_TABLES; i++)
        memcpy(dst + h.table[i].offset, o->t[i].p, o->t[i].n);
    return h.size;
}

/* ---- the JSON observation, formatted from the same records ---------------------------------------- */
static void json_order(JW *w, const BinUnit *r) {
    jw_key(w, "order");
    if (!r->order_id) {
        jw_null(w);
        return;
    }
    jw_open(w, '{');
    jw_key(w, "name");
    const char *name = r->order_id == 851970u ? "harvest" : NULL; /* the harvest command's own id (act.c O_HARVEST) */
    for (size_t i = 0; i < sizeof ORDER_NAMES / sizeof ORDER_NAMES[0] && !name; i++)
        if (ORDER_NAMES[i].id == r->order_id)
            name = ORDER_NAMES[i].name;
    if (name)
        jw_string(w, name);
    else if (r->order_id >> 24)
        jw_fourcc(w, r->order_id);
    else
        jw_uint(w, r->order_id);
    jw_key(w, "target_id");
    if (r->order_target != 0xffffffff)
        jw_uint(w, r->order_target);
    else
        jw_null(w);
    jw_key(w, "x");
    jw_num(w, r->order_x);
    jw_key(w, "y");
    jw_num(w, r->order_y);
    jw_close(w, '}');
}
/* where one unit's records end in a table, from `at`: each table lists them contiguously in unit order,
 * and every record there starts with the unit id */
static size_t unit_end(const Obs *o, int table, size_t at, DWORD unit) {
    while (at < COUNT(o, table) && *(const DWORD *)(o->t[table].p + at * RECORD_SIZE[table]) == unit)
        at++;
    return at;
}
static void json_units(JW *own, JW *inside, JW *others, JW *inventory, const Obs *o) {
    size_t buff = 0, ability = 0, queued = 0, item = 0;
    for (size_t i = 0; i < COUNT(o, T_UNITS); i++) {
        const BinUnit *r = &RECORDS(o, T_UNITS, BinUnit)[i];
        size_t b0 = buff, a0 = ability, q0 = queued, v0 = item;
        buff = unit_end(o, T_BUFFS, buff, r->unit_id);
        ability = unit_end(o, T_ABILITIES, ability, r->unit_id);
        queued = unit_end(o, T_QUEUE, queued, r->unit_id);
        item = unit_end(o, T_INVENTORY, item, r->unit_id);
        if (r->flags & BU_DEAD)
            continue; /* binary observations only */
        JW *w = r->flags & BU_INSIDE ? inside : r->flags & BU_OWN ? own : others;
        jw_open(w, '{');
        jw_key(w, "unit_id");
        jw_uint(w, r->unit_id);
        jw_key(w, "type_id");
        jw_fourcc(w, r->type_id);
        if (r->flags & BU_INSIDE) {
            jw_key(w, "x");
            jw_num(w, r->x);
            jw_key(w, "y");
            jw_num(w, r->y);
            jw_key(w, "hp");
            jw_int(w, (int)r->hp);
            json_order(w, r);
            jw_close(w, '}');
            continue;
        }
        jw_key(w, "owner");
        jw_int(w, r->owner);
        jw_key(w, "x");
        jw_num(w, r->x);
        jw_key(w, "y");
        jw_num(w, r->y);
        jw_key(w, "hp");
        jw_int(w, (int)r->hp);
        jw_key(w, "max_hp");
        jw_int(w, (int)r->max_hp);
        jw_key(w, "mana");
        jw_int(w, (int)r->mana);
        jw_key(w, "max_mana");
        jw_int(w, (int)r->max_mana);
        jw_key(w, "structure");
        jw_bool(w, (r->flags & BU_STRUCTURE) != 0);
        jw_key(w, "hero");
        jw_bool(w, (r->flags & BU_HERO) != 0);
        jw_key(w, "level");
        jw_int(w, r->level);
        jw_key(w, "buffs");
        jw_open(w, '[');
        for (size_t k = b0; k < buff; k++)
            jw_fourcc(w, RECORDS(o, T_BUFFS, BinBuff)[k].buff_id);
        jw_close(w, ']');
        if (r->flags & BU_OWN) {
            json_order(w, r);
            if (r->flags & BU_STRUCTURE) {
                jw_key(w, "state");
                if (r->state)
                    jw_string(w, r->state == 1 ? "constructing" : "upgrading");
                else
                    jw_null(w);
                jw_key(w, "state_seconds");
                jw_num(w, r->state_seconds);
                jw_key(w, "queue");
                jw_open(w, '[');
                for (size_t k = q0; k < queued; k++)
                    jw_fourcc(w, RECORDS(o, T_QUEUE, BinQueue)[k].type_id);
                jw_close(w, ']');
                jw_key(w, "queue_seconds");
                jw_num(w, r->queue_seconds);
            }
            jw_key(w, "abilities");
            jw_open(w, '[');
            for (size_t k = a0; k < ability; k++) {
                const BinAbility *a = &RECORDS(o, T_ABILITIES, BinAbility)[k];
                jw_open(w, '{');
                jw_key(w, "ability_id");
                jw_fourcc(w, a->ability_id);
                jw_key(w, "level");
                jw_int(w, a->level);
                jw_key(w, "mana_cost");
                jw_int(w, a->mana_cost);
                jw_key(w, "cooldown_seconds");
                jw_num(w, a->cooldown_seconds);
                jw_key(w, "cooldown_remaining");
                jw_num(w, a->cooldown_remaining);
                jw_close(w, '}');
            }
            jw_close(w, ']');
            for (size_t k = v0; k < item; k++) {
                const BinInventory *v = &RECORDS(o, T_INVENTORY, BinInventory)[k];
                jw_open(inventory, '{');
                jw_key(inventory, "unit_id");
                jw_uint(inventory, v->unit_id);
                jw_key(inventory, "slot");
                jw_int(inventory, v->slot);
                jw_key(inventory, "type_id");
                jw_fourcc(inventory, v->type_id);
                jw_key(inventory, "charges");
                jw_int(inventory, v->charges);
                jw_close(inventory, '}');
            }
        }
        jw_close(w, '}');
    }
}
static void json_list(JW *w, const char *key, const JW *items) {
    jw_key(w, key);
    jw_open(w, '[');
    jw_splice(w, items);
    jw_close(w, ']');
}

/* The observation as one JSON object (docs/specs/observations.md). Game thread. */
void obs_write_json(JW *w, int player, int seq) {
    static JW own, inside, others, inventory;
    BinHeader h;
    const Obs *o = gather(player, seq, &h);
    jw_reset(&own);
    jw_reset(&inside);
    jw_reset(&others);
    jw_reset(&inventory);
    json_units(&own, &inside, &others, &inventory, o);
    jw_open(w, '{');
    jw_key(w, "protocol_version");
    jw_int(w, 1);
    jw_key(w, "observer");
    jw_int(w, player);
    metadata_write(w, player, h.score);
    jw_key(w, "sequence");
    jw_int(w, seq);
    jw_key(w, "game_time_seconds");
    jw_num(w, h.game_time_ms / 1000.0);
    jw_key(w, "player");
    jw_open(w, '{');
    jw_key(w, "gold");
    jw_int(w, h.gold);
    jw_key(w, "lumber");
    jw_int(w, h.lumber);
    jw_key(w, "food_used");
    jw_int(w, h.food_used);
    jw_key(w, "food_cap");
    jw_int(w, h.food_cap);
    jw_close(w, '}');
    json_list(w, "units", &own);
    json_list(w, "inside", &inside);
    json_list(w, "visible_enemies", &others);
    jw_key(w, "items");
    jw_open(w, '[');
    for (size_t k = 0; k < COUNT(o, T_ITEMS); k++) {
        const BinItem *it = &RECORDS(o, T_ITEMS, BinItem)[k];
        jw_open(w, '{');
        jw_key(w, "item_id");
        jw_uint(w, it->item_id);
        jw_key(w, "type_id");
        jw_fourcc(w, it->type_id);
        jw_key(w, "x");
        jw_num(w, it->x);
        jw_key(w, "y");
        jw_num(w, it->y);
        jw_close(w, '}');
    }
    jw_close(w, ']');
    json_list(w, "inventory", &inventory);
    jw_key(w, "events");
    jw_open(w, '[');
    events_json(w, RECORDS(o, T_EVENTS, BinEvent), COUNT(o, T_EVENTS));
    jw_close(w, ']');
    jw_key(w, "events_lost");
    jw_uint(w, h.events_lost);
    jw_key(w, "chat");
    if (NATIVE(RVA_N_PLAYER, NativeI_I)(player) == NATIVE(RVA_N_LOCAL_PLAYER, NativeI_V)())
        chat_write_json(w);
    else {
        jw_open(w, '[');
        jw_close(w, ']');
    }
    jw_key(w, "destructables");
    jw_open(w, '[');
    for (size_t k = 0; k < COUNT(o, T_DESTRUCTABLES); k++) {
        const BinDestructable *d = &RECORDS(o, T_DESTRUCTABLES, BinDestructable)[k];
        jw_open(w, '{');
        jw_key(w, "id");
        jw_uint(w, d->id);
        jw_key(w, "type_id");
        jw_fourcc(w, d->type_id);
        jw_key(w, "x");
        jw_num(w, d->x);
        jw_key(w, "y");
        jw_num(w, d->y);
        jw_key(w, "hp");
        jw_num(w, d->hp);
        jw_key(w, "resource");
        if (d->flags & BD_LUMBER)
            jw_string(w, "lumber");
        else
            jw_null(w);
        jw_key(w, "invulnerable");
        jw_bool(w, (d->flags & BD_INVULNERABLE) != 0);
        jw_close(w, '}');
    }
    jw_close(w, ']');
    jw_key(w, "result");
    jw_string(w, player_result(player));
    jw_close(w, '}');
}

/* ---- the game result, per player ---------------------------------------------------------------
 * Blizzard's melee script ends a player's game with RemovePlayer(player, PLAYER_GAME_RESULT_*)
 * (MeleeDoVictoryEnum, MeleeDoDefeat), right after it sets bj_meleeVictoried / bj_meleeDefeated.
 * The native is detoured: the result is recorded for that player and, for slot 0 (the local
 * player), the RPC's lifecycle moves to `ended`. The step finishes after GameUpdate returns,
 * so both players' results and the final turn are complete before observation. Result handles
 * are the enum values: 0 victory, 1 defeat, 2 tie. */
static char g_result_of[16][8];
void results_clear(void) {
    memset(g_result_of, 0, sizeof g_result_of);
}
const char *player_result(int jass_id) {
    return jass_id >= 0 && jass_id < 16 ? g_result_of[jass_id] : "";
}
int agents_finished(void) {
    int agents = 0;
    for (int p = 0; p < 16; p++)
        if (player_is_agent(p)) {
            agents++;
            if (!g_result_of[p][0])
                return 0;
        }
    return agents != 0;
}
static void result_set(int jass_id, const char *result) {
    if (g_status == ST_LAUNCHED)
        return; /* ignore startup and teardown callbacks */
    if (jass_id < 0 || jass_id >= 16 || g_result_of[jass_id][0])
        return;
    strncpy(g_result_of[jass_id], result, sizeof g_result_of[jass_id] - 1);
    hook_log("result: player %d %s (gametime %lu)", jass_id, result, game_time());
    if (agents_finished()) {
        rpc_set_ended();
        step_end_after_turn();
    }
}
typedef void(__cdecl *RemovePlayerFn)(int player, int result);
typedef BYTE *(__cdecl *PlayerObjectFn)(int player);
static RemovePlayerFn RemovePlayer_orig;
static void __cdecl RemovePlayer_hook(int player, int result) {
    BYTE *pl = ((PlayerObjectFn)(g_base + RVA_PLAYER_OBJECT))(player);
    if (pl && result >= 0 && result <= 2)
        result_set(player_jass_id(pl), result == 0 ? "victory" : result == 1 ? "defeat" : "draw");
    RemovePlayer_orig(player, result);
}
void result_init_hooks(void) {
    MH_STATUS s = MH_CreateHook(g_base + RVA_N_REMOVEPLAYER, (void *)RemovePlayer_hook, (void **)&RemovePlayer_orig);
    hook_log("RemovePlayer hook: %s", MH_StatusToString(s));
}
