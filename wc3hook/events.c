/* Game events from C: detours on the game's per-kind event fire functions.
 *
 * Every playerunitevent the observation reports (EVENT_PLAYER_UNIT_DEATH and the rest of common.j's
 * ConvertPlayerUnitEvent ids) has a fire function (thiscall: this = the player, then the event's objects and
 * values). It checks 0x43f0c0 for a registered trigger and only then builds an event object and dispatches
 * it. That object is an engine agent: building one reorders the engine's reused object ids, and a replay's
 * recorded orders name units by those ids. So nothing is registered on the game's behalf: each fire function
 * is detoured at entry and the event read from its arguments, which are what the event object would hold
 * (+0x38 the trigger unit, +0x44 a second unit, +0x50 an item or a type id; read from each function's stores).
 * Events enter each observer's log only if visible when fired (own unit, or in that player's fog). Separate
 * bounded logs keep hidden activity out of both retention and loss counts. `observe(p)` consumes only that
 * player's events. Written as JSON objects (docs/specs/observations.md): object ids as integers, type ids as
 * four-character strings.
 *
 * Orders: every order a player's command gives a unit, and every order a script or the engine itself gives one.
 * See order_record.
 */
#include "wc3hook.h"

/* Per-observer rings, written and read on the game thread only. */
#define EV_CAP 1024
typedef struct {
    int id, owner;
    unsigned unit_id, other_id, item_id, seen, other_seen, item_seen;
    DWORD type, other_type, item_type, argument;
} Event;
static Event g_ev[16][EV_CAP];
static unsigned g_ev_head[16]; /* visible events ever logged for each player */
static unsigned g_cursor[16];  /* per player: the first event it has not been given yet */
static unsigned g_players;     /* bit p: player p is in the game (create_game), so events are judged for it */

/* Per-owner order rings, the same way: a unit's orders are its owner's alone. */
#define ORD_CAP 2048
static BinOrder g_ord[16][ORD_CAP];
static unsigned g_ord_head[16], g_ord_cursor[16];

void events_set_players(unsigned mask) {
    g_players = mask;
    memset(g_ev_head, 0, sizeof g_ev_head);
    memset(g_cursor, 0, sizeof g_cursor);
    memset(g_ord_head, 0, sizeof g_ord_head);
    memset(g_ord_cursor, 0, sizeof g_ord_cursor);
}

static int owner_id(BYTE *unit) {
    BYTE *pl = unit_owner(unit);
    return pl ? player_jass_id(pl) : -1;
}
/* who can see an event on this object, right now: `owner`, and every player whose fog shows it */
static unsigned seen_by(BYTE *object, int cls, int owner) {
    unsigned mask = 0;
    for (int p = 0; p < 16; p++)
        if ((g_players >> p & 1) && (p == owner || object_visible(object, cls, p)))
            mask |= 1u << p;
    return mask;
}

/* One event: the trigger unit, optionally a second unit and an item, and a kind-specific value. */
static void ev_record(int id, BYTE *unit, BYTE *other, BYTE *item, DWORD argument) {
    if (!g_players || !unit)
        return;
    Event e = {id, owner_id(unit), obs_id(unit)};
    e.type = *(DWORD *)(unit + 0x34);
    e.seen = seen_by(unit, UNIT_CLASS, e.owner);
    e.argument = argument;
    if (other) {
        e.other_id = obs_id(other);
        e.other_type = *(DWORD *)(other + 0x34);
        e.other_seen = seen_by(other, UNIT_CLASS, owner_id(other));
    }
    if (item) {
        e.item_id = obs_id(item);
        e.item_type = *(DWORD *)(item + 0x34);
        e.item_seen = seen_by(item, ITEM_CLASS, e.owner);
    }
    for (int p = 0; p < 16; p++)
        if (e.seen & (1u << p))
            g_ev[p][g_ev_head[p]++ % EV_CAP] = e;
}

/* Orders come from two places.
 *
 * A player's command: every order action (the network, a replay, act) gives each unit its order through
 * 0x2c9360 (cdecl: the unit, the command's order object, the command's W3G flags, a fourth value; its only
 * callers are the six order-action handlers). It runs whether the engine then issues the order at once, queues
 * it (Shift), issues it later from its own update (a ladder replay's worker right-clicks often go that way) or
 * refuses it (no mana), so its detour records what the player commanded.
 *
 * Everything else: the unit-side functions that fire EVENT_PLAYER_UNIT_ISSUED_* (thiscall: the unit, the
 * order object; ret 4) see every order a unit is actually given. Where it came from is read from the call
 * chain (the exe keeps frame pointers): orders issued inside 0x2c9360 are the player's command, already
 * recorded; JASS natives run inside the interpreter loop 0x4d75b0 (AI scripts, map triggers); anything else is
 * the engine's own (a worker returning its load, a unit acquiring a target, a held order). The innermost wins:
 * a trigger ordering a unit while a player's command is handled is the script's.
 */
#define RVA_PLAYER_ORDER 0x2c9360
#define RVA_PLAYER_ORDER_END 0x2c95a0
#define RVA_JASS_RUN 0x4d75b0
#define RVA_JASS_RUN_END 0x4d84a0
static DWORD order_origin(void) {
    NT_TIB *tib = (NT_TIB *)NtCurrentTeb();
    DWORD *fp;
    __asm { mov fp, ebp }
    while ((BYTE *)fp >= (BYTE *)tib->StackLimit && (BYTE *)(fp + 2) <= (BYTE *)tib->StackBase) {
        DWORD rva = fp[1] - (DWORD)g_base;
        if (rva >= RVA_PLAYER_ORDER && rva < RVA_PLAYER_ORDER_END)
            return ORIGIN_PLAYER;
        if (rva >= RVA_JASS_RUN && rva < RVA_JASS_RUN_END)
            return ORIGIN_SCRIPT;
        if ((DWORD *)fp[0] <= fp)
            break;
        fp = (DWORD *)fp[0];
    }
    return ORIGIN_ENGINE;
}

static void order_record(BYTE *unit, BYTE *order, DWORD kind, DWORD origin, DWORD queued) {
    int owner = unit && order ? owner_id(unit) : -1;
    if (owner < 0 || owner >= 16 || !(g_players >> owner & 1))
        return;
    BinOrder r = {obs_id(unit), *(DWORD *)(order + 0x24), kind, 0xffffffff};
    if (kind != ORDER_IMMEDIATE) {
        r.x = *(float *)(order + 0x48);
        r.y = *(float *)(order + 0x50);
    }
    if (kind == ORDER_TARGET)
        r.target_id = *(DWORD *)(order + 0x58);
    if (*(DWORD *)order - (DWORD)g_base == RVA_ITEM_ORDER_VTABLE) {
        BYTE *item = object_by_rpc_id(*(DWORD *)(order + 0x88));
        r.item_type = item ? *(DWORD *)(item + 0x34) : 0;
    }
    r.origin = origin;
    r.queued = queued;
    r.time_ms = game_time();
    g_ord[owner][g_ord_head[owner]++ % ORD_CAP] = r;
}
static void issued_record(BYTE *unit, BYTE *order, DWORD kind) {
    DWORD origin = order_origin();
    if (origin != ORIGIN_PLAYER)
        order_record(unit, order, kind, origin, 0);
}
static int(__cdecl *PlayerOrder_orig)(BYTE *unit, BYTE *order, DWORD flags, DWORD fourth);
static int __cdecl PlayerOrder_hook(BYTE *unit, BYTE *order, DWORD flags, DWORD fourth) {
    __try {
        DWORD target = order ? *(DWORD *)(order + 0x58) : 0;
        DWORD kind = order && *(DWORD *)order - (DWORD)g_base == RVA_IMMEDIATE_ORDER_VTABLE ? ORDER_IMMEDIATE
                     : target && target != 0xffffffff                                      ? ORDER_TARGET
                                                                                           : ORDER_POINT;
        order_record(unit, order, kind, ORIGIN_PLAYER, flags & 1); /* W3G order flag 1: Shift */
    } __except (EXCEPTION_EXECUTE_HANDLER) {
    }
    return PlayerOrder_orig(unit, order, flags, fourth);
}

/* The kinds: name, fire function, its stack arguments a, b, c (thiscall: the callee pops them, so the count
 * must match its `ret`), and what is recorded from them. The issued-order functions are unit-side (this = the
 * unit, a the order object): their player-unit fire functions are called only when a trigger is registered. */
#define OBJ(x) ((BYTE *)(x))
#define FIRE_KINDS(X1, X2, X3)                                                                                       \
    X1(ConstructStart, 0x0ba2b0, ev_record(26, OBJ(a), NULL, NULL, 0))                                             \
    X1(ConstructCancel, 0x0ba070, ev_record(27, OBJ(a), NULL, NULL, 0))                                            \
    X1(ConstructFinish, 0x0ba190, ev_record(28, OBJ(a), NULL, NULL, 0))                                            \
    X1(HeroLevel, 0x0b95f0, ev_record(41, OBJ(a), NULL, NULL, unit_hero_level(OBJ(a))))                            \
    X2(Attacked, 0x0b9df0, ev_record(18, OBJ(b), OBJ(a), NULL, 0))       /* a the attacker */                     \
    X2(Death, 0x0ba3d0, ev_record(20, OBJ(a), NULL, NULL, 0))            /* b the killer */                       \
    X2(UpgradeStart, 0x0bc9a0, ev_record(29, OBJ(a), NULL, NULL, b))                                               \
    X2(UpgradeCancel, 0x0bc740, ev_record(30, OBJ(a), NULL, NULL, b))                                              \
    X2(UpgradeFinish, 0x0bc870, ev_record(31, OBJ(a), NULL, NULL, b))                                              \
    X2(TrainStart, 0x0bc610, ev_record(32, OBJ(a), NULL, NULL, b))       /* b the type id */                      \
    X2(TrainCancel, 0x0bc380, ev_record(33, OBJ(a), NULL, NULL, b))                                                \
    X2(TrainFinish, 0x0bc4b0, ev_record(34, OBJ(a), OBJ(b), NULL, 0))    /* b the trained unit */                 \
    X2(ResearchStart, 0x0bb580, ev_record(35, OBJ(a), NULL, NULL, b))                                              \
    X2(ResearchCancel, 0x0bb340, ev_record(36, OBJ(a), NULL, NULL, b))                                             \
    X2(ResearchFinish, 0x0bb460, ev_record(37, OBJ(a), NULL, NULL, b))                                             \
    X2(Summon, 0x0bc0f0, ev_record(47, OBJ(b), OBJ(a), NULL, 0))         /* a the summoner, b the summoned */     \
    X2(ItemPickup, 0x0baf50, ev_record(49, OBJ(a), NULL, OBJ(b), 0))                                               \
    X2(ItemUse, 0x0bcad0, ev_record(50, OBJ(a), NULL, OBJ(b), 0))                                                  \
    X3(HeroLearn, 0x0b9b90, ev_record(42, OBJ(a), NULL, NULL, b))        /* b the ability id */                   \
    X3(ItemSold, 0x0bb960, ev_record(274, OBJ(a), OBJ(b), OBJ(c), 0))    /* a the shop, b the buyer */            \
    X3(UnitSold, 0x0bb7c0, ev_record(272, OBJ(a), OBJ(c), NULL, b ? *(DWORD *)(OBJ(b) + 0x34) : 0)) /* b sold */ \
    X1(IssuedOrder, 0x28b8b0, issued_record(OBJ(pl), OBJ(a), ORDER_IMMEDIATE))                                     \
    X1(IssuedPoint, 0x28be70, issued_record(OBJ(pl), OBJ(a), ORDER_POINT))                                         \
    X1(IssuedTarget, 0x28d270, issued_record(OBJ(pl), OBJ(a), ORDER_TARGET))

#define DETOUR(name, params, args, record)                                                                           \
    static int(__fastcall *name##_orig) params;                                                                     \
    static int __fastcall name##_hook params {                                                                      \
        __try {                                                                                                      \
            record;                                                                                                  \
        } __except (EXCEPTION_EXECUTE_HANDLER) {                                                                     \
        }                                                                                                            \
        return name##_orig args;                                                                                     \
    }
#define DETOUR1(name, rva, record) DETOUR(name, (void *pl, void *edx, DWORD a), (pl, edx, a), record)
#define DETOUR2(name, rva, record) DETOUR(name, (void *pl, void *edx, DWORD a, DWORD b), (pl, edx, a, b), record)
#define DETOUR3(name, rva, record)                                                                                   \
    DETOUR(name, (void *pl, void *edx, DWORD a, DWORD b, DWORD c), (pl, edx, a, b, c), record)
FIRE_KINDS(DETOUR1, DETOUR2, DETOUR3)

/* Spell effects: the unit-side function 0x28ccd0 (thiscall: this = the casting unit, arg the ability, ret 4)
 * calls the player-unit fire function (0xbbd60) only if a trigger is registered for either kind, so the detour
 * sits on it. The ability object's id is at +0x34, where GetSpellAbilityId reads it. */
#define RVA_SPELL_EFFECT 0x28ccd0
static int(__fastcall *SpellEffect_orig)(BYTE *unit, void *edx, BYTE *ability);
static int __fastcall SpellEffect_hook(BYTE *unit, void *edx, BYTE *ability) {
    __try {
        ev_record(277, unit, NULL, NULL, ability ? *(DWORD *)(ability + 0x34) : 0);
    } __except (EXCEPTION_EXECUTE_HANDLER) {
    }
    return SpellEffect_orig(unit, edx, ability);
}

/* An event as its observer may see it (obsbin.h BinEvent). A hidden secondary unit or item reads 0. */
static int event_record(const Event *e, int player, BinEvent *r) {
    unsigned bit = 1u << player, other = (e->other_seen & bit) ? e->other_id : 0;
    int item_seen = (e->item_seen & bit) != 0;
    *r = (BinEvent){(DWORD)e->id, e->unit_id};
    switch (e->id) {
    case 20: /* death */
        r->type_id = e->type;
        r->value = e->owner;
        break;
    case 26: case 27: case 28: case 29: case 30: case 31: /* construct_* and upgrade_*: the structure's type */
        r->type_id = e->type;
        break;
    case 32: case 33: case 35: case 36: case 37: case 42: case 277: /* train, research, hero_learn, spell_effect */
        r->type_id = e->argument;
        break;
    case 34: /* train_finish */
        r->other_id = other;
        r->type_id = other ? e->other_type : 0;
        break;
    case 41: /* hero_level */
        r->value = (int)e->argument;
        break;
    case 47: /* summon: the event's unit is the summoned one; the record names the summoner first */
        r->unit_id = other;
        r->other_id = e->unit_id;
        r->type_id = e->type;
        break;
    case 49: /* item_pickup */
        r->other_id = item_seen ? e->item_id : 0;
        r->type_id = item_seen ? e->item_type : 0;
        break;
    case 50: /* item_use */
        r->type_id = item_seen ? e->item_type : 0;
        break;
    case 18: /* attacked */
        r->other_id = other;
        break;
    case 272: /* unit_sold: a mercenary or tavern hero; the record names the buyer and the sold unit's type */
        r->other_id = other;
        r->type_id = e->argument;
        break;
    case 274: /* item_sold */
        r->other_id = other;
        r->type_id = item_seen ? e->item_type : 0;
        break;
    default:
        return 0;
    }
    return 1;
}

/* The events `player` could see since its previous observation, oldest first, as records; returns the
 * number lost to ring overflow. */
unsigned events_gather(BB *out, int player) {
    unsigned from = g_cursor[player], head = g_ev_head[player];
    unsigned lost = head - from > EV_CAP ? head - from - EV_CAP : 0;
    if (lost)
        from = head - EV_CAP;
    for (unsigned k = from; k != head; k++) {
        BinEvent r;
        if (event_record(&g_ev[player][k % EV_CAP], player, &r))
            bb_push(out, &r, sizeof r);
    }
    g_cursor[player] = head;
    return lost;
}

/* The orders given to `player`'s units since its previous observation, oldest first; returns the number lost. */
unsigned orders_gather(BB *out, int player) {
    unsigned from = g_ord_cursor[player], head = g_ord_head[player];
    unsigned lost = head - from > ORD_CAP ? head - from - ORD_CAP : 0;
    if (lost)
        from = head - ORD_CAP;
    for (unsigned k = from; k != head; k++)
        bb_push(out, &g_ord[player][k % ORD_CAP], sizeof(BinOrder));
    g_ord_cursor[player] = head;
    return lost;
}

/* The JSON names of each kind's fields, written in this order after unit_id; NULL when it has none. */
static const struct {
    DWORD id;
    const char *kind, *other, *type, *value;
} KINDS[] = {
    {18, "attacked", "attacker_id", NULL, NULL},
    {20, "death", NULL, "type_id", "owner"},
    {26, "construct_start", NULL, "type_id", NULL},
    {27, "construct_cancel", NULL, "type_id", NULL},
    {28, "construct_finish", NULL, "type_id", NULL},
    {29, "upgrade_start", NULL, "type_id", NULL},
    {30, "upgrade_cancel", NULL, "type_id", NULL},
    {31, "upgrade_finish", NULL, "type_id", NULL},
    {32, "train_start", NULL, "type_id", NULL},
    {33, "train_cancel", NULL, "type_id", NULL},
    {34, "train_finish", "trained_id", "type_id", NULL},
    {35, "research_start", NULL, "type_id", NULL},
    {36, "research_cancel", NULL, "type_id", NULL},
    {37, "research_finish", NULL, "type_id", NULL},
    {41, "hero_level", NULL, NULL, "level"},
    {42, "hero_learn", NULL, "ability_id", NULL},
    {47, "summon", "summoned_id", "type_id", NULL},
    {49, "item_pickup", "item_id", "type_id", NULL},
    {50, "item_use", NULL, "type_id", NULL},
    {272, "unit_sold", "buyer_id", "type_id", NULL},
    {274, "item_sold", "buyer_id", "type_id", NULL},
    {277, "spell_effect", NULL, "ability_id", NULL},
};
void events_json(JW *w, const BinEvent *events, size_t n) {
    for (size_t i = 0; i < n; i++) {
        const BinEvent *e = &events[i];
        size_t k = 0;
        while (KINDS[k].id != e->kind)
            k++; /* events_gather records only these kinds */
        jw_open(w, '{');
        jw_key(w, "kind");
        jw_string(w, KINDS[k].kind);
        jw_key(w, "unit_id");
        jw_uint(w, e->unit_id);
        if (KINDS[k].other) {
            jw_key(w, KINDS[k].other);
            jw_uint(w, e->other_id);
        }
        if (KINDS[k].type) {
            jw_key(w, KINDS[k].type);
            jw_fourcc(w, e->type_id);
        }
        if (KINDS[k].value) {
            jw_key(w, KINDS[k].value);
            jw_int(w, e->value);
        }
        jw_close(w, '}');
    }
}

void events_init_hooks(void) {
#define INSTALL(name, rva, record) MH_CreateHook(g_base + (rva), (void *)name##_hook, (void **)&name##_orig),
    MH_STATUS s[] = {FIRE_KINDS(INSTALL, INSTALL, INSTALL) MH_CreateHook(
        g_base + RVA_SPELL_EFFECT, (void *)SpellEffect_hook, (void **)&SpellEffect_orig),
                     MH_CreateHook(g_base + RVA_PLAYER_ORDER, (void *)PlayerOrder_hook, (void **)&PlayerOrder_orig)};
    int ok = 0;
    for (size_t i = 0; i < sizeof s / sizeof s[0]; i++)
        ok += s[i] == MH_OK;
    hook_log("event hooks: %d of %u", ok, (unsigned)(sizeof s / sizeof s[0]));
}
