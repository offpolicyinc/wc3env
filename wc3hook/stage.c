/* Staging: test state through the game's own natives, called from C on the game thread. These are
 * the `debug` ops of docs/specs/protocol.md (spawn, kill, give, ...): what the JASS bridge's test ops did
 * before the map was retired. The dispatch table also registers the shared
 * readback/display helpers documented in docs/specs/protocol.md.
 *
 * Natives are cdecl, take JASS handles for objects and a pointer to the float for reals, and
 * return an int (a handle, a bool, or float bits). An object's handle comes from the VM the way every
 * native that returns a unit gets one: the VM's handle context, then "handle for object",
 * which allocates on first sight. So a unit no script ever touched still gets a handle here. */
#include "wc3hook.h"

#define RVA_VM_CONTEXT 0x077710 /* thiscall(this = *g_vm_global) -> the handle context */
#define RVA_HANDLE_FOR 0x0e72c0 /* thiscall(ctx; obj, 0) -> handle, allocated if new */
#define RVA_ITEM_OBJECT                                                                                                \
    0x082050 /* cdecl(Hitem) -> item object (the lookup is per class: units 0x83940, players 0x82770) */
#define RVA_N_CREATEUNIT 0x08ee00     /* (Hplayer;IRRR)Hunit; */
#define RVA_N_CREATEITEM 0x08e430     /* (IRR)Hitem; */
#define RVA_N_KILLUNIT 0x0a1ca0       /* (Hunit;)V */
#define RVA_N_REMOVEUNIT 0x0a53e0     /* (Hunit;)V */
#define RVA_N_SETHEROLEVEL 0x0a88b0   /* (Hunit;IB)V */
#define RVA_N_SETTECHRESEARCHED 0x0a9920 /* SetPlayerTechResearched (Hplayer;II)V */
#define RVA_N_ADDITEMBYID 0x0aead0    /* UnitAddItemById (Hunit;I)Hitem; */
#define RVA_N_SETUNITSTATE 0x0abba0   /* (Hunit;Hunitstate;R)V */
#define RVA_N_SETMAXHP 0x0ab7c0       /* BlzSetUnitMaxHP (Hunit;I)V */
#define RVA_N_SETPLAYERSTATE 0x0a97a0 /* (Hplayer;Hplayerstate;I)V */
#define RVA_N_INVULNERABLE 0x0ab600   /* SetUnitInvulnerable (Hunit;B)V */
/* the Convert* natives are the identity, so these enum values are the handles */
#define UNIT_STATE_LIFE 0
#define UNIT_STATE_MAX_LIFE 1
#define UNIT_STATE_MANA 2
#define PLAYER_STATE_GOLD 1
#define PLAYER_STATE_LUMBER 2

typedef int(__cdecl *N1)(int);
typedef int(__cdecl *N2)(int, int);
typedef int(__cdecl *N3)(int, int, int);
typedef int(__cdecl *N5)(int, int, int, int, int);
typedef void *(__fastcall *VmContextFn)(void *vm, void *edx);
typedef int(__fastcall *HandleForFn)(void *ctx, void *edx, BYTE *obj, int zero);
typedef BYTE *(__cdecl *HandleObjectFn)(int handle);

static int real(float *slot, double v) {
    *slot = (float)v;
    return (int)slot;
} /* a real argument: the address of the float */

int handle_of(BYTE *obj) {
    void *vm = *(void **)g_vm_global;
    void *ctx = vm ? ((VmContextFn)(g_base + RVA_VM_CONTEXT))(vm, NULL) : NULL;
    return ctx && obj ? ((HandleForFn)(g_base + RVA_HANDLE_FOR))(ctx, NULL, obj, 0) : 0;
}
static BYTE *unit_of(int handle) {
    return handle ? ((HandleObjectFn)(g_base + RVA_HANDLE_OBJECT))(handle) : NULL;
}
static BYTE *item_of(int handle) {
    return handle ? ((HandleObjectFn)(g_base + RVA_ITEM_OBJECT))(handle) : NULL;
}
static int player_handle(int jass_id) {
    return NATIVE(RVA_N_PLAYER, NativeI_I)(jass_id);
}

/* ---- argument helpers: NULL when present and valid, else the error detail ------------------- */
static const char *arg_unit(yyjson_val *args, int *handle) {
    long long id;
    BYTE *u;
    if (!jr_is_int(yyjson_obj_get(args, "unit_id"), &id))
        return "unit_id must be an integer";
    if (!(u = unit_by_rpc_id(id)) || !(*handle = handle_of(u)))
        return "no unit with that unit_id";
    return NULL;
}
static const char *arg_int(yyjson_val *args, const char *key, long long lo, long long hi, long long *out) {
    if (!jr_is_int(yyjson_obj_get(args, key), out) || *out < lo || *out > hi)
        return "integer argument missing or out of range";
    return NULL;
}
static const char *arg_value(yyjson_val *args, double *out) {
    if (!jr_is_num(yyjson_obj_get(args, "value"), out) || *out < 0 || *out > 2147483647.0)
        return "value must be a number in 0..2147483647";
    return NULL;
}
static const char *arg_xy(yyjson_val *args, double *x, double *y) {
    if (!jr_is_num(yyjson_obj_get(args, "x"), x) || !jr_is_num(yyjson_obj_get(args, "y"), y) || fabs(*x) > 1000000 ||
        fabs(*y) > 1000000)
        return "x and y must be numbers";
    return NULL;
}
static const char *arg_type(yyjson_val *args, unsigned *out) {
    if (!(*out = jr_fourcc(yyjson_obj_get(args, "type_id"))))
        return "type_id must be four characters";
    return NULL;
}
#define NEED(e)                                                                                                        \
    do {                                                                                                               \
        const char *err_ = (e);                                                                                        \
        if (err_)                                                                                                      \
            return err_;                                                                                               \
    } while (0)

/* ---- the ops: each reads its arguments, calls the native, and may add keys to the result ---- */
/* spawn {type_id, player, x, y, n=1, columns=4, spacing=64}: grid facing south; result unit_ids */
static const char *op_spawn(yyjson_val *args, JW *w) {
    unsigned type;
    int pl;
    double x, y, spacing = 64;
    long long n = 1, columns = 4;
    float fx, fy, face;
    NEED(arg_type(args, &type));
    NEED(player_arg(args, &pl));
    NEED(arg_xy(args, &x, &y));
    if (yyjson_obj_get(args, "n") != NULL)
        NEED(arg_int(args, "n", 1, 500, &n));
    if (yyjson_obj_get(args, "columns") != NULL)
        NEED(arg_int(args, "columns", 1, 500, &columns));
    if (yyjson_obj_get(args, "spacing") != NULL &&
        (!jr_is_num(yyjson_obj_get(args, "spacing"), &spacing) || spacing < 0 || spacing > 10000))
        return "spacing must be a number in 0..10000";
    jw_key(w, "unit_ids");
    jw_open(w, '[');
    for (int i = 0; i < n; i++) {
        int h = NATIVE(RVA_N_CREATEUNIT, N5)(pl, (int)type, real(&fx, x + spacing * (i % columns)),
                                             real(&fy, y + spacing * (i / columns)), real(&face, 270.0));
        BYTE *u = unit_of(h);
        if (u)
            jw_uint(w, *(DWORD *)(u + 0xc));
    }
    jw_close(w, ']');
    return NULL;
}
static const char *op_kill(yyjson_val *args, JW *w) {
    int u;
    NEED(arg_unit(args, &u));
    NATIVE(RVA_N_KILLUNIT, N1)(u);
    return NULL;
}
static const char *op_remove(yyjson_val *args, JW *w) {
    int u;
    NEED(arg_unit(args, &u));
    NATIVE(RVA_N_REMOVEUNIT, N1)(u);
    return NULL;
}
/* level {unit_id, level}: SetHeroLevel without the level-up effect */
static const char *op_level(yyjson_val *args, JW *w) {
    int u;
    long long lvl;
    NEED(arg_unit(args, &u));
    NEED(arg_int(args, "level", 1, 10, &lvl));
    NATIVE(RVA_N_SETHEROLEVEL, N3)(u, (int)lvl, 0);
    return NULL;
}
/* research {player, type_id, level}: the player owns that upgrade at that level, as if researched */
static const char *op_research(yyjson_val *args, JW *w) {
    int pl;
    unsigned type;
    long long lvl;
    NEED(player_arg(args, &pl));
    NEED(arg_type(args, &type));
    NEED(arg_int(args, "level", 0, 100, &lvl));
    NATIVE(RVA_N_SETTECHRESEARCHED, N3)(pl, (int)type, (int)lvl);
    return NULL;
}
/* give {unit_id, type_id}: an item into the unit's inventory */
static const char *op_give(yyjson_val *args, JW *w) {
    int u;
    unsigned type;
    NEED(arg_unit(args, &u));
    NEED(arg_type(args, &type));
    return NATIVE(RVA_N_ADDITEMBYID, N2)(u, (int)type) ? NULL : "the unit could not take the item";
}
/* hp {unit_id, value}: the max is raised first when the value is above it (SetUnitState clamps) */
static const char *op_hp(yyjson_val *args, JW *w) {
    int u;
    double v;
    float f;
    NEED(arg_unit(args, &u));
    NEED(arg_value(args, &v));
    BYTE *obj = unit_of(u);
    if (obj && v > bits_to_f(unit_state_bits(obj, UNIT_STATE_MAX_LIFE)))
        NATIVE(RVA_N_SETMAXHP, N2)(u, (int)v);
    NATIVE(RVA_N_SETUNITSTATE, N3)(u, UNIT_STATE_LIFE, real(&f, v));
    return NULL;
}
static const char *op_mana(yyjson_val *args, JW *w) {
    int u;
    double v;
    float f;
    NEED(arg_unit(args, &u));
    NEED(arg_value(args, &v));
    NATIVE(RVA_N_SETUNITSTATE, N3)(u, UNIT_STATE_MANA, real(&f, v));
    return NULL;
}
/* item {type_id, x, y}: an item on the ground; result item_id */
static const char *op_item(yyjson_val *args, JW *w) {
    unsigned type;
    double x, y;
    float fx, fy;
    NEED(arg_type(args, &type));
    NEED(arg_xy(args, &x, &y));
    BYTE *it = item_of(NATIVE(RVA_N_CREATEITEM, N3)((int)type, real(&fx, x), real(&fy, y)));
    if (!it)
        return "the item was not created";
    jw_key(w, "item_id");
    jw_uint(w, *(DWORD *)(it + 0xc));
    return NULL;
}
/* resources {player, gold, lumber} */
static const char *op_resources(yyjson_val *args, JW *w) {
    int pl;
    long long gold, lumber;
    NEED(player_arg(args, &pl));
    NEED(arg_int(args, "gold", 0, 1000000, &gold));
    NEED(arg_int(args, "lumber", 0, 1000000, &lumber));
    NATIVE(RVA_N_SETPLAYERSTATE, N3)(pl, PLAYER_STATE_GOLD, (int)gold);
    NATIVE(RVA_N_SETPLAYERSTATE, N3)(pl, PLAYER_STATE_LUMBER, (int)lumber);
    return NULL;
}
/* ai {player, paused}: PauseCompAI; a paused computer never attacks */
static const char *op_ai(yyjson_val *args, JW *w) {
    int pl;
    long long paused;
    NEED(player_arg(args, &pl));
    NEED(arg_int(args, "paused", 0, 1, &paused));
    NATIVE(RVA_N_PAUSECOMPAI, N2)(pl, (int)paused);
    return NULL;
}
/* Visible test fixtures for destructable resource classification. */
static const char *op_destructable(yyjson_val *args, JW *w) {
    typedef int(__cdecl * CreateFn)(int, int, int, int, int, int);
    unsigned type;
    double x, y;
    float fx, fy, face, scale;
    long long invulnerable = 0;
    NEED(arg_type(args, &type));
    NEED(arg_xy(args, &x, &y));
    if (yyjson_obj_get(args, "invulnerable"))
        NEED(arg_int(args, "invulnerable", 0, 1, &invulnerable));
    int h = NATIVE(0x08de70, CreateFn)((int)type, real(&fx, x), real(&fy, y), real(&face, 0), real(&scale, 1), 0);
    BYTE *d = h ? NATIVE(0x081470, HandleObjectFn)(h) : NULL;
    if (!d)
        return "the destructable was not created";
    NATIVE(0x0a7b60, N2)(h, (int)invulnerable);
    jw_key(w, "id");
    jw_uint(w, *(DWORD *)(d + 0xc));
    return NULL;
}
static const char *op_alliance(yyjson_val *args, JW *w) {
    typedef void(__cdecl * AllianceFn)(int, int, int, int);
    int player;
    long long other, kind, enabled;
    NEED(player_arg(args, &player));
    NEED(arg_int(args, "other", 0, 15, &other));
    NEED(arg_int(args, "kind", 0, 9, &kind));
    NEED(arg_int(args, "enabled", 0, 1, &enabled));
    NATIVE(0x0a94a0, AllianceFn)(player, player_handle((int)other), (int)kind, (int)enabled);
    return NULL;
}
/* order {unit_id}: inspect execution after the command stream has consumed an action. */
static const char *op_order(yyjson_val *args, JW *w) {
    int unit;
    NEED(arg_unit(args, &unit));
    jw_key(w, "order_id");
    jw_int(w, NATIVE(RVA_N_UNITORDER, NativeI_I)(unit));
    return NULL;
}
/* camera {x, y}: point the local player's camera there (SetCameraPosition), for watching a staged scene. */
static const char *op_camera(yyjson_val *args, JW *w) {
    double x, y;
    float fx, fy;
    NEED(arg_xy(args, &x, &y));
    NATIVE(0x0a7260, N2)(real(&fx, x), real(&fy, y));
    return NULL;
}
/* invulnerable {player, on}: every unit the player owns, now and at the end of every step (so
 * units it gets later too), as the whosyourdaddy cheat would. Off stops applying it; units stay. */
static int g_invulnerable_player = -1;
/* handicap {player, hp, damage=1}: the player's units as if their hit points were `hp` of their own and their
 * damage `damage` of it (each in (0, 1]; 1, 1 turns it off): every hit on one of its units is scaled by 1/hp, every
 * hit by one by `damage`, in the damage event's hook before the game applies it (events.c), so it holds for units
 * made later, hero levels and upgrades alike. */
static int g_handicap_player = -1;
static float g_handicap_taken = 1.0f, g_handicap_dealt = 1.0f;
void handicap_damage(BYTE *victim, BYTE *source, float *damage) {
    BYTE *pl;
    if (g_handicap_player < 0 || !damage)
        return;
    if (victim && (pl = unit_owner(victim)) && player_jass_id(pl) == g_handicap_player)
        *damage *= g_handicap_taken;
    if (source && (pl = unit_owner(source)) && player_jass_id(pl) == g_handicap_player)
        *damage *= g_handicap_dealt;
}
static const char *op_handicap(yyjson_val *args, JW *w) {
    long long p;
    double hp, damage;
    NEED(arg_int(args, "player", 0, 15, &p));
    damage = 1.0; /* optional: health alone is the usual handicap */
    if (!jr_is_num(yyjson_obj_get(args, "hp"), &hp) || !(hp > 0 && hp <= 1) ||
        (yyjson_obj_get(args, "damage") && (!jr_is_num(yyjson_obj_get(args, "damage"), &damage) || !(damage > 0 && damage <= 1))))
        return "hp (and damage, if given) must be numbers in (0, 1]";
    g_handicap_player = hp == 1 && damage == 1 ? -1 : (int)p;
    g_handicap_taken = (float)(1.0 / hp);
    g_handicap_dealt = (float)damage;
    return NULL;
}
void stage_clear(void) {
    g_invulnerable_player = -1;
    g_handicap_player = -1;
}
static int __cdecl invulnerable_cb(BYTE *u, void *ctx) {
    BYTE *pl = unit_owner(u);
    if (pl && player_jass_id(pl) == g_invulnerable_player && bits_to_f(unit_state_bits(u, UNIT_STATE_LIFE)) > 0)
        NATIVE(RVA_N_INVULNERABLE, N2)(handle_of(u), 1);
    return 1;
}
void stage_each_step(void) { /* game thread, from step_finish */
    if (g_invulnerable_player < 0)
        return;
    __try {
        ((EnumObjectsFn)(g_base + RVA_ENUM_OBJECTS))(UNIT_CLASS, (void *)invulnerable_cb, NULL, 0);
    } __except (native_exc(GetExceptionInformation())) {
        hook_log("stage: invulnerable faulted");
    }
}
static const char *op_invulnerable(yyjson_val *args, JW *w) {
    long long p, on;
    NEED(arg_int(args, "player", 0, 15, &p));
    NEED(arg_int(args, "on", 0, 1, &on));
    g_invulnerable_player = on ? (int)p : -1;
    if (on)
        stage_each_step();
    return NULL;
}

static const struct {
    const char *name;
    StageOp fn;
} OPS[] = {
    {"ai_difficulty", difficulty_read},
    {"overlay", overlay_show},
    {"spawn", op_spawn},
    {"kill", op_kill},
    {"remove", op_remove},
    {"level", op_level},
    {"research", op_research},
    {"give", op_give},
    {"hp", op_hp},
    {"mana", op_mana},
    {"item", op_item},
    {"resources", op_resources},
    {"ai", op_ai},
    {"invulnerable", op_invulnerable},
    {"handicap", op_handicap},
    {"order", op_order},
    {"camera", op_camera},
    {"alliance", op_alliance},
    {"destructable", op_destructable},
};

/* the op's function, or NULL for a name that is not a staging op */
StageOp stage_op(const char *name) {
    for (size_t i = 0; i < sizeof OPS / sizeof OPS[0]; i++)
        if (strcmp(OPS[i].name, name) == 0)
            return OPS[i].fn;
    return NULL;
}
