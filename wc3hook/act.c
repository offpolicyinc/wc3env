/* Orders through the player command queue: the RPC's actions as W3G records, and the UI-record trace. */
#include "wc3hook.h"
#include "orders_table.h"

volatile LONG g_trace, g_trace_n; /* `trace 1`: log who starts an action buffer */

/* ---- command-queue hunt: CDataStore::PutByte at 0x5ce80 (thiscall, ret 4) is how the game
 * serialises everything, including replay time slots (0x1b09e0 <- turn consumer 0x1a86b0) and,
 * presumably, the UI's outgoing actions. With `trace 1`, a byte that is a W3G action opcode
 * written as the FIRST byte of a store ([this+0x10] == 0) logs the writer's stack: that is the
 * producer, not the replay copy (which lands mid-stream). */
typedef BYTE *(__fastcall *PutByteFn)(BYTE *self, void *edx, BYTE b);
static PutByteFn PutByte_orig;
static BYTE *__fastcall PutByte_hook(BYTE *self, void *edx, BYTE b) {
    if (g_trace && (b == 0x16 || b == 0x11) && g_trace_n < 12) {
        InterlockedIncrement(&g_trace_n);
        hook_log("trace: action opcode %02x at offset %lu of store %p (cap %lu)", b, *(DWORD *)(self + 0x10), self,
                 *(DWORD *)(self + 0xc));
        log_stack_here("  stack");
    }
    return PutByte_orig(self, edx, b);
}

/* ---- the player command queue (design.md: one order path) --------------------------------------
 * A UI action (click, hotkey, selection) becomes a W3G action record and goes through
 * QueueLocalAction at 0x1ae190(store, player): checks the player is the local one and in game,
 * takes the bytes out of a CDataStore, and appends them to the game's outgoing store
 * (game+0x2040) that the network layer sends with the next turn; every client applies them at
 * the same turn. The UI builds the store on its stack: 0x18-byte header (vtable 0xa78c2c,
 * +4 data ptr, +8 base, +0xc capacity, +0x10 size, +0x14 -1) with a 1460-byte inline buffer,
 * ctor 0xd7140(&ptr, &base, &cap), PutByte 0x5ce80 / PutData 0x5d230, dtor 0xd7040. Found by
 * tracing who writes opcode 0x16 first into a fresh store (the `trace` command).
 * Records are queued here and the game thread sends them on its next GameUpdate.
 * Record formats are the replay ones (w3g_format): 16 mode count (idA idB)* = selection,
 * 12 flags orderid ff*8 x y = point order, 13 ... + targetA targetB = target order. */
typedef void(__fastcall *StoreCtorFn)(BYTE *self, void *edx, DWORD *ptr, DWORD *base, DWORD *cap);
typedef BYTE *(__fastcall *PutDataFn)(BYTE *self, void *edx, const void *p, DWORD len);
typedef void(__cdecl *QueueActionFn)(BYTE *store, int player);
static QueueActionFn QueueAction_orig;
/* Grow each player's buffer as needed. Game thread only. */
#define ACT_RECORDS 10
typedef struct {
    BYTE bytes[48], length; /* the longest W3G record, 0x14, is 44 bytes */
} ActRecord;
typedef struct {
    ActRecord *records;
    size_t count, capacity;
} ActQueue;
static ActQueue g_act[16];
static int g_encoding_player;
static DWORD g_sel_prev[16][2]; /* object ids survive removal; never retain unit pointers */

void act_clear(void) {
    for (int p = 0; p < 16; p++)
        free(g_act[p].records);
    memset(g_sel_prev, 0, sizeof g_sel_prev);
    memset(g_act, 0, sizeof g_act);
}
const char *act_configure(unsigned agents) {
    const char *error = players_configure(agents);
    if (error)
        return error;
    act_clear();
    return NULL;
}

static int act_reserve(ActQueue *q) {
    if (q->capacity - q->count >= ACT_RECORDS)
        return 1;
    if (q->capacity > (size_t)-1 / sizeof(ActRecord) / 2)
        return 0;
    size_t capacity = q->capacity ? q->capacity * 2 : ACT_RECORDS;
    ActRecord *records = realloc(q->records, capacity * sizeof *records);
    if (!records)
        return 0;
    q->records = records;
    q->capacity = capacity;
    return 1;
}

static void act_enqueue(const BYTE *bytes, int n) { /* the caller reserves room for the whole order */
    if (n > (int)sizeof ((ActRecord *)0)->bytes)
        abort(); /* a record that does not fit would be sent corrupted */
    ActQueue *q = &g_act[g_encoding_player];
    ActRecord *record = &q->records[q->count++];
    memcpy(record->bytes, bytes, n);
    record->length = (BYTE)n;
}

/* An offline action packet: zero time credit, then wire-player id, u16 length and W3G records.
 * Warcraft's normal receiver applies it and records it in the replay before the next turn. */
typedef void(__fastcall *SendPacketFn)(BYTE *, void *, int, BYTE *, int);
static void act_send_packet(int player, const BYTE *bytes, int len) {
    BYTE st[0x18 + 0x5b4 + 32], payload[1024];
    DWORD *f = (DWORD *)st;
    payload[0] = payload[1] = 0;
    payload[2] = (BYTE)player_wire_id(player);
    payload[3] = (BYTE)len;
    payload[4] = (BYTE)(len >> 8);
    memcpy(payload + 5, bytes, len);
    f[1] = f[2] = f[3] = f[4] = 0;
    f[5] = (DWORD)-1;
    ((StoreCtorFn)(g_base + RVA_STORE_CTOR))(st, NULL, &f[1], &f[2], &f[3]);
    f[0] = (DWORD)(g_base + RVA_STORE_VTABLE);
    ((PutDataFn)(g_base + RVA_PUTDATA))(st, NULL, payload, len + 5);
    ((SendPacketFn)(g_base + RVA_SEND_LOOPBACK))(g_game, NULL, 0x1f, st, 2);
    if (f[3] != (DWORD)-1)
        ((StoreCtorFn)(g_base + RVA_STORE_DTOR))(st, NULL, &f[1], &f[2], &f[3]);
    hook_log("act: packet player=%d wire=%u bytes=%d", player, player_wire_id(player), len);
}

/* One packet path for all offline agents, including the local player. */
void act_flush(void) {
    for (int p = 0; p < 16; p++) {
        ActQueue *q = &g_act[p];
        BYTE bytes[1000];
        int len = 0;
        for (size_t i = 0; i < q->count; i++) {
            ActRecord *record = &q->records[i];
            if (len + record->length > sizeof bytes) {
                act_send_packet(p, bytes, len);
                len = 0;
            }
            memcpy(bytes + len, record->bytes, record->length);
            len += record->length;
        }
        if (len)
            act_send_packet(p, bytes, len);
        q->count = 0;
    }
}

/* with `trace 1`, every record the UI queues is logged in full (the reference for our own) */
static void __cdecl QueueAction_hook(BYTE *store, int player) {
    if (g_trace) {
        BYTE *d = *(BYTE **)(store + 4);
        DWORD n = *(DWORD *)(store + 0x10);
        char b[256];
        int k = 0;
        k += _snprintf(b + k, sizeof b - k, "queued player=%d len=%lu:", player, n);
        for (DWORD i = 0; i < n && i < 48; i++)
            k += _snprintf(b + k, sizeof b - k, " %02x", d[i]);
        hook_log("%s", b);
        pipe_send(b);
    }
    QueueAction_orig(store, player);
}

void act_init_hooks(void) {
    MH_CreateHook(g_base + RVA_QUEUE_ACTION, (void *)QueueAction_hook, (void **)&QueueAction_orig);
    MH_CreateHook(g_base + RVA_PUTBYTE, (void *)PutByte_hook, (void **)&PutByte_orig);
}

/* ---- the RPC's act: protocol actions -> W3G records (the shapes captured from the UI with `trace 1`) ----
 * Runs on the game thread. unit_id and target_id are stable engine IDs from the observation;
 * a unit is known if the enumerator still lists it, controllable if its owner is the player.
 * Every accepted action becomes: deselect the previous selection, select the unit, pick its
 * subgroup, then the order record, queued for the next GameUpdate (the next step) to send:
 * Offline packets are delivered from inside GameUpdate before turn time is supplied. */
#define O_MOVE 851986u
#define O_STOP 851972u
#define O_ATTACK 851983u
#define O_SMART 851971u
#define O_HARVEST 851970u
#define O_USE_SLOT0 852008u
#define O_REVIVE 852039u
#define O_DROP_ITEM 852001u
/* Enumerate until the requested ID is found; no fixed-size snapshot or stale pointers. */
typedef struct {
    long long id;
    BYTE *object;
} FindObject;
static int __cdecl find_cb(BYTE *o, void *ctx) {
    FindObject *find = (FindObject *)ctx;
    if ((long long)obs_id(o) == find->id) {
        find->object = o;
        return 0;
    }
    return 1;
}
static BYTE *find_class(long long id, int cls) {
    FindObject find = {id, NULL};
    ((EnumObjectsFn)(g_base + RVA_ENUM_OBJECTS))(cls, (void *)find_cb, &find, 0);
    return find.object;
}
static BYTE *find_widget(long long id, int *cls) {
    static const int classes[] = {UNIT_CLASS, ITEM_CLASS, DESTRUCTABLE_CLASS};
    if (id <= 0 || id > 0xffffffffLL)
        return NULL;
    for (int i = 0; i < 3; i++) {
        BYTE *o = find_class(id, classes[i]);
        if (o) {
            if (cls)
                *cls = classes[i];
            return o;
        }
    }
    return NULL;
}
BYTE *unit_by_rpc_id(long long id) {
    return find_class(id, UNIT_CLASS);
}
static unsigned order_id(const char *name) {
    if (name[0] >= '0' && name[0] <= '9') { /* an order id: the engine's are 851970 and up, below type ids */
        unsigned long id = strtoul(name, NULL, 10);
        return id >= 851970 && id < 0x1000000 ? (unsigned)id : 0;
    }
    for (size_t i = 0; i < sizeof ORDER_NAMES / sizeof ORDER_NAMES[0]; i++)
        if (_stricmp(ORDER_NAMES[i].name, name) == 0)
            return ORDER_NAMES[i].id;
    return 0;
}
static unsigned fourcc(const char *s) {
    return strlen(s) == 4
               ? ((unsigned)(BYTE)s[0] << 24) | ((unsigned)(BYTE)s[1] << 16) | ((unsigned)(BYTE)s[2] << 8) | (BYTE)s[3]
               : 0;
}
static int put_u16(BYTE *b, int n, unsigned v) {
    b[n] = (BYTE)v;
    b[n + 1] = (BYTE)(v >> 8);
    return n + 2;
}
static int put_u32(BYTE *b, int n, unsigned v) {
    memcpy(b + n, &v, 4);
    return n + 4;
}
static int put_f32(BYTE *b, int n, float v) {
    memcpy(b + n, &v, 4);
    return n + 4;
}
static int put_pair(BYTE *b, int n, BYTE *o) {
    n = put_u32(b, n, *(DWORD *)(o + 0xc));
    return put_u32(b, n, *(DWORD *)(o + 0x10));
}
static int put_none(BYTE *b, int n) {
    memset(b + n, 0xff, 8);
    return n + 8;
}

static void rec_select(BYTE **units, int n, int mode) {
    BYTE b[256];
    int k = 0;
    b[k++] = 0x16;
    b[k++] = (BYTE)mode;
    k = put_u16(b, k, n);
    for (int i = 0; i < n; i++)
        k = put_pair(b, k, units[i]);
    act_enqueue(b, k);
}
static void rec_subgroup(BYTE *u) {
    BYTE a[1] = {0x1a};
    act_enqueue(a, 1);
    BYTE b[16];
    int k = 0;
    b[k++] = 0x19;
    k = put_u32(b, k, *(DWORD *)(u + 0x34));
    k = put_pair(b, k, u);
    act_enqueue(b, k);
}
static void select_unit(BYTE *u) {
    DWORD *previous = g_sel_prev[g_encoding_player];
    if (previous[0]) {
        BYTE b[12] = {0x16, 2, 1, 0};
        memcpy(b + 4, previous, 8);
        act_enqueue(b, sizeof b);
    }
    rec_select(&u, 1, 1);
    rec_subgroup(u);
    memcpy(previous, u + 0xc, 8);
}
static void rec_order(unsigned flags, unsigned order) {
    BYTE b[16];
    int k = 0;
    b[k++] = 0x10;
    k = put_u16(b, k, flags);
    k = put_u32(b, k, order);
    k = put_none(b, k);
    act_enqueue(b, k);
}
static void rec_point(unsigned flags, unsigned order, float x, float y, BYTE *target) {
    BYTE b[32];
    int k = 0;
    b[k++] = 0x12;
    k = put_u16(b, k, flags);
    k = put_u32(b, k, order);
    k = put_none(b, k);
    k = put_f32(b, k, x);
    k = put_f32(b, k, y);
    k = target ? put_pair(b, k, target) : put_none(b, k);
    act_enqueue(b, k);
}
/* An aimed item use is the point/target record with the item where an ability order has none, and the
 * same 0x60 flags the unaimed record carries: without the item the game does not know what to use. */
static void rec_use_item_at(int slot, BYTE *item, unsigned flags, float x, float y, BYTE *target) {
    BYTE b[40];
    int k = 0;
    b[k++] = 0x12;
    k = put_u16(b, k, 0x60 | flags);
    k = put_u32(b, k, O_USE_SLOT0 + slot);
    k = put_pair(b, k, item);
    k = put_f32(b, k, x);
    k = put_f32(b, k, y);
    k = target ? put_pair(b, k, target) : put_none(b, k);
    act_enqueue(b, k);
}
static void rec_use_item(int slot, BYTE *item, unsigned flags) {
    BYTE b[24];
    int k = 0;
    b[k++] = 0x10;
    k = put_u16(b, k, 0x60 | flags);
    k = put_u32(b, k, O_USE_SLOT0 + slot);
    k = put_pair(b, k, item);
    act_enqueue(b, k);
}

/* Give or drop an item: a player's drag of an item onto a unit (a shop buys it) or the ground, the record
 * ladder replays carry: 13 flags 0x44 dropitem ff*8 x y (target or ff*8) item. The unit walks there first. */
static void rec_drop_item(BYTE *item, unsigned flags, float x, float y, BYTE *target) {
    BYTE b[40];
    int k = 0;
    b[k++] = 0x13;
    k = put_u16(b, k, 0x44 | flags);
    k = put_u32(b, k, O_DROP_ITEM);
    k = put_none(b, k);
    k = put_f32(b, k, x);
    k = put_f32(b, k, y);
    k = target ? put_pair(b, k, target) : put_none(b, k);
    k = put_pair(b, k, item);
    act_enqueue(b, k);
}

#define MAX_PENDING 256
/* Shift-queued build sites from earlier batches. The scan of workers' current orders cannot see a
 * queued build, so its site would look free to the next search. A site is kept until its worker gets
 * an unqueued order (which clears its queue), dies, or QUEUED_SITE_MS of game time pass. */
#define MAX_QUEUED_SITES 64
#define QUEUED_SITE_MS 60000
typedef struct {
    PendingSite site;
    long long worker;
    DWORD at;
} QueuedSite;
static QueuedSite g_queued[MAX_QUEUED_SITES];
static int g_n_queued;

static void forget_queued_sites(long long worker) {
    int k = 0;
    for (int i = 0; i < g_n_queued; i++)
        if (g_queued[i].worker != worker)
            g_queued[k++] = g_queued[i];
    g_n_queued = k;
}

/* Append the live queued sites to `pending`, pointing each at its worker as it is now. */
static int add_queued_sites(PendingSite *pending, int n, int max) {
    int k = 0;
    for (int i = 0; i < g_n_queued; i++) {
        BYTE *w = unit_by_rpc_id(g_queued[i].worker);
        if (!w || game_time() - g_queued[i].at >= QUEUED_SITE_MS)
            continue;
        g_queued[k] = g_queued[i];
        g_queued[k].site.worker = w;
        if (n < max)
            pending[n++] = g_queued[k].site;
        k++;
    }
    g_n_queued = k;
    return n;
}

void act_apply(ActJob *job) {
    g_encoding_player = job->player;
    /* Sites taken by workers' current and queued build orders, and by earlier builds in this batch. */
    PendingSite pending[MAX_PENDING];
    int n_pending = -1;
    for (int i = 0; i < job->n; i++) {
        ActItem *a = &job->items[i];
        if (a->reason)
            continue;
        BYTE *u = unit_by_rpc_id(a->unit_id);
        if (!u) {
            a->reason = RJ_UNKNOWN_UNIT;
            continue;
        }
        if (!a->queued)
            forget_queued_sites(a->unit_id);
        BYTE *pl = unit_owner(u);
        if (!pl || player_jass_id(pl) != job->player) {
            a->reason = RJ_NOT_YOURS;
            continue;
        }
        int target_class = 0;
        int revive = strcmp(a->command, "revive") == 0;
        BYTE *t = !a->target ? NULL : revive ? unit_by_rpc_id(a->target) : find_widget(a->target, &target_class);
        /* A dead hero is hidden, so revive matches its target by ownership instead of visibility. */
        if (a->target && (!t || (revive ? unit_owner(t) != pl : !object_visible(t, target_class, job->player)))) {
            a->reason = RJ_BAD_ARGS;
            continue;
        }
        if (!act_reserve(&g_act[job->player])) {
            a->reason = RJ_QUEUE_FULL;
            continue;
        }
        float tx = t ? bits_to_f(unit_xy_bits(t, 0)) : (float)a->x,
              ty = t ? bits_to_f(unit_xy_bits(t, 4)) : (float)a->y;
        const char *c = a->command;
        unsigned flags = a->queued ? 1u : 0u; /* W3G order flag: append (Shift) */
        if (strcmp(c, "move") == 0) {
            if (!a->has_xy) {
                a->reason = RJ_BAD_ARGS;
                continue;
            }
            select_unit(u);
            rec_point(flags, O_MOVE, (float)a->x, (float)a->y, NULL);
        } else if (strcmp(c, "stop") == 0) {
            select_unit(u);
            rec_order(flags, O_STOP);
        } else if (strcmp(c, "attack") == 0) {
            if (!t && !a->has_xy) {
                a->reason = RJ_BAD_ARGS;
                continue;
            }
            select_unit(u);
            rec_point(flags, O_ATTACK, tx, ty, t);
        } else if (strcmp(c, "smart") == 0) {
            if (!t) {
                a->reason = RJ_BAD_ARGS;
                continue;
            }
            select_unit(u);
            rec_point(flags, O_SMART, tx, ty, t);
        } else if (strcmp(c, "harvest") == 0) {
            if (!t && !a->has_xy) {
                a->reason = RJ_BAD_ARGS;
                continue;
            }
            select_unit(u);
            rec_point(flags, O_HARVEST, tx, ty, t);
        } else if (strcmp(c, "build") == 0) {
            unsigned ty4 = fourcc(a->type_id);
            if (!ty4 || (!t && !a->has_xy) || (t && a->auto_place)) {
                a->reason = RJ_BAD_ARGS;
                continue;
            }
            if (n_pending < 0)
                n_pending = add_queued_sites(pending, placement_pending(job->player, pending, MAX_PENDING / 2),
                                             MAX_PENDING * 3 / 4);
            if (a->auto_place) {
                float x = (float)a->x, y = (float)a->y;
                /* A replacing (unqueued) order frees the worker's own earlier site. */
                int found = placement_find(u, ty4, &x, &y, pending, n_pending, a->queued ? NULL : u);
                if (found != 1) {
                    a->reason = found < 0 ? RJ_QUEUE_FULL : RJ_NO_BUILD_SITE;
                    continue;
                }
                a->x = x;
                a->y = y;
            }
            PendingSite site = {ty4, t ? tx : (float)a->x, t ? ty : (float)a->y, u};
            if (n_pending < MAX_PENDING)
                pending[n_pending++] = site;
            if (a->queued && g_n_queued < MAX_QUEUED_SITES)
                g_queued[g_n_queued++] = (QueuedSite){site, a->unit_id, game_time()};
            select_unit(u);
            /* Stock AI uses a target order for Haunted Mines (0x6c2d92), not its ground site. */
            rec_point(flags, ty4, t ? tx : (float)a->x, t ? ty : (float)a->y, t);
        } else if (strcmp(c, "train") == 0 || strcmp(c, "research") == 0) {
            unsigned ty4 = fourcc(a->type_id);
            if (!ty4) {
                a->reason = RJ_BAD_ARGS;
                continue;
            }
            select_unit(u);
            rec_order(0, ty4);
        } else if (strcmp(c, "learn") == 0) {
            unsigned ab = fourcc(a->type_id);
            if (!ab) {
                a->reason = RJ_BAD_ARGS;
                continue;
            }
            select_unit(u);
            rec_order(0x42, ab);
        } else if (strcmp(c, "cast") == 0) {
            unsigned o = order_id(a->order);
            if (!o) {
                a->reason = RJ_BAD_ARGS;
                continue;
            }
            select_unit(u);
            if (t || a->has_xy)
                rec_point(flags, o, tx, ty, t);
            else
                rec_order(flags, o);
        } else if (strcmp(c, "use_item") == 0) {
            if (a->slot < 0 || a->slot > 5) {
                a->reason = RJ_BAD_ARGS;
                continue;
            }
            BYTE *it = unit_item_in_slot(u, a->slot);
            if (!it) {
                a->reason = RJ_BAD_ARGS;
                continue;
            }
            select_unit(u);
            if (t || a->has_xy)
                rec_use_item_at(a->slot, it, flags, tx, ty, t);
            else
                rec_use_item(a->slot, it, flags);
        } else if (strcmp(c, "drop_item") == 0) {
            BYTE *it = a->slot < 0 || a->slot > 5 ? NULL : unit_item_in_slot(u, a->slot);
            if (!it) {
                a->reason = RJ_BAD_ARGS;
                continue;
            }
            if (!t && !a->has_xy) { /* at the unit's feet */
                tx = bits_to_f(unit_xy_bits(u, 0));
                ty = bits_to_f(unit_xy_bits(u, 4));
            }
            select_unit(u);
            rec_drop_item(it, flags, tx, ty, t);
        } else if (revive) {
            /* The altar's target order on its own dead hero. Warcraft refuses it until the death
             * has resolved (about 20 seconds) and when the player cannot pay. */
            if (!t) {
                a->reason = RJ_BAD_ARGS;
                continue;
            }
            select_unit(u);
            rec_point(0, O_REVIVE, tx, ty, t);
        } else if (strcmp(c, "select") == 0) {
            select_unit(u);
        } else if (strcmp(c, "buy") == 0) {
            BYTE *shop = unit_by_rpc_id(a->shop);
            unsigned it4 = fourcc(a->item_type);
            if (!shop || !it4 || !object_visible(shop, UNIT_CLASS, job->player)) {
                a->reason = RJ_BAD_ARGS;
                continue;
            }
            select_unit(u); /* the shop sells to the player's selected unit in range */
            select_unit(shop);
            rec_order(0, it4); /* then the shop gets the item type like a train order */
        } else
            a->reason = RJ_UNKNOWN_COMMAND;
    }
}

/* game thread: the object the RPC id names, or NULL */
BYTE *object_by_rpc_id(long long id) {
    return find_widget(id, NULL);
}
