/* The masks of one observation (features.h's FeatOut): training/policy/masks.py's Own and its legality checks, and
 * decide.tables built from them, in C. The same rule as there: allow what might work, forbid only what a player
 * can see is impossible. Sets are compared as sets, so nothing here depends on iteration order. */
#include "features_obs.h"

#define INCOME_MARGIN 30
#define RESUME_HARVESTING 852017u
#define HARVEST 852018u
#define ORDER_WORDS ((FT_ORDER_COUNT + 31) / 32)
#define MAX_OWN 1024

enum { K_NOOP, K_MOVE, K_RALLY, K_CANCEL, K_BUY, K_REVIVE, K_TRAIN, K_RESEARCH, K_BUILD, K_LEARN, K_CAST, K_ITEM };

static int kind_of(const char *name) {
    static const struct { const char *name; int kind; } KINDS[] = {
        {"noop", K_NOOP}, {"move", K_MOVE}, {"attack_move", K_MOVE}, {"smart", K_MOVE}, {"attack", K_MOVE},
        {"stop", K_MOVE}, {"hold", K_MOVE}, {"train", K_TRAIN}, {"build", K_BUILD}, {"research", K_RESEARCH},
        {"rally", K_RALLY}, {"cast", K_CAST}, {"cast_point", K_CAST}, {"cast_unit", K_CAST}, {"learn", K_LEARN},
        {"use_item", K_ITEM}, {"buy", K_BUY}, {"sell", K_ITEM}, {"drop_item", K_ITEM}, {"revive", K_REVIVE},
        {"cancel", K_CANCEL},
    };
    for (size_t i = 0; i < sizeof(KINDS) / sizeof(KINDS[0]); i++)
        if (!strcmp(KINDS[i].name, name))
            return KINDS[i].kind;
    abort(); /* an action type this file does not know: tools/feature_tables.py and this list disagree */
}

static const FtUnit *unit_info(DWORD id) {
    int lo = 0, hi = FT_UNITS_N - 1;
    while (lo <= hi) {
        int mid = (lo + hi) / 2;
        if (FT_UNITS[mid].id == id)
            return &FT_UNITS[mid];
        if (FT_UNITS[mid].id < id)
            lo = mid + 1;
        else
            hi = mid - 1;
    }
    return NULL;
}
#define FIND(table, n, key)                                                                                           \
    do {                                                                                                              \
        int lo = 0, hi = (n)-1;                                                                                       \
        found = NULL;                                                                                                 \
        while (lo <= hi) {                                                                                            \
            int mid = (lo + hi) / 2;                                                                                  \
            if ((table)[mid].id == (key)) {                                                                           \
                found = &(table)[mid];                                                                                \
                break;                                                                                                \
            }                                                                                                         \
            if ((table)[mid].id < (key))                                                                              \
                lo = mid + 1;                                                                                         \
            else                                                                                                      \
                hi = mid - 1;                                                                                         \
        }                                                                                                             \
    } while (0)
static const FtUpgrade *upgrade_info(DWORD id) {
    const FtUpgrade *found;
    FIND(FT_UPGRADE_INFO, FT_UPGRADE_INFO_N, id);
    return found;
}
static const FtAbility *ability_info(DWORD id) {
    const FtAbility *found;
    FIND(FT_ABILITY_INFO, FT_ABILITY_INFO_N, id);
    return found;
}
static const FtTypeOrders *type_orders(DWORD id) {
    const FtTypeOrders *found;
    FIND(FT_TYPE_ORDERS, FT_TYPE_ORDERS_N, id);
    return found;
}
#define ARGUMENT(raw) lookup(FT_ARGUMENTS, FT_ARGUMENTS_N, (raw), -1)

/* one of the observer's living units, with what masks need to know about it */
typedef struct {
    const BinUnit *u;
    const FtUnit *info;
    DWORD orders[ORDER_WORDS]; /* castable orders in FT_ORDER_FIRST.. */
    int castable, cast_other;  /* any castable order; any outside that range */
    Map learnable;             /* ability codes it can learn now */
    Map items;                 /* item types it carries */
} OwnUnit;

typedef struct {
    const Obs *o;
    OwnUnit units[MAX_OWN];
    int n;
    Map index;      /* unit id -> units[] */
    Map have_soon;  /* structure types done (or nearly), what they were, and what upgrading ones become */
    Map research;   /* upgrade -> level */
    Map makes;      /* types some done structure trains, researches or upgrades into */
    Map builds;     /* structure types some worker builds */
    int food_free, hero_count, dead_heroes, structures, mobile, cancel;
    const BinUnit *dead[64];
} Own;

static int nearly_built(const BinUnit *u) {
    return u->max_hp > 0 && ((double)u->hp / (double)u->max_hp - 0.1) / 0.9 >= 0.9;
}

static void own_init(Own *w, const Obs *o) {
    memset(w, 0, sizeof(*w));
    w->o = o;
    map_init(&w->index, 2048), map_init(&w->have_soon, 64), map_init(&w->research, 128), map_init(&w->makes, 128);
    map_init(&w->builds, 64);
    Map done;
    map_init(&done, 64);
    for (DWORD i = 0; i < o->n[T_UNITS]; i++) {
        const BinUnit *u = &o->units[i];
        if (!(u->flags & BU_OWN))
            continue;
        if (u->flags & BU_DEAD) {
            if ((u->flags & BU_HERO) && w->dead_heroes < 64)
                w->dead[w->dead_heroes++] = u;
            continue;
        }
        if (w->n == MAX_OWN)
            continue;
        OwnUnit *ou = &w->units[w->n];
        ou->u = u, ou->info = unit_info(u->type_id);
        map_init(&ou->learnable, 16), map_init(&ou->items, 16);
        map_put(&w->index, u->unit_id, w->n++);
        if ((u->flags & BU_HERO) && !(u->flags & BU_ILLUSION))
            w->hero_count++;
        if (!(u->flags & (BU_STRUCTURE | BU_INSIDE)))
            w->mobile++;
        if (u->flags & BU_STRUCTURE) {
            w->structures++;
            if (u->state != 0)
                w->cancel = 1;
            int built = u->state != 1 || nearly_built(u);
            if (built)
                map_put(&done, u->type_id, 0);
            if (built && ou->info) {
                const FtList *lists[3] = {&ou->info->trains, &ou->info->researches, &ou->info->upgrades_to};
                for (int l = 0; l < 3; l++)
                    for (int k = 0; k < lists[l]->count; k++)
                        map_put(&w->makes, (DWORD)FT_POOL[lists[l]->offset + k], 0);
            }
            if (u->state == 2 && ou->info) /* upgrading: what it becomes counts (it may finish within the step) */
                for (int k = 0; k < ou->info->upgrades_to.count; k++)
                    map_put(&w->have_soon, (DWORD)FT_POOL[ou->info->upgrades_to.offset + k], 0);
        }
        if (ou->info)
            for (int k = 0; k < ou->info->builds.count; k++)
                map_put(&w->builds, (DWORD)FT_POOL[ou->info->builds.offset + k], 0);
    }
    w->hero_count += w->dead_heroes;
    w->cancel = w->cancel || o->n[T_QUEUE] > 0;
    /* a Keep also counts as the Town Hall it was: add whatever upgrades into a type already had, until none */
    for (size_t i = 0; i < done.cap; i++)
        if (done.used[i])
            map_put(&w->have_soon, done.keys[i], 0);
    Map have;
    map_init(&have, 64);
    for (size_t i = 0; i < done.cap; i++)
        if (done.used[i])
            map_put(&have, done.keys[i], 0);
    for (int changed = 1; changed;) {
        changed = 0;
        for (int k = 0; k < FT_UNITS_N; k++) {
            const FtUnit *base = &FT_UNITS[k];
            if (map_get(&have, base->id, NULL))
                continue;
            for (int j = 0; j < base->upgrades_to.count; j++)
                if (map_get(&have, (DWORD)FT_POOL[base->upgrades_to.offset + j], NULL)) {
                    map_put(&have, base->id, 0), map_put(&w->have_soon, base->id, 0);
                    changed = 1;
                    break;
                }
        }
    }
    map_free(&have), map_free(&done);
    for (DWORD i = 0; i < o->n[T_RESEARCH]; i++)
        map_put(&w->research, o->research[i].type_id, o->research[i].level);
    w->food_free = o->h->food_cap - o->h->food_used;
    for (DWORD i = 0; i < o->n[T_INVENTORY]; i++) {
        double at;
        if (map_get(&w->index, o->inventory[i].unit_id, &at))
            map_put(&w->units[(int)at].items, o->inventory[i].type_id, 0);
    }
}

static void own_free(Own *w) {
    for (int i = 0; i < w->n; i++)
        map_free(&w->units[i].learnable), map_free(&w->units[i].items);
    map_free(&w->index), map_free(&w->have_soon), map_free(&w->research), map_free(&w->makes), map_free(&w->builds);
}

static int affords(const Own *w, int gold, int lumber, int food) {
    return w->o->h->gold + INCOME_MARGIN >= gold && w->o->h->lumber + INCOME_MARGIN >= lumber &&
           (food <= 0 || w->food_free >= food);
}
static int requirements_met(const Own *w, FtList requires) {
    for (int k = 0; k < requires.count; k++)
        if (!map_get(&w->have_soon, (DWORD)FT_POOL[requires.offset + k], NULL))
            return 0;
    return 1;
}

/* the arguments each type allows, as argument-index bits (index 0 off) */
typedef struct {
    unsigned char bits[FEAT_ARG_BYTES];
    int any; /* something is allowed (also an order outside the argument indices, as Python's sets count it) */
} Allowed;

static void allow(Allowed *a, DWORD raw) {
    a->any = 1;
    int k = ARGUMENT(raw);
    if (k > 0)
        a->bits[k >> 3] |= (unsigned char)(0x80 >> (k & 7));
}

static void trainable(const Own *w, Allowed *a) {
    for (size_t i = 0; i < w->makes.cap; i++) {
        if (!w->makes.used[i])
            continue;
        const FtUnit *u = unit_info(w->makes.keys[i]);
        if (!u || ((u->flags & FT_STRUCTURE) && !(u->flags & FT_HERO)))
            continue;
        int gold = u->gold, lumber = u->lumber;
        int ok = 1;
        if (u->flags & FT_HERO) {
            if (w->hero_count >= 3)
                continue;
            if (w->hero_count == 0)
                gold = lumber = 0; /* the first hero is free in melee games */
            if (w->hero_count < u->groups_n)
                ok = requirements_met(w, FT_GROUPS[u->groups + w->hero_count]);
        }
        if (ok && affords(w, gold, lumber, u->food) && requirements_met(w, u->requires))
            allow(a, u->id);
    }
}

static void researchable(const Own *w, Allowed *a) {
    for (size_t i = 0; i < w->makes.cap; i++) {
        if (!w->makes.used[i])
            continue;
        DWORD code = w->makes.keys[i];
        const FtUpgrade *up = upgrade_info(code);
        if (up) {
            double level = 0;
            map_get(&w->research, code, &level);
            if ((int)level >= up->levels_n)
                continue;
            const FtUpgradeLevel *lv = &FT_UPGRADE_LEVELS[up->levels + ((int)level < up->levels_n - 1 ? (int)level : up->levels_n - 1)];
            if (affords(w, lv->gold, lv->lumber, 0) && requirements_met(w, lv->requires))
                allow(a, code);
            continue;
        }
        const FtUnit *u = unit_info(code);
        if (u && (u->flags & FT_STRUCTURE) && affords(w, u->gold, u->lumber, 0) && requirements_met(w, u->requires))
            allow(a, code); /* a town hall tier */
    }
}

static void buildable(const Own *w, Allowed *a) {
    for (size_t i = 0; i < w->builds.cap; i++) {
        if (!w->builds.used[i])
            continue;
        const FtUnit *u = unit_info(w->builds.keys[i]);
        if (u && affords(w, u->gold, u->lumber, 0) && requirements_met(w, u->requires))
            allow(a, u->id);
    }
}

static void learnable(Own *w, Allowed *a) { /* fills each hero's set too */
    const Obs *o = w->o;
    for (int i = 0; i < w->n; i++) {
        OwnUnit *h = &w->units[i];
        if (!(h->u->flags & BU_HERO) || (h->u->flags & BU_ILLUSION) || !h->info)
            continue;
        int points = 0;
        for (DWORD k = 0; k < o->n[T_HEROES]; k++)
            if (o->heroes[k].unit_id == h->u->unit_id)
                points = o->heroes[k].skill_points;
        if (points <= 0)
            continue;
        for (int j = 0; j < h->info->hero_abilities.count; j++) {
            DWORD code = (DWORD)FT_POOL[h->info->hero_abilities.offset + j];
            const FtAbility *ab = ability_info(code);
            int known = 0;
            for (DWORD k = 0; k < o->n[T_ABILITIES]; k++)
                if (o->abilities[k].unit_id == h->u->unit_id && o->abilities[k].ability_id == code)
                    known = o->abilities[k].level;
            int max_level = ab ? ab->max_level : 3, required = ab ? ab->required_level : 1, skip = ab ? ab->level_skip : 2;
            if (known >= max_level)
                continue;
            if (h->u->level >= required + known * skip) {
                map_put(&h->learnable, code, 0);
                allow(a, code);
            }
        }
    }
}

static void add_order(OwnUnit *u, DWORD order, Allowed *a) {
    u->castable = 1;
    if (order >= FT_ORDER_FIRST && order < FT_ORDER_FIRST + FT_ORDER_COUNT) {
        DWORD k = order - FT_ORDER_FIRST;
        u->orders[k >> 5] |= 1u << (k & 31);
    } else
        u->cast_other = 1;
    allow(a, order);
}

static void castable(Own *w, Allowed *a) { /* fills each own unit's orders; others' count for the argument mask only */
    const Obs *o = w->o;
    static OwnUnit other; /* a unit that is not the observer's: masks.py lists its orders too (mana 0) */
    for (int i = 0; i < w->n; i++) {
        OwnUnit *u = &w->units[i];
        if (u->info && (u->info->flags & FT_WORKER))
            add_order(u, RESUME_HARVESTING, a), add_order(u, HARVEST, a);
        const FtTypeOrders *t = type_orders(u->u->type_id);
        if (t) {
            for (int k = 0; k < t->orders.count; k++)
                add_order(u, (DWORD)FT_POOL[t->orders.offset + k], a);
        } else if ((u->u->type_id >> 24) == 'n') /* a neutral unit no replay showed casting: any unknown order */
            for (int k = 0; k < FT_UNKNOWN_ORDERS_N; k++)
                add_order(u, FT_UNKNOWN_ORDERS[k], a);
    }
    for (DWORD k = 0; k < o->n[T_ABILITIES]; k++) {
        const BinAbility *rec = &o->abilities[k];
        const FtAbility *ab = ability_info(rec->ability_id);
        if (!ab || rec->level <= 0)
            continue;
        double at;
        OwnUnit *u = map_get(&w->index, rec->unit_id, &at) ? &w->units[(int)at] : &other;
        double mana = u == &other ? 0 : (double)u->u->mana;
        const FtAbilityLevel *lv = NULL, *first = NULL;
        for (int j = 0; j < ab->levels_n; j++) {
            const FtAbilityLevel *x = &FT_ABILITY_LEVELS[ab->levels + j];
            if (x->level == rec->level)
                lv = x;
            if (x->level == 1)
                first = x;
        }
        if (!lv)
            lv = first;
        if (!lv)
            continue;
        int ready = (double)rec->cooldown_remaining <= 0;
        for (int j = 0; j < lv->orders_n; j++) {
            const FtOrder *ord = &FT_ORDERS[lv->orders + j];
            if (ord->cast && !(ready && mana >= rec->mana_cost))
                continue;
            add_order(u, ord->order, a); /* toggles (autocast, deactivate) are always allowed */
        }
    }
}

static void carried(const Own *w, Allowed *a) {
    for (int i = 0; i < w->n; i++)
        for (size_t k = 0; k < w->units[i].items.cap; k++)
            if (w->units[i].items.used[k])
                allow(a, w->units[i].items.keys[k]);
}

/* ---- decide.tables */
typedef struct {
    unsigned char rows[FEAT_N_ARGS][FEAT_MAX_UNITS / 8];
    unsigned char touched[FEAT_N_ARGS];
} ArgRows;

static void mark(ArgRows *r, DWORD raw, int row) {
    int k = ARGUMENT(raw);
    if (k <= 0)
        return; /* masks.py keys these by raw id; decide.tables only looks up argument indices */
    r->rows[k][row >> 3] |= (unsigned char)(0x80 >> (row & 7));
    r->touched[k] = 1;
}

void feat_masks(const Obs *o, FeatOut *out) {
    static Own w;
    static Allowed allowed[12];
    static ArgRows by_group[12];
    static int kinds[FEAT_KINDS];
    static int ready;
    if (!ready) {
        for (int k = 0; k < FEAT_KINDS; k++)
            kinds[k] = kind_of(FT_KINDS[k].name);
        ready = 1;
    }
    own_init(&w, o);
    memset(allowed, 0, sizeof(allowed));
    trainable(&w, &allowed[K_TRAIN]);
    researchable(&w, &allowed[K_RESEARCH]);
    buildable(&w, &allowed[K_BUILD]);
    learnable(&w, &allowed[K_LEARN]);
    castable(&w, &allowed[K_CAST]);
    carried(&w, &allowed[K_ITEM]);
    for (int i = 0; i < w.dead_heroes; i++)
        allow(&allowed[K_REVIVE], w.dead[i]->type_id);
    memset(allowed[K_BUY].bits, 0xff, FEAT_ARG_BYTES); /* what shops sell is not observed: every argument */
    if (FEAT_N_ARGS % 8) /* not the padding bits after the last argument */
        allowed[K_BUY].bits[FEAT_ARG_BYTES - 1] = (unsigned char)(0xff << (8 - FEAT_N_ARGS % 8));
    allowed[K_BUY].any = 1;

    /* the type mask */
    for (int k = 0; k < FEAT_KINDS; k++) {
        int kind = kinds[k], m = 0;
        switch (kind) {
        case K_NOOP: m = 1; break;
        case K_MOVE: m = w.mobile > 0; break;
        case K_RALLY: m = w.structures > 0; break;
        case K_CANCEL: m = w.cancel; break;
        case K_BUY: m = o->h->gold > 0 && w.n > 0; break;
        case K_REVIVE: m = w.dead_heroes > 0; break;
        default: m = allowed[kind].any;
        }
        out->type_mask[k] = (unsigned char)m;
    }

    /* the rows each argument allows, per group of types that share them (masks.actor_masks) */
    int rows_of[MAX_OWN]; /* own unit -> its feature row, -1 if none */
    for (int i = 0; i < w.n; i++)
        rows_of[i] = -1;
    for (int r = 0; r < out->count; r++) {
        double at;
        if (out->owner[r] == 1 && map_get(&w.index, out->unit_id[r], &at))
            rows_of[(int)at] = r;
    }
    static const int GROUPS[] = {K_BUILD, K_LEARN, K_CAST, K_ITEM};
    for (int g = 0; g < 4; g++) {
        ArgRows *r = &by_group[GROUPS[g]];
        memset(r->touched, 0, sizeof(r->touched));
        memset(r->rows, 0, sizeof(r->rows));
        for (int i = 0; i < w.n; i++) {
            int row = rows_of[i];
            OwnUnit *u = &w.units[i];
            if (row < 0)
                continue;
            if (GROUPS[g] == K_BUILD && u->info)
                for (int k = 0; k < u->info->builds.count; k++)
                    mark(r, (DWORD)FT_POOL[u->info->builds.offset + k], row);
            else if (GROUPS[g] == K_LEARN) {
                for (size_t k = 0; k < u->learnable.cap; k++)
                    if (u->learnable.used[k])
                        mark(r, u->learnable.keys[k], row);
            } else if (GROUPS[g] == K_CAST) {
                for (int k = 0; k < FT_ORDER_COUNT; k++)
                    if (u->orders[k >> 5] >> (k & 31) & 1)
                        mark(r, (DWORD)(FT_ORDER_FIRST + k), row);
            } else if (GROUPS[g] == K_ITEM) {
                for (size_t k = 0; k < u->items.cap; k++)
                    if (u->items.used[k])
                        mark(r, u->items.keys[k], row);
            }
        }
    }

    /* decide.tables: argument masks of the allowed types that take one, actor masks of each (type, argument) */
    out->argument_n = 0, out->actor_n = 0;
    for (int k = 0; k < FEAT_KINDS; k++) {
        int kind = kinds[k], flags = FT_KINDS[k].flags;
        if (!out->type_mask[k] || kind == K_NOOP)
            continue;
        unsigned char *bits = NULL;
        if (flags & 1) {
            bits = out->argument_bits[out->argument_n];
            memcpy(bits, allowed[kind].bits, FEAT_ARG_BYTES);
            bits[0] &= 0x7f; /* index 0: no argument */
            int any = 0;
            for (int b = 0; b < FEAT_ARG_BYTES; b++)
                any |= bits[b];
            if (any)
                out->argument_types[out->argument_n++] = k;
            else
                bits = NULL;
        }
        if (!(flags & 2))
            continue;
        if (flags & 4) { /* the actor does not depend on the argument */
            if (out->actor_n == FEAT_MAX_ACTOR)
                continue;
            unsigned char *row = out->actor_rows[out->actor_n];
            memset(row, 0, FEAT_MAX_UNITS / 8);
            for (int i = 0; i < w.n; i++) {
                int r = rows_of[i];
                const BinUnit *u = w.units[i].u;
                int ok = kind == K_BUY ? 1
                         : (kind == K_RALLY || kind == K_CANCEL) ? (u->flags & BU_STRUCTURE) != 0
                         : !(u->flags & BU_STRUCTURE) || (w.units[i].info && (w.units[i].info->flags & FT_ACTING)); /* units inside a burrow, mine or transport come out */
                if (r >= 0 && ok)
                    row[r >> 3] |= (unsigned char)(0x80 >> (r & 7));
            }
            out->actor_keys[out->actor_n++] = k * FEAT_N_ARGS;
            continue;
        }
        if (!bits)
            continue;
        ArgRows *r = &by_group[kind];
        for (int a = 1; a < FEAT_N_ARGS && out->actor_n < FEAT_MAX_ACTOR; a++)
            if ((bits[a >> 3] >> (7 - (a & 7)) & 1) && r->touched[a]) {
                memcpy(out->actor_rows[out->actor_n], r->rows[a], FEAT_MAX_UNITS / 8);
                out->actor_keys[out->actor_n++] = k * FEAT_N_ARGS + a;
            }
    }
    own_free(&w);
}
