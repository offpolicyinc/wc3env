/* features.h: training/policy/features.py's Memory.update + features(map=False), in C. Each number is computed
 * as numpy computes it there: in double where Python works on its own floats or on float64 arrays, in float where
 * numpy keeps float32 arrays (NEP 50: a Python float meeting a float32 array becomes float32), so the results
 * agree to float32 rounding (sin, cos, log1p may differ from numpy's in the last bit). */
#include "features_obs.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>

#define PI 3.14159265358979323846
#define CREEP_PLAYER 12
#define SIGHT 800.0
#define NO_TARGET 0xffffffffu
enum { OWNER_NONE, OWNER_ME, OWNER_ENEMY, OWNER_CREEP, OWNER_NEUTRAL, OWNER_ITEM, OWNER_TREE };
enum { EV_ATTACKED = 18, EV_DEATH = 20, EV_CONSTRUCT = 26, EV_UPGRADE = 29, EV_TRAIN = 32, EV_RESEARCH = 35 };

static int is_worker(DWORD type) {
    for (int i = 0; i < FT_WORKERS_N; i++)
        if (FT_WORKERS[i] == type)
            return 1;
    return 0;
}

static int order_class(DWORD id) { /* features.order_class: ORDER_CLASSES' index */
    if (!id)
        return 1; /* idle */
    if (id >> 24)
        return 8; /* build */
    switch (id) {
    case 851986: case 851990: return 2; /* move */
    case 851983: case 851984: return 3; /* attack */
    case 851971: return 4;              /* smart */
    case 851972: case 851993: return 5; /* stop/hold */
    case 852018: case 851970: return 6; /* harvest */
    case 852017: return 7;              /* return */
    case 852024: return 9;              /* repair */
    }
    return id >= 852000 ? 10 : 11; /* cast, other */
}

/* ---- state */
typedef struct {
    BinUnit row;
    double seen;
} Remembered;

struct FeatState {
    FeatStatic s;
    float trees[FEAT_MAP * FEAT_MAP];
    Remembered *remembered; /* in the order first seen, as Python's dict keeps it */
    int remembered_n, remembered_cap;
    Map started, hp_before, hp_now, attacked;
    FeatDecision recent[FEAT_HISTORY]; /* newest first */
    int recent_n;
};

FeatState *feat_new(const FeatStatic *s) {
    FeatState *st = (FeatState *)calloc(1, sizeof(FeatState));
    if (!st)
        abort();
    st->s = *s;
    memcpy(st->trees, s->trees, sizeof(st->trees));
    map_init(&st->started, 256), map_init(&st->hp_before, 512), map_init(&st->hp_now, 512), map_init(&st->attacked, 64);
    return st;
}

void feat_free(FeatState *st) {
    if (!st)
        return;
    free(st->remembered);
    map_free(&st->started), map_free(&st->hp_before), map_free(&st->hp_now), map_free(&st->attacked);
    free(st);
}


/* features._cell on a float32 array (destructables): numpy keeps float32 */
static int cell_f32(float x, float y, const double *b) {
    float fx = (x - (float)b[0]) / (float)(b[2] - b[0]) * (float)FEAT_MAP;
    float fy = (y - (float)b[1]) / (float)(b[3] - b[1]) * (float)FEAT_MAP;
    int cx = (int)fx, cy = (int)fy;
    cx = cx < 0 ? 0 : cx > FEAT_MAP - 1 ? FEAT_MAP - 1 : cx;
    cy = cy < 0 ? 0 : cy > FEAT_MAP - 1 ? FEAT_MAP - 1 : cy;
    return cy * FEAT_MAP + cx;
}
/* features._cell on float64 (Python floats, or rows joined with them) */
static int cell_f64(double x, double y, const double *b) {
    int cx = (int)((x - b[0]) / (b[2] - b[0]) * FEAT_MAP);
    int cy = (int)((y - b[1]) / (b[3] - b[1]) * FEAT_MAP);
    cx = cx < 0 ? 0 : cx > FEAT_MAP - 1 ? FEAT_MAP - 1 : cx;
    cy = cy < 0 ? 0 : cy > FEAT_MAP - 1 ? FEAT_MAP - 1 : cy;
    return cy * FEAT_MAP + cx;
}

static int in_sight(const Obs *o, const BinUnit *row) { /* Memory._in_sight, in float32 as numpy does */
    for (DWORD i = 0; i < o->n[T_UNITS]; i++) {
        const BinUnit *u = &o->units[i];
        if (!(u->flags & BU_OWN))
            continue;
        float dx = u->x - row->x, dy = u->y - row->y;
        float d2 = dx * dx + dy * dy;
        if (d2 < (float)(SIGHT * SIGHT))
            return 1;
    }
    return 0;
}

static int remembered_find(const FeatState *st, DWORD uid) {
    for (int i = 0; i < st->remembered_n; i++)
        if (st->remembered[i].row.unit_id == uid)
            return i;
    return -1;
}
static void remembered_remove(FeatState *st, int i) {
    memmove(&st->remembered[i], &st->remembered[i + 1], (st->remembered_n - i - 1) * sizeof(Remembered));
    st->remembered_n--;
}

static void update(FeatState *st, const Obs *o, const FeatDecision *last) { /* Memory.update */
    double now = o->h->game_time_ms / 1000.0;
    const double *b = st->s.bounds;
    Map visible;
    map_init(&visible, 1024);
    for (DWORD i = 0; i < o->n[T_UNITS]; i++) {
        const BinUnit *u = &o->units[i];
        map_put(&visible, u->unit_id, 0);
        if ((u->flags & BU_STRUCTURE) && !(u->flags & BU_OWN) && u->owner != CREEP_PLAYER) {
            int k = remembered_find(st, u->unit_id);
            if (k < 0) {
                if (st->remembered_n == st->remembered_cap) {
                    st->remembered_cap = st->remembered_cap ? st->remembered_cap * 2 : 64;
                    st->remembered = (Remembered *)realloc(st->remembered, st->remembered_cap * sizeof(Remembered));
                    if (!st->remembered)
                        abort();
                }
                k = st->remembered_n++;
            }
            st->remembered[k].row = *u, st->remembered[k].seen = now;
        }
    }
    map_clear(&st->attacked);
    for (DWORD i = 0; i < o->n[T_EVENTS]; i++) {
        const BinEvent *e = &o->events[i];
        if (e->kind == EV_CONSTRUCT || e->kind == EV_UPGRADE || e->kind == EV_TRAIN || e->kind == EV_RESEARCH)
            map_put(&st->started, e->unit_id, now);
        else if (e->kind == EV_DEATH) {
            int k = remembered_find(st, e->unit_id);
            if (k >= 0)
                remembered_remove(st, k);
        } else if (e->kind == EV_ATTACKED)
            map_put(&st->attacked, e->unit_id, 0);
    }
    for (int k = 0; k < st->remembered_n;) { /* in sight but not there: destroyed unseen */
        const BinUnit *r = &st->remembered[k].row;
        if (!map_get(&visible, r->unit_id, NULL) && in_sight(o, r))
            remembered_remove(st, k);
        else
            k++;
    }
    map_free(&visible);
    /* trees: cells within SIGHT of an own unit show the standing trees there now */
    static unsigned char seen[FEAT_MAP * FEAT_MAP];
    static float now_trees[FEAT_MAP * FEAT_MAP];
    memset(seen, 0, sizeof(seen));
    memset(now_trees, 0, sizeof(now_trees));
    double w = (b[2] - b[0]) / FEAT_MAP, hgt = (b[3] - b[1]) / FEAT_MAP;
    for (DWORD i = 0; i < o->n[T_UNITS]; i++) {
        const BinUnit *u = &o->units[i];
        if (!(u->flags & BU_OWN))
            continue;
        double x = u->x, y = u->y;
        int i0 = (int)((x - SIGHT - b[0]) / w), i1 = (int)((x + SIGHT - b[0]) / w) + 1;
        int j0 = (int)((y - SIGHT - b[1]) / hgt), j1 = (int)((y + SIGHT - b[1]) / hgt) + 1;
        i0 = i0 < 0 ? 0 : i0, j0 = j0 < 0 ? 0 : j0;
        i1 = i1 > FEAT_MAP ? FEAT_MAP : i1, j1 = j1 > FEAT_MAP ? FEAT_MAP : j1;
        for (int j = j0; j < j1; j++) {
            double cy = b[1] + (j + 0.5) * hgt;
            for (int i2 = i0; i2 < i1; i2++) {
                double cx = b[0] + (i2 + 0.5) * w;
                if ((cy - y) * (cy - y) + (cx - x) * (cx - x) < SIGHT * SIGHT)
                    seen[j * FEAT_MAP + i2] = 1;
            }
        }
    }
    for (DWORD i = 0; i < o->n[T_DESTRUCTABLES]; i++) {
        const BinDestructable *d = &o->destructables[i];
        if (d->flags & BD_LUMBER)
            now_trees[cell_f32(d->x, d->y, b)] += 1.0f;
    }
    for (int c = 0; c < FEAT_MAP * FEAT_MAP; c++)
        if (seen[c])
            st->trees[c] = now_trees[c];
    if (last) {
        memmove(&st->recent[1], &st->recent[0], (FEAT_HISTORY - 1) * sizeof(FeatDecision));
        st->recent[0] = *last;
        if (st->recent_n < FEAT_HISTORY)
            st->recent_n++;
    }
    Map t = st->hp_before; /* hp_before, hp_now = hp_now, {this observation's} */
    st->hp_before = st->hp_now, st->hp_now = t;
    map_clear(&st->hp_now);
    for (DWORD i = 0; i < o->n[T_UNITS]; i++)
        map_put(&st->hp_now, o->units[i].unit_id, o->units[i].hp);
}

/* the rows, in features()' order */
typedef struct {
    const BinUnit *u;
    double last_seen;
} Row;

static int cmp_tree(const void *a, const void *b) { /* by distance, then by index: argsort(kind="stable") */
    const float *x = (const float *)a, *y = (const float *)b;
    if (x[0] != y[0])
        return x[0] < y[0] ? -1 : 1;
    return x[1] < y[1] ? -1 : x[1] > y[1] ? 1 : 0;
}

static int nearest_trees(const Obs *o, const BinDestructable **out) { /* features._nearest_trees */
    static const BinUnit *close[4096];
    static float order[8192][2]; /* distance, index */
    static const BinDestructable *lumber[8192];
    int n_lumber = 0, n_own = 0, n_close = 0;
    for (DWORD i = 0; i < o->n[T_DESTRUCTABLES] && n_lumber < 8192; i++)
        if (o->destructables[i].flags & BD_LUMBER)
            lumber[n_lumber++] = &o->destructables[i];
    for (DWORD i = 0; i < o->n[T_UNITS]; i++)
        n_own += (o->units[i].flags & BU_OWN) != 0;
    if (!n_lumber || !n_own)
        return 0;
    for (DWORD i = 0; i < o->n[T_UNITS] && n_close < 4096; i++) {
        const BinUnit *u = &o->units[i];
        if ((u->flags & BU_OWN) && (is_worker(u->type_id) || (u->flags & BU_STRUCTURE)))
            close[n_close++] = u;
    }
    if (!n_close)
        for (DWORD i = 0; i < o->n[T_UNITS] && n_close < 4096; i++)
            if (o->units[i].flags & BU_OWN)
                close[n_close++] = &o->units[i];
    for (int t = 0; t < n_lumber; t++) {
        float best = INFINITY;
        for (int k = 0; k < n_close; k++) {
            float dx = lumber[t]->x - close[k]->x, dy = lumber[t]->y - close[k]->y;
            float d2 = dx * dx + dy * dy;
            if (d2 < best)
                best = d2;
        }
        order[t][0] = best, order[t][1] = (float)t;
    }
    qsort(order, n_lumber, sizeof(order[0]), cmp_tree);
    int n = n_lumber < FEAT_MAX_TREES ? n_lumber : FEAT_MAX_TREES;
    for (int i = 0; i < n; i++)
        out[i] = lumber[(int)order[i][1]];
    return n;
}

static void positions(double x, double y, const double *b, float *f) { /* features._positions, one row */
    double fx = 2 * (x - b[0]) / (b[2] - b[0]) - 1, fy = 2 * (y - b[1]) / (b[3] - b[1]) - 1;
    f[0] = (float)fx, f[1] = (float)fy;
    for (int k = 0; k < 4; k++) {
        double w = PI * (double)(1 << k);
        f[2 + 4 * k] = (float)sin(w * fx), f[3 + 4 * k] = (float)cos(w * fx);
        f[4 + 4 * k] = (float)sin(w * fy), f[5 + 4 * k] = (float)cos(w * fy);
    }
}

static int owner_class(int owner, int player) {
    if (owner == player)
        return OWNER_ME;
    if (owner < CREEP_PLAYER)
        return OWNER_ENEMY;
    return owner == CREEP_PLAYER ? OWNER_CREEP : OWNER_NEUTRAL;
}

static void unit_row(const FeatState *st, const Obs *o, const BinUnit *u, double last_seen, int i, FeatOut *out) {
    double now = o->h->game_time_ms / 1000.0;
    DWORD uid = u->unit_id, flags = u->flags;
    double x = u->x, y = u->y, max_hp = u->max_hp, max_mana = u->max_mana, hp = u->hp;
    double v[FEAT_FLOATS - 18];
    int n = 0;
    v[n++] = max_hp ? hp / max_hp : 0;
    v[n++] = max_hp / 1000;
    v[n++] = max_mana ? (double)u->mana / max_mana : 0;
    v[n++] = max_mana / 1000;
    v[n++] = (flags & BU_OWN) != 0, v[n++] = (flags & BU_STRUCTURE) != 0, v[n++] = (flags & BU_HERO) != 0;
    v[n++] = (flags & BU_INSIDE) != 0, v[n++] = last_seen > 0;
    v[n++] = (flags & BU_ILLUSION) != 0, v[n++] = (flags & BU_DEAD) != 0;
    v[n++] = last_seen / 60 < 5 ? last_seen / 60 : 5;
    v[n++] = u->level / 10.0;
    if (u->order_id && (u->order_target != NO_TARGET || u->order_x != 0 || u->order_y != 0)) {
        v[n++] = ((double)u->order_x - x) / 1000, v[n++] = ((double)u->order_y - y) / 1000, v[n++] = 1;
    } else {
        v[n++] = 0, v[n++] = 0, v[n++] = 0;
    }
    double progress = 0;
    if (u->state == 1) { /* constructing: hit points rise from 10% to full */
        if (max_hp) {
            progress = (hp / max_hp - 0.1) / 0.9;
            progress = progress < 0 ? 0 : progress > 1 ? 1 : progress;
        }
    } else {
        double total = u->state_seconds ? (double)u->state_seconds : (double)u->queue_seconds, started;
        if (total && map_get(&st->started, uid, &started)) {
            progress = (now - started) / total;
            progress = progress < 1 ? progress : 1;
        }
    }
    int queued = 0;
    DWORD first_queued = 0;
    for (DWORD k = 0; k < o->n[T_QUEUE]; k++)
        if (o->queue[k].unit_id == uid) {
            if (!queued)
                first_queued = o->queue[k].type_id;
            queued++;
        }
    v[n++] = progress, v[n++] = queued / 7.0;
    double facing = (double)u->facing * (PI / 180.0);
    v[n++] = (double)u->armor / 10, v[n++] = u->damage_min / 100.0, v[n++] = u->damage_max / 100.0;
    v[n++] = (double)u->attack_period / 2, v[n++] = (double)u->move_speed / 500, v[n++] = sin(facing), v[n++] = cos(facing);
    v[n++] = u->resource / 10000.0;
    const BinHero *hero = NULL;
    for (DWORD k = 0; k < o->n[T_HEROES]; k++)
        if (o->heroes[k].unit_id == uid)
            hero = &o->heroes[k]; /* the last one, as the dict keeps it */
    if (hero) {
        v[n++] = hero->xp / 5000.0, v[n++] = hero->skill_points / 3.0;
        v[n++] = hero->strength / 50.0, v[n++] = hero->agility / 50.0, v[n++] = hero->intelligence / 50.0;
    } else {
        for (int k = 0; k < 5; k++)
            v[n++] = 0;
    }
    for (int k = 0; k < FEAT_HISTORY; k++) {
        int actor = k < st->recent_n && st->recent[k].actor == uid, target = k < st->recent_n && st->recent[k].target == uid;
        v[n++] = actor, v[n++] = target;
    }
    double before;
    v[n++] = map_get(&st->attacked, uid, NULL);
    v[n++] = (map_get(&st->hp_before, uid, &before) && before && max_hp) ? ((before - hp) / max_hp > 0 ? (before - hp) / max_hp : 0) : 0;
    int abilities = 0;
    for (DWORD k = 0; k < o->n[T_ABILITIES] && abilities < FEAT_ABILITIES; k++) {
        const BinAbility *a = &o->abilities[k];
        if (a->unit_id != uid)
            continue;
        double cd = a->cooldown_seconds;
        v[n++] = a->level / 3.0, v[n++] = cd ? (double)a->cooldown_remaining / cd : 0;
        out->abilities[i][abilities++] = lookup(FT_ABILITIES, FT_ABILITIES_N, a->ability_id, 0);
    }
    for (; abilities < FEAT_ABILITIES; abilities++)
        v[n++] = 0, v[n++] = 0;
    for (int k = 0; k < n; k++)
        out->units[i][18 + k] = (float)v[k];
    out->type[i] = TYPE_ID(u->type_id);
    out->owner[i] = owner_class(u->owner, (int)o->h->player);
    out->order[i] = order_class(u->order_id);
    DWORD building = (flags & BU_OWN) && (u->order_id >> 24) ? u->order_id : 0;
    out->queue[i] = TYPE_ID(queued ? first_queued : building);
    for (DWORD k = 0; k < o->n[T_INVENTORY]; k++) /* the last record per slot wins, as in the dict */
        if (o->inventory[k].unit_id == uid && o->inventory[k].slot < FEAT_INVENTORY)
            out->inventory[i][o->inventory[k].slot] = TYPE_ID(o->inventory[k].type_id);
    int buffs = 0;
    for (DWORD k = 0; k < o->n[T_BUFFS] && buffs < FEAT_BUFFS; k++)
        if (o->buffs[k].unit_id == uid)
            out->buffs[i][buffs++] = lookup(FT_BUFFS, FT_BUFFS_N, o->buffs[k].buff_id, 0);
    out->unit_id[i] = uid;
}

static void scalars(const FeatState *st, const Obs *o, FeatOut *out) { /* features._scalars */
    const BinHeader *h = o->h;
    double food_cap = h->food_cap > 1 ? h->food_cap : 1;
    int upkeep = h->food_used <= 50 ? 0 : h->food_used <= 80 ? 1 : 2;
    double tod = (double)h->time_of_day / 24 * 2 * PI;
    float *s = out->scalars;
    int n = 0;
    s[n++] = (float)(log1p((double)h->gold) / 8), s[n++] = (float)(log1p((double)h->lumber) / 8);
    s[n++] = (float)(h->food_used / 100.0), s[n++] = (float)(h->food_cap / 100.0);
    s[n++] = (float)(h->food_used / food_cap), s[n++] = (float)(upkeep / 2.0);
    s[n++] = (float)(h->game_time_ms / 1000.0 / 1800), s[n++] = (float)sin(tod), s[n++] = (float)cos(tod);
    for (int k = 0; k < 2; k++)
        for (int r = 0; r < 4; r++)
            s[n++] = (float)(st->s.races[k] == r);
    float *research = s + n;
    memset(research, 0, FT_UPGRADES_N * sizeof(float));
    for (DWORD k = 0; k < o->n[T_RESEARCH]; k++) {
        int at = lookup(FT_UPGRADES, FT_UPGRADES_N, o->research[k].type_id, -1);
        if (at >= 0)
            research[at] = (float)(o->research[k].level / 3.0);
    }
    n += FT_UPGRADES_N;
    float *counts = s + n;
    memset(counts, 0, (FT_TYPES_N + 1) * sizeof(float));
    for (DWORD k = 0; k < o->n[T_UNITS]; k++)
        if (o->units[k].flags & BU_OWN)
            counts[TYPE_ID(o->units[k].type_id)] += 1.0f;
    for (int k = 0; k <= FT_TYPES_N; k++)
        counts[k] = sqrtf(counts[k]) / 4.0f;
    n += FT_TYPES_N + 1;
    out->scalar_count = n;
}

int feat_step(FeatState *st, const BYTE *bytes, size_t size, const FeatDecision *last, FeatOut *out) {
    Obs o;
    if (parse(bytes, size, &o))
        return -1;
    update(st, &o, last);
    memset(out, 0, sizeof(*out));
    const double *b = st->s.bounds;
    double now = o.h->game_time_ms / 1000.0;
    int player = (int)o.h->player;

    static Row rows[4096 + 1024];
    int n_rows = 0;
#define ADD(cond)                                                                                                     \
    for (DWORD i = 0; i < o.n[T_UNITS]; i++) {                                                                        \
        const BinUnit *u = &o.units[i];                                                                               \
        if (cond)                                                                                                     \
            rows[n_rows].u = u, rows[n_rows++].last_seen = 0;                                                         \
    }
    Map visible;
    map_init(&visible, 1024);
    for (DWORD i = 0; i < o.n[T_UNITS]; i++)
        map_put(&visible, o.units[i].unit_id, 0);
    ADD(u->flags & BU_OWN)
    ADD(!(u->flags & BU_OWN) && u->owner < CREEP_PLAYER)
    for (int k = 0; k < st->remembered_n; k++)
        if (!map_get(&visible, st->remembered[k].row.unit_id, NULL) && st->remembered[k].row.owner < CREEP_PLAYER)
            rows[n_rows].u = &st->remembered[k].row, rows[n_rows++].last_seen = now - st->remembered[k].seen;
    ADD(!(u->flags & BU_OWN) && u->owner == CREEP_PLAYER)
    ADD(!(u->flags & BU_OWN) && u->owner > CREEP_PLAYER)
    for (int k = 0; k < st->remembered_n; k++)
        if (!map_get(&visible, st->remembered[k].row.unit_id, NULL) && st->remembered[k].row.owner > CREEP_PLAYER)
            rows[n_rows].u = &st->remembered[k].row, rows[n_rows++].last_seen = now - st->remembered[k].seen;
#undef ADD
    map_free(&visible);

    const BinDestructable *trees[FEAT_MAX_TREES];
    int n_trees = nearest_trees(&o, trees);
    if (n_rows > FEAT_MAX_UNITS - n_trees)
        n_rows = FEAT_MAX_UNITS - n_trees;
    int room = FEAT_MAX_UNITS - n_trees - n_rows, n_items = (int)o.n[T_ITEMS] < room ? (int)o.n[T_ITEMS] : room;

    int i = 0;
    for (int r = 0; r < n_rows; r++, i++) {
        unit_row(st, &o, rows[r].u, rows[r].last_seen, i, out);
        positions(rows[r].u->x, rows[r].u->y, b, out->units[i]);
        out->cell[i] = cell_f64(rows[r].u->x, rows[r].u->y, b);
    }
    for (int k = 0; k < n_items; k++, i++) {
        out->type[i] = TYPE_ID(o.items[k].type_id), out->owner[i] = OWNER_ITEM, out->unit_id[i] = o.items[k].item_id;
        positions(o.items[k].x, o.items[k].y, b, out->units[i]);
        out->cell[i] = cell_f64(o.items[k].x, o.items[k].y, b);
    }
    for (int k = 0; k < n_trees; k++, i++) {
        out->owner[i] = OWNER_TREE, out->unit_id[i] = trees[k]->id;
        positions(trees[k]->x, trees[k]->y, b, out->units[i]);
        out->cell[i] = cell_f64(trees[k]->x, trees[k]->y, b);
    }
    out->count = i;
    (void)player;

    for (int k = 0; k < FEAT_HISTORY; k++)
        out->previous_type[k] = k < st->recent_n ? st->recent[k].type : 0, out->recent[k] = -1;
    for (int k = 0, c = 0; k < st->recent_n; k++)
        if (st->recent[k].has_point)
            out->recent[c++] = cell_f64(st->recent[k].x, st->recent[k].y, b);
    for (int c = 0; c < FEAT_MAP * FEAT_MAP; c++)
        out->trees[c] = (unsigned char)(st->trees[c] < 4 ? st->trees[c] : 4);
    scalars(st, &o, out);
    feat_masks(&o, out);
    return 0;
}

DWORD feat_write(const FeatOut *f, BYTE *dst, DWORD cap) {
    DWORD n = (DWORD)f->count, at = sizeof(FeatBlockHeader);
    struct {
        const void *p;
        DWORD bytes;
    } parts[] = {
        {f->units, n * FEAT_FLOATS * 4}, {f->type, n * 4}, {f->owner, n * 4}, {f->order, n * 4}, {f->queue, n * 4},
        {f->inventory, n * FEAT_INVENTORY * 4}, {f->buffs, n * FEAT_BUFFS * 4}, {f->abilities, n * FEAT_ABILITIES * 4},
        {f->unit_id, n * 4}, {f->cell, n * 4}, {f->trees, FEAT_MAP * FEAT_MAP}, {f->scalars, (DWORD)f->scalar_count * 4},
        {f->argument_types, (DWORD)f->argument_n * 4},
    };
    DWORD size = at;
    for (size_t k = 0; k < sizeof(parts) / sizeof(parts[0]); k++)
        size += (parts[k].bytes + 3) & ~3u;
    size += (DWORD)f->argument_n * FEAT_ARG_STRIDE + (DWORD)f->actor_n * 4 + (DWORD)f->actor_n * (FEAT_MAX_UNITS / 8);
    if (size > cap)
        return 0;
    FeatBlockHeader h;
    memset(&h, 0, sizeof h);
    h.magic = FEAT_MAGIC, h.version = FEAT_VERSION, h.size = size;
    h.count = n, h.scalar_count = (DWORD)f->scalar_count, h.argument_n = (DWORD)f->argument_n, h.actor_n = (DWORD)f->actor_n;
    memcpy(h.previous_type, f->previous_type, sizeof h.previous_type);
    memcpy(h.recent, f->recent, sizeof h.recent);
    memcpy(h.type_mask, f->type_mask, FEAT_KINDS);
    memcpy(dst, &h, sizeof h);
    for (size_t k = 0; k < sizeof(parts) / sizeof(parts[0]); k++) {
        memcpy(dst + at, parts[k].p, parts[k].bytes);
        memset(dst + at + parts[k].bytes, 0, ((parts[k].bytes + 3) & ~3u) - parts[k].bytes);
        at += (parts[k].bytes + 3) & ~3u;
    }
    for (int a = 0; a < f->argument_n; a++, at += FEAT_ARG_STRIDE) {
        memcpy(dst + at, f->argument_bits[a], FEAT_ARG_BYTES);
        memset(dst + at + FEAT_ARG_BYTES, 0, FEAT_ARG_STRIDE - FEAT_ARG_BYTES);
    }
    memcpy(dst + at, f->actor_keys, (DWORD)f->actor_n * 4), at += (DWORD)f->actor_n * 4;
    memcpy(dst + at, f->actor_rows, (DWORD)f->actor_n * (FEAT_MAX_UNITS / 8));
    return size;
}

size_t feat_out_size(void) { /* for the library's callers, which only pass a buffer through to feat_write */
    return sizeof(FeatOut);
}
