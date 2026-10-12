/* What features.c and masks.c share: a map from unit id to a number (also a set), the reference tables' lookup and
 * the parsed binary observation. Internal to them. */
#pragma once
#include "features.h"
#include "features_table.h"
#include <stdlib.h>
#include <string.h>

/* ---- an open-addressing map from unit id to a number (also used as a set) */
typedef struct {
    DWORD *keys;
    double *vals;
    unsigned char *used;
    size_t cap, n;
} Map;

static void map_init(Map *m, size_t cap) {
    m->cap = cap, m->n = 0;
    m->keys = (DWORD *)calloc(cap, sizeof(DWORD));
    m->vals = (double *)calloc(cap, sizeof(double));
    m->used = (unsigned char *)calloc(cap, 1);
    if (!m->keys || !m->vals || !m->used)
        abort();
}
static void map_free(Map *m) {
    free(m->keys), free(m->vals), free(m->used);
}
static void map_clear(Map *m) {
    memset(m->used, 0, m->cap);
    m->n = 0;
}
static size_t map_slot(const Map *m, DWORD key) {
    size_t i = (key * 2654435761u) & (m->cap - 1);
    while (m->used[i] && m->keys[i] != key)
        i = (i + 1) & (m->cap - 1);
    return i;
}
static void map_put(Map *m, DWORD key, double value);
static void map_grow(Map *m) {
    Map bigger;
    map_init(&bigger, m->cap * 2);
    for (size_t i = 0; i < m->cap; i++)
        if (m->used[i])
            map_put(&bigger, m->keys[i], m->vals[i]);
    map_free(m);
    *m = bigger;
}
static void map_put(Map *m, DWORD key, double value) {
    if ((m->n + 1) * 2 > m->cap)
        map_grow(m);
    size_t i = map_slot(m, key);
    if (!m->used[i])
        m->used[i] = 1, m->keys[i] = key, m->n++;
    m->vals[i] = value;
}
static int map_get(const Map *m, DWORD key, double *value) {
    size_t i = map_slot(m, key);
    if (!m->used[i])
        return 0;
    if (value)
        *value = m->vals[i];
    return 1;
}

/* ---- the reference tables */
static int lookup(const FeatId *table, int n, DWORD id, int missing) {
    int lo = 0, hi = n - 1;
    while (lo <= hi) {
        int mid = (lo + hi) / 2;
        if (table[mid].id == id)
            return table[mid].embedding;
        if (table[mid].id < id)
            lo = mid + 1;
        else
            hi = mid - 1;
    }
    return missing;
}
#define TYPE_ID(t) lookup(FT_TYPES, FT_TYPES_N, (t), 0)

typedef struct {
    const BinHeader *h;
    const BinUnit *units;
    const BinAbility *abilities;
    const BinBuff *buffs;
    const BinQueue *queue;
    const BinInventory *inventory;
    const BinItem *items;
    const BinDestructable *destructables;
    const BinEvent *events;
    const BinHero *heroes;
    const BinResearch *research;
    DWORD n[OBS_TABLES];
} Obs;

void feat_masks(const Obs *o, FeatOut *out); /* masks.c */

static int parse(const BYTE *p, size_t size, Obs *o) {
    if (size < sizeof(BinHeader))
        return -1;
    const BinHeader *h = (const BinHeader *)p;
    if (h->magic != OBS_MAGIC || h->version != OBS_VERSION || h->size != size || h->tables < OBS_TABLES)
        return -1;
    o->h = h;
    for (int i = 0; i < OBS_TABLES; i++)
        o->n[i] = h->table[i].count;
#define TABLE(i, type) (const type *)(p + h->table[i].offset)
    o->units = TABLE(T_UNITS, BinUnit);
    o->abilities = TABLE(T_ABILITIES, BinAbility);
    o->buffs = TABLE(T_BUFFS, BinBuff);
    o->queue = TABLE(T_QUEUE, BinQueue);
    o->inventory = TABLE(T_INVENTORY, BinInventory);
    o->items = TABLE(T_ITEMS, BinItem);
    o->destructables = TABLE(T_DESTRUCTABLES, BinDestructable);
    o->events = TABLE(T_EVENTS, BinEvent);
    o->heroes = TABLE(T_HEROES, BinHero);
    o->research = TABLE(T_RESEARCH, BinResearch);
#undef TABLE
    return 0;
}
