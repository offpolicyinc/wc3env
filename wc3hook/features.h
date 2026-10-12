/* The model's features and masks of a binary observation (docs/training/12-hook-features.md): the one
 * implementation, compiled into the game hook for live play and as a library (training/policy/native.py) for
 * observations already recorded. Plain C on the observation's bytes. */
#pragma once
#include "obsbin.h"

#define FEAT_MAX_UNITS 256
#define FEAT_FLOATS 69
#define FEAT_MAP 128
#define FEAT_HISTORY 4
#define FEAT_MAX_TREES 48
#define FEAT_INVENTORY 6
#define FEAT_BUFFS 8
#define FEAT_ABILITIES 5
#define FEAT_KINDS 21     /* training's action TYPES */
#define FEAT_N_ARGS 2828  /* argument indices, 0 for none (masks.argument_ids) */
#define FEAT_ARG_BYTES ((FEAT_N_ARGS + 7) / 8)
#define FEAT_MAX_ACTOR 4096

/* A decision as features need it: its type (training's TYPES index), the actor and target unit ids (0 if none)
 * and its point, if it had one. */
typedef struct {
    int type;
    DWORD actor, target;
    int has_point;
    double x, y;
} FeatDecision;

/* What a game's observations don't repeat: the playable bounds (min_x, min_y, max_x, max_y), both races (the
 * observer's first; 0 human, 1 orc, 2 undead, 3 night elf) and the map's starting trees per cell. */
typedef struct {
    double bounds[4];
    int races[2];
    float trees[FEAT_MAP * FEAT_MAP];
} FeatStatic;

typedef struct {
    int count; /* rows filled */
    float units[FEAT_MAX_UNITS][FEAT_FLOATS];
    int type[FEAT_MAX_UNITS], owner[FEAT_MAX_UNITS], order[FEAT_MAX_UNITS], queue[FEAT_MAX_UNITS];
    int inventory[FEAT_MAX_UNITS][FEAT_INVENTORY], buffs[FEAT_MAX_UNITS][FEAT_BUFFS];
    int abilities[FEAT_MAX_UNITS][FEAT_ABILITIES];
    DWORD unit_id[FEAT_MAX_UNITS];
    int cell[FEAT_MAX_UNITS];
    int previous_type[FEAT_HISTORY], recent[FEAT_HISTORY]; /* recent: the last decisions' point cells, -1 none */
    unsigned char trees[FEAT_MAP * FEAT_MAP];              /* remembered trees per cell, at most 4 */
    int scalar_count;
    float scalars[2048];
    /* masks (training/policy/decide.py's tables), bits packed as numpy.packbits packs them (most significant first) */
    unsigned char type_mask[FEAT_KINDS];
    int argument_n;                                       /* the allowed types that take an argument, in type order */
    int argument_types[FEAT_KINDS];
    unsigned char argument_bits[FEAT_KINDS][FEAT_ARG_BYTES]; /* index 0 (no argument) off */
    int actor_n;                                          /* one per (type, argument) with an actor mask */
    int actor_keys[FEAT_MAX_ACTOR];                       /* type * FEAT_N_ARGS + argument index (0: any argument) */
    unsigned char actor_rows[FEAT_MAX_ACTOR][FEAT_MAX_UNITS / 8];
} FeatOut;

typedef struct FeatState FeatState;

#if defined(FEAT_DLL) && defined(_WIN32) /* the library training/policy/native.py loads */
#define FEAT_API __declspec(dllexport)
#elif defined(FEAT_DLL)
#define FEAT_API __attribute__((visibility("default")))
#else
#define FEAT_API
#endif
FEAT_API FeatState *feat_new(const FeatStatic *s);
FEAT_API void feat_free(FeatState *st);
/* Fold in one observation (the bytes binary.py parses) and the decision taken on the one before (NULL: none),
 * then write its features. Returns 0, or -1 if the bytes are not an observation of this version. */
FEAT_API int feat_step(FeatState *st, const BYTE *obs, size_t size, const FeatDecision *last, FeatOut *out);

/* ---- the features block the hook writes after a player's binary observation (src/wc3env/binary.py's
 * parse_features reads it). A header, then each array for `count` rows (or the other counts) in this order:
 *   units float[count][FEAT_FLOATS]; type, owner, order, queue int[count]; inventory int[count][6];
 *   buffs int[count][8]; abilities int[count][5]; unit_id DWORD[count]; cell int[count];
 *   trees BYTE[FEAT_MAP * FEAT_MAP]; scalars float[scalar_count];
 *   argument_types int[argument_n]; argument_bits BYTE[argument_n][FEAT_ARG_STRIDE];
 *   actor_keys int[actor_n]; actor_rows BYTE[actor_n][FEAT_MAX_UNITS / 8]
 * Every array starts 4-byte aligned. */
#define FEAT_MAGIC 0x31544657 /* "WFT1" little-endian */
#define FEAT_VERSION 1
#define FEAT_ARG_STRIDE ((FEAT_ARG_BYTES + 3) & ~3)
typedef struct {
    DWORD magic, version, size;
    DWORD count, scalar_count, argument_n, actor_n;
    int previous_type[FEAT_HISTORY], recent[FEAT_HISTORY];
    BYTE type_mask[(FEAT_KINDS + 3) & ~3];
} FeatBlockHeader;
/* Writes out's block at dst; returns its size, 0 if it does not fit in cap. */
FEAT_API DWORD feat_write(const FeatOut *out, BYTE *dst, DWORD cap);
FEAT_API size_t feat_out_size(void);
