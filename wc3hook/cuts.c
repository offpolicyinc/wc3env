/* WC3HOOK_CUTS (bits): game work training does not need, done faster with the same results. Ported from
 * warcraftsim's shim/cuts.c (MIT, Robin Kahlow; same 1.29.2.9231 executable), which checked each against replays
 * played back with and without it (every unit, player, event and AI order at every step). Here
 * training/bench/local_games.py checks them the same way: every step's state hashed against the game without them.
 *
 *    1  the map-version lookup behind the Player / GetPlayerId natives (RVA_SCRIPT_VERSION) memoised: it copies the
 *       cached 14 KB map info of the map path into a stack buffer to read one field, on every call
 *    8  textures without pixels once the match is held (render off): the BLP decoder allocates the same mip table
 *       and buffers but skips each level's JPEG decode; dimensions and mip counts stay as they were
 *   64  A*: the occupancy checks of a node's ring of cells remembered per cell for the length of one search (90%
 *       were repeats; nothing in a search changes a cell's answer), the cell's side effects replayed
 *  128  soft float: count leading zeros as LZCNT; the integer square root as floor(sqrt(double)) (each checked in
 *       process against the original before it is used)
 *  256  render off: the frame's paint handler every WC3HOOK_PAINT_EVERY steps (default 100: 25 s of game time)
 *       instead of never (step.c). Never painting leaves the world frame's unit list that the fog pass walks
 *       unrebuilt: warcraftsim measured that pass grow to 17% of the main thread by 350 s of game time
 * 1024  A*: a neighbour's heuristic (pure: the node's cell, the goal) computed after the relaxation's early return
 *       instead of before it (the same stores, in the same order)
 */
#include "wc3hook.h"

#include <intrin.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

#define VA_(x) (g_base + ((x) - 0x400000))

static int g_cuts = -1;
static long g_paint_every = 100;

int cuts_flags(void) {
    if (g_cuts < 0) {
        char buf[16];
        g_cuts = GetEnvironmentVariableA("WC3HOOK_CUTS", buf, sizeof buf) ? atoi(buf) : 0;
        if (GetEnvironmentVariableA("WC3HOOK_PAINT_EVERY", buf, sizeof buf) && atoi(buf) > 0)
            g_paint_every = atoi(buf);
    }
    return g_cuts;
}

long cuts_paint_every(void) {
    return g_paint_every;
}

static int hook(DWORD va, void *fn, void **orig, const char *name) {
    BYTE *p = VA_(va);
    if (!(p[0] == 0x55 && p[1] == 0x8b && p[2] == 0xec)) {
        hook_log("cuts: %s: unexpected prologue, not hooked", name);
        return 0;
    }
    if (MH_CreateHook(p, fn, orig) != MH_OK) {
        hook_log("cuts: %s: hook failed", name);
        return 0;
    }
    return 1;
}

/* ---- 1: map version ------------------------------------------------------------------------- */
typedef int(__cdecl *MapVerFn)(const char *path);
static MapVerFn MapVer_orig;
static CRITICAL_SECTION g_mv_lock;
static char g_mv_path[MAX_PATH];
static int g_mv_val, g_mv_ok;

static int __cdecl MapVer_hook(const char *path) {
    if (!path)
        return MapVer_orig(path); /* (the current map: read directly, cheap) */
    EnterCriticalSection(&g_mv_lock);
    if (g_mv_ok && strncmp(path, g_mv_path, sizeof g_mv_path) == 0) {
        int v = g_mv_val;
        LeaveCriticalSection(&g_mv_lock);
        return v;
    }
    LeaveCriticalSection(&g_mv_lock);
    int v = MapVer_orig(path);
    if (v != -1 && strlen(path) < sizeof g_mv_path) {
        EnterCriticalSection(&g_mv_lock);
        strcpy(g_mv_path, path);
        g_mv_val = v;
        g_mv_ok = 1;
        LeaveCriticalSection(&g_mv_lock);
    }
    return v;
}

/* ---- 8: textures without pixels --------------------------------------------------------------- */
#define VA_BLP_DECODE 0x83cdf0 /* thiscall int decode_mips(BLP *, int format, int *table, unsigned first) */
#define VA_MIP_ALLOC 0xcbecc0  /* cdecl int mip_alloc(int format, unsigned w, unsigned h, int fill) */
typedef int(__fastcall *BlpDecodeFn)(BYTE *self, void *edx, int format, int *table, unsigned first);
typedef int(__cdecl *MipAllocFn)(int format, unsigned w, unsigned h, int fill);
static BlpDecodeFn BlpDecode_orig;

static int __fastcall BlpDecode_hook(BYTE *self, void *edx, int format, int *table, unsigned first) {
    if (!g_held || g_render)
        return BlpDecode_orig(self, edx, format, table, first);
    unsigned nmips = *(unsigned *)(self + 0x4b8);
    if (!((*(int *)(self + 0x20) != 0 || first == 0) && first < nmips))
        return 0;
    unsigned w = *(unsigned *)(self + 0x14) >> (first & 31), h = *(unsigned *)(self + 0x18) >> (first & 31);
    int t = ((MipAllocFn)(void *)VA_(VA_MIP_ALLOC))(format, w > 1 ? w : 1, h > 1 ? h : 1, 0);
    *table = t;
    return t != 0;
}

/* ---- 64: the ring checks of an A* run ----------------------------------------------------------
 * Within one call of the search loop nothing changes what a cell query answers: its callees pop and push the open
 * list, relax nodes, and map cells to node indices (a type-2 entry linked at the head of the cell's list, which the
 * query skips). The query (thiscall(search, x, y), AL: 1 free, 0 blocked) reads the cell's bits and its pathing
 * objects with the search's mask (+0xa4), target (+0xa8) and flag (+0xce), sets +0xcc when it meets the target and
 * +0xcd when it answers blocked. */
#define VA_RING                                                                                                        \
    0x8be080 /* thiscall unsigned ring(search, x, y, r, n), ret 0x10: bit k = k-th cell of the ring blocked */
#define VA_CELL 0x8bdbe0  /* thiscall char cell(search, x, y), ret 8 */
#define VA_ASTAR 0x8bf8b0 /* fastcall int run(search) */
typedef unsigned(__fastcall *RingFn)(BYTE *, void *, int, int, int, int);
typedef unsigned(__fastcall *CellFn)(BYTE *, void *, int, int);
typedef int(__fastcall *AstarFn)(BYTE *);
static RingFn Ring_orig;
static AstarFn Astar_orig;
static CellFn Cell_fn;
static BYTE *g_srch, *g_smap;
static DWORD g_sgen, g_smask, g_starget;
static BYTE g_sce;
static DWORD *g_stamp;
static BYTE *g_sval;
static unsigned g_sw, g_sh, g_scap;

static int __fastcall Astar_hook(BYTE *s) {
    BYTE *map = *(BYTE **)(s + 0x1c);
    if (g_srch || !map) /* (nested: not remembered) */
        return Astar_orig(s);
    unsigned w = *(DWORD *)(map + 0x3c), h = *(DWORD *)(map + 0x40);
    if (w && h && w <= 4096 && h <= 4096) {
        if (w * h > g_scap) {
            free(g_stamp);
            free(g_sval);
            g_scap = w * h;
            g_stamp = calloc(g_scap, sizeof *g_stamp);
            g_sval = malloc(g_scap);
            if (!g_stamp || !g_sval) {
                g_scap = 0;
                return Astar_orig(s);
            }
        }
        if (++g_sgen == 0) {
            memset(g_stamp, 0, g_scap * sizeof *g_stamp);
            g_sgen = 1;
        }
        g_sw = w;
        g_sh = h;
        g_smap = map;
        g_smask = *(DWORD *)(s + 0xa4);
        g_starget = *(DWORD *)(s + 0xa8);
        g_sce = s[0xce];
        g_srch = s;
    }
    int r = Astar_orig(s);
    g_srch = NULL;
    return r;
}

static __inline int cell_q(BYTE *s, int x, int y) {
    if ((unsigned)x < g_sw && (unsigned)y < g_sh) {
        unsigned i = (unsigned)y * g_sw + (unsigned)x;
        if (g_stamp[i] == g_sgen) {
            BYTE v = g_sval[i];
            if (v & 2)
                s[0xcc] = 1;
            if (!(v & 1))
                s[0xcd] = 1;
            return v & 1;
        }
        BYTE cc = s[0xcc];
        s[0xcc] = 0;
        int r = (BYTE)Cell_fn(s, NULL, x, y) != 0;
        BYTE met = s[0xcc];
        s[0xcc] = cc | met;
        g_stamp[i] = g_sgen;
        g_sval[i] = (BYTE)((r ? 1 : 0) | (met ? 2 : 0));
        return r;
    }
    return (BYTE)Cell_fn(s, NULL, x, y) != 0;
}

static unsigned __fastcall Ring_hook(BYTE *s, void *edx, int x, int y, int r, int n) {
    if (s != g_srch || *(BYTE **)(s + 0x1c) != g_smap || *(DWORD *)(s + 0xa4) != g_smask ||
        *(DWORD *)(s + 0xa8) != g_starget || s[0xce] != g_sce)
        return Ring_orig(s, edx, x, y, r, n);
    /* the original's walk: the top row left to right, the right column down, the bottom row right to left, the left
     * column up; bit k for the k-th cell */
    int x0 = x - r, y0 = y - r, i, j;
    unsigned mask = 0, bit = 1;
    for (i = x0; i < x0 + n; i++, bit += bit)
        if (!cell_q(s, i, y0))
            mask |= bit;
    i--;
    for (j = y0 + 1; j < y0 + n; j++, bit += bit)
        if (!cell_q(s, i, j))
            mask |= bit;
    i--;
    j--;
    for (; i >= x0; i--, bit += bit)
        if (!cell_q(s, i, j))
            mask |= bit;
    j--;
    i++;
    for (; j >= y0 + 1; j--, bit += bit)
        if (!cell_q(s, i, j))
            mask |= bit;
    return mask;
}

/* ---- 1024: A* relaxation with the heuristic after the early return ----------------------------- */
#define VA_RELAX 0x8bf980  /* thiscall void relax(search, node, parent, cost), ret 0xc */
#define VA_HEUR 0x8bda00   /* cdecl unsigned h(int *xy, int *goal, int k) */
#define VA_UNLINK 0x8bfbc0 /* thiscall void unlink(search, node), ret 4 */
#define VA_PUSH 0x8a3090   /* thiscall void push(heap, key, node, gen), ret 0xc */
typedef void(__fastcall *RelaxFn)(BYTE *, void *, int, int, int);
typedef unsigned(__cdecl *HeurFn)(int *, int *, int);
typedef void(__fastcall *UnlinkFn)(BYTE *, void *, int);
typedef void(__fastcall *PushFn)(BYTE *, void *, unsigned, int, unsigned);
static RelaxFn Relax_orig;

static void __fastcall Relax_hook(BYTE *s, void *edx, int idx, int parent, int cost) {
    DWORD *n = (DWORD *)(*(BYTE **)(s + 0x30) + idx * 0x24);
    unsigned g = *(DWORD *)(*(BYTE **)(s + 0x30) + parent * 0x24 + 0x14) + (unsigned)cost;
    if (n[3] == 0xffffffffu) {
        int dy = (int)n[1] - *(int *)(s + 0x8c), dx = (int)n[0] - *(int *)(s + 0x88);
        unsigned d2 = (unsigned)dy * (unsigned)dy + (unsigned)dx * (unsigned)dx;
        if (d2 < *(DWORD *)(s + 0x98)) {
            *(DWORD *)(s + 0x98) = d2;
            *(int *)(s + 0x9c) = idx;
        }
    } else {
        if (g >= n[5])
            return;
        if (n[3] < 0xfffffffeu)
            ((UnlinkFn)(void *)VA_(VA_UNLINK))(s, NULL, idx);
        else {
            n[2]++;
            n[3] = 0xffffffffu;
            n[4] = 0xffffffffu;
        }
    }
    unsigned h = ((HeurFn)(void *)VA_(VA_HEUR))((int *)n, (int *)(s + 0x88), 0xf);
    n[5] = g;
    n[7] = (DWORD)parent;
    n[6] = h;
    DWORD *m = (DWORD *)(*(BYTE **)(s + 0x30) + idx * 0x24);
    unsigned key = m[6] + m[5];
    m[2]++;
    unsigned gen = m[2];
    m[3] = 0xfffffffeu;
    m[4] = 0xffffffffu;
    ((PushFn)(void *)VA_(VA_PUSH))(s + 0x44, NULL, key, idx, gen);
}

/* ---- 128: soft float helpers ------------------------------------------------------------------ */
#define VA_CLZ 0x9a0330   /* fastcall int clz(unsigned): bsr-based, 32 for 0 */
#define VA_ISQRT 0x9a2150 /* cdecl unsigned isqrt(unsigned): floor(sqrt(x)), 16 rounds */
typedef unsigned(__cdecl *IsqrtFn)(unsigned);
static IsqrtFn Isqrt_orig;

static unsigned isqrt_orig_c(unsigned x) { /* (the original, transcribed: for the self-test) */
    unsigned rem = 0, root = 0;
    for (int k = 0; k < 16; k++) {
        rem = (rem << 2) | (x >> 30);
        x <<= 2;
        root += root;
        if (root < rem) {
            rem += ~root;
            root += 2;
        }
    }
    return root >> 1;
}

static unsigned __cdecl Isqrt_hook(unsigned x) {
    return (unsigned)sqrt((double)x); /* correctly rounded, and every 32-bit integer is exact in a double */
}

static void install_softfloat(void) {
    BYTE *clz = VA_(VA_CLZ), *sq = VA_(VA_ISQRT);
    static const BYTE clz_code[] = {0x55, 0x8b, 0xec, 0x51, 0x89, 0x4d, 0xfc, 0x0f, 0xbd, 0x55, 0xfc};
    int info[4];
    __cpuid(info, 0x80000001);
    if (memcmp(clz, clz_code, sizeof clz_code) == 0 && ((info[2] >> 5) & 1)) {
        /* in-process check against the original on edge cases and a sweep */
        typedef int(__fastcall * ClzFn)(unsigned);
        static const BYTE lz[] = {0xf3, 0x0f, 0xbd, 0xc1, 0xc3}; /* lzcnt eax, ecx; ret */
        BYTE *buf = VirtualAlloc(NULL, 4096, MEM_COMMIT | MEM_RESERVE, PAGE_EXECUTE_READWRITE);
        int bad = !buf;
        if (buf) {
            memcpy(buf, lz, sizeof lz);
            unsigned v = 0;
            for (unsigned k = 0; k < 1u << 20 && !bad; k++, v = v * 1664525u + 1013904223u) {
                unsigned t = k < 64 ? (k < 33 ? (k == 32 ? 0 : 1u << k) : ~0u >> (k - 33)) : v >> (k & 31);
                bad = ((ClzFn)(void *)clz)(t) != ((ClzFn)(void *)buf)(t);
            }
            VirtualFree(buf, 0, MEM_RELEASE);
        }
        DWORD old;
        if (!bad && VirtualProtect(clz, sizeof lz, PAGE_EXECUTE_READWRITE, &old)) {
            memcpy(clz, lz, sizeof lz);
            VirtualProtect(clz, sizeof lz, old, &old);
            FlushInstructionCache(GetCurrentProcess(), clz, sizeof lz);
            hook_log("cuts: clz as lzcnt");
        } else
            hook_log("cuts: clz NOT patched (bad=%d)", bad);
    } else
        hook_log("cuts: clz: unexpected code or no lzcnt, not patched");
    if (sq[0] == 0x55 && sq[1] == 0x8b && sq[2] == 0xec && sq[3] == 0x53) {
        /* the transcription against the code, then the replacement against the transcription */
        int bad = 0;
        unsigned v = 12345;
        for (unsigned k = 0; k < 1u << 18 && !bad; k++, v = v * 1664525u + 1013904223u) {
            unsigned t = k < 4096 ? k * k + (k & 1 ? 0u : ~0u) : v;
            unsigned o = ((IsqrtFn)(void *)sq)(t);
            bad = o != isqrt_orig_c(t) || o != Isqrt_hook(t);
        }
        if (!bad && hook(VA_ISQRT, (void *)Isqrt_hook, (void **)&Isqrt_orig, "isqrt"))
            hook_log("cuts: isqrt as floor(sqrt(double))");
        else if (bad)
            hook_log("cuts: isqrt: MISMATCH in the self-test, not hooked");
    }
}

/* With the other hooks, before MH_EnableHook(MH_ALL_HOOKS) (main.c). */
void cuts_init_hooks(void) {
    int cuts = cuts_flags();
    if (!cuts)
        return;
    InitializeCriticalSection(&g_mv_lock);
    if (cuts & 1)
        hook(0x400000 + RVA_SCRIPT_VERSION, (void *)MapVer_hook, (void **)&MapVer_orig, "map version");
    if (cuts & 8)
        hook(VA_BLP_DECODE, (void *)BlpDecode_hook, (void **)&BlpDecode_orig, "texture decode");
    if (cuts & 64) {
        Cell_fn = (CellFn)(void *)VA_(VA_CELL);
        if (hook(VA_ASTAR, (void *)Astar_hook, (void **)&Astar_orig, "A* run"))
            hook(VA_RING, (void *)Ring_hook, (void **)&Ring_orig, "A* ring");
    }
    if (cuts & 128)
        install_softfloat();
    if (cuts & 1024)
        hook(VA_RELAX, (void *)Relax_hook, (void **)&Relax_orig, "A* relax");
    hook_log("cuts: %d (paint every %ld steps)", cuts, g_paint_every);
}
