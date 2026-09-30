/* wc3hook.dll: injected into Warcraft III.exe 1.29.2.9231 (32-bit) by inject.py, which
 * starts the game suspended and resumes it after this DLL signals `wc3hook-ready-<pid>`.
 *
 * One file per concern (docs/design.md):
 *   main.c        DllMain, hook installation order, the ready signal, heartbeat
 *   transport.c   the named pipe; RPC lines go to rpc.c, the rest is the diagnostic command set
 *   rpc.c         the JSON-lines RPC of docs/specs/protocol.md: lifecycle, jobs on the game thread
 *   clock.c       virtual QPC / GetTickCount / FILETIME / rdtsc, `speed` times wall time,
 *                 freezable, read lock-free; Sleep and wait timeouts scaled; the wait floor
 *   allocator.c   native free-space search and current-arena selection for reallocations
 *   step.c        GameUpdate hook and step mode, the hold at a fixed start time, render off,
 *                 turn packet delivery
 *   observe.c     the observation: accessors copied from the natives, fog, inventory, ground
 *                 items, destructables, the game result per player
 *   events.c      game events from detours on the per-kind event fire functions
 *   act.c         orders through the player command queue
 *   players.c     startup AI suppression and offline command identities
 *   stage.c       staging natives behind the `debug` ops (spawn, kill, give, ...)
 *   instance.c    per-process mutex/event names, per-instance Documents, focus immunity
 *   diag.c        scan, dump, watch, profile, where; exit and dialog logging
 *
 * Host -> DLL: a JSON object per line is an RPC request (docs/specs/protocol.md). Anything else is a
 * diagnostic command: ping | echo | info | where | pktlog <n> | scan <hex> | dump <addr> <n> |
 * watch <addr> [skip_game] [hits] | profile <n> [lo hi] | trace 0|1
 * DLL -> host: one JSON reply per request; diagnostic output as `scan ...`, `dump ...`,
 * `watch ...`, `queued ...` lines, which the RPC client skips.
 *
 * Every RVA below is into the 1.29.2.9231 exe (image base randomised; add g_base). The file
 * is pinned to that exact binary. tools/disasm.py, tools/imports.py and tools/natives.py read it. */
#pragma once
#define _CRT_SECURE_NO_WARNINGS
#include <windows.h>
#include <objbase.h>
#include <intrin.h>
#include <stdio.h>
#include <stdarg.h>
#include <tlhelp32.h>
#include "MinHook.h"
#include "json.h"

#define DLL_VERSION "0.8"
#define MAX_STEP_MS 60000
#define MAX_SPEED 2048.0
#define MAX_WAIT_FLOOR_MS 1000

/* ---- RVA table ------------------------------------------------------------------------------ */
#define RVA_BLOCK_ALLOC 0x040190      /* cdecl(lock group*, arena, flags, size) */
#define RVA_ARENA_SEARCH_END 0x03f2d2 /* free-list search result before native growth */
#define RVA_ARENA_COALESCE 0x03f640   /* cdecl(arena): merge adjacent free blocks */
#define RVA_REALLOC_MOVE_RET 0x0403fa /* allocation fallback inside SMemReAlloc */
#define RVA_ALLOCATOR_ARENAS 0xd30f40 /* heads of 256 native arena groups */
#define RVA_GAME_UI_DESTROY 0x1e9660  /* CGameUI destructor */
#define RVA_EVENT_UNREGISTER 0x4405d0 /* cdecl(event, callback, context, match_flags) */
#define RVA_GAME_UI_EVENT12 0x1f0180
#define RVA_GAME_UI_EVENT14 0x1f0200
#define RVA_OBSERVER_COLLECT 0x43f700 /* thiscall(table, &ring): detach buckets */
#define RVA_OBSERVER_REHASH 0x43f220  /* thiscall(table, &ring): prune empty nodes, restore live ones */
#define RVA_RDTSC_HELPER 0x396c80     /* rdtsc; ret, calibrated against QPC at 0x396bd0 */
#define RVA_GAMEUPDATE 0x1aefd0       /* GameUpdate(this, now_ms) thiscall, once per frame */
#define FRAME_LOOP_LO 0x445630        /* the frame loop, for `profile` */
#define FRAME_LOOP_HI 0x445a30
#define RVA_GXPRESENT 0x3d1070       /* end of the frame function 0x559330 */
#define RVA_PAINT 0x559330           /* cdecl frame paint handler (event 0x11); returns 1 */
#define RVA_PAINT_PRESENT_FLAGS 0xce9a88 /* the flags it passes to GxPresent */
#define RVA_GX_DEVICE 0xd4ab54       /* graphics backend singleton */
#define RVA_GX_D3D9_VTABLE 0xaba2e0  /* Direct3D 9 backend */
#define GX_D3D9_DEVICE 0x59c         /* IDirect3DDevice9* inside that backend */
#define RVA_PKT_DELIVER 0x1aa930     /* Storm event handler (id 0x18): time-slot packet -> list game+0x1308 */
#define RVA_TURN_PRODUCER 0x557be0   /* offline host: build and send a time slot */
#define RVA_TURN_TIME_WRITE 0x54c960 /* serialize a u16 by reference */
#define RVA_TURN_TIME_RET 0x557db2   /* only this producer call supplies turn time */
#define RVA_SAVE_REPLAY 0x1b1490     /* game->SaveReplay(path): close the recorder, move TempReplay.w3g */
#define RVA_TLS_SLOT 0xd3b518        /* the per-thread component table's TLS slot index */
#define RVA_FRAME_WAIT_RET 0x3c2ede  /* return address of the frame loop's pacing wait (clock.c) */
#define RVA_FRAME_LOOP_WAIT_RET 0x44574f /* where the frame loop's call of that wait wrapper returns; other
                                           * callers (0x575dcb) block inside game code */
/* game object (GameUpdate's this) */
#define GAME_TIME_MS 0x2038      /* game time */
#define GAME_PREV_NOW 0x2614     /* previous now_ms */
#define GAME_PENDING_MS 0x2618   /* pending game ms */
#define GAME_TURN_CREDIT 0x3c4   /* turn ms available */
#define GAME_OUT_STORE 0x2040    /* the outgoing action store */
#define GAME_CONNECTION 0x898    /* active connection: 0 live game, 1 replay */
#define GAME_REPLAY_LENGTH 0xe80 /* recorded duration from the native replay header */
/* natives (tools/natives.py) and the internals they call */
#define RVA_N_PLAYER 0x0a4240    /* (I)Hplayer; */
#define RVA_N_LOCAL_PLAYER 0x093790 /* GetLocalPlayer: ()Hplayer; */
#define RVA_N_PLAYERID 0x0945f0  /* GetPlayerId (Hplayer;)I */
#define RVA_N_UNITORDER 0x0973a0 /* GetUnitCurrentOrder (Hunit;)I */
/* Live ability readback; this build's mana/cooldown natives take ONE-based levels (disassembled). */
#define RVA_N_UNITABILITYLEVEL 0x0971c0    /* GetUnitAbilityLevel (Hunit;I)I */
#define RVA_N_ABILITYMANACOST 0x097220     /* BlzGetUnitAbilityManaCost (Hunit;II)I */
#define RVA_N_ABILITYCOOLDOWN 0x097150     /* BlzGetUnitAbilityCooldown (Hunit;II)R */
#define RVA_N_ABILITYCOOLDOWNLEFT 0x097190 /* BlzGetUnitAbilityCooldownRemaining (Hunit;I)R */
/* Live combat numbers and hero stats for the binary observation (observe.c) */
#define RVA_N_UNITARMOR 0x097280       /* BlzGetUnitArmor (Hunit;)R */
#define RVA_N_UNITBASEDAMAGE 0x097320  /* BlzGetUnitBaseDamage (Hunit;I)I */
#define RVA_N_UNITDICENUMBER 0x0975b0  /* BlzGetUnitDiceNumber (Hunit;I)I */
#define RVA_N_UNITDICESIDES 0x097600   /* BlzGetUnitDiceSides (Hunit;I)I */
#define RVA_N_UNITATTACKPERIOD 0x0972c0 /* BlzGetUnitAttackCooldown (Hunit;I)R: the attack period */
#define RVA_N_UNITMOVESPEED 0x0978b0   /* GetUnitMoveSpeed (Hunit;)R */
#define RVA_N_UNITFACING 0x097650      /* GetUnitFacing (Hunit;)R */
#define RVA_N_RESOURCEAMOUNT 0x095300  /* GetResourceAmount (Hunit;)I */
#define RVA_N_HEROXP 0x092be0          /* GetHeroXP (Hunit;)I */
#define RVA_N_HEROSKILLPOINTS 0x092a90 /* GetHeroSkillPoints (Hunit;)I */
#define RVA_N_HEROSTR 0x092b70         /* GetHeroStr (Hunit;B)I */
#define RVA_N_HEROAGI 0x092830         /* GetHeroAgi (Hunit;B)I */
#define RVA_N_HEROINT 0x0928a0         /* GetHeroInt (Hunit;B)I */
#define RVA_N_TECHCOUNT 0x094bb0       /* GetPlayerTechCount (Hplayer;IB)I */
#define RVA_N_FLOATGAMESTATE 0x092530  /* GetFloatGameState (Hfgamestate;)R: 2 is the time of day */
#define RVA_N_STARTMELEEAI 0x0ac230        /* StartMeleeAI (Hplayer;S)V */
#define RVA_N_STARTCAMPAIGNAI 0x0ac200     /* StartCampaignAI (Hplayer;S)V */
#define RVA_N_GETPLAYERSTATE 0x094a80      /* (Hplayer;Hplayerstate;)I */
#define RVA_N_PLAYERRACE 0x0946e0          /* GetPlayerRace (Hplayer;)Hrace; */
#define RVA_N_PLAYERCONTROL 0x094560       /* GetPlayerController (Hplayer;)Hmapcontrol; */
#define RVA_N_PAUSECOMPAI 0x0a3e60         /* PauseCompAI (Hplayer;B)V */
#define RVA_N_PLAYERSLOTSTATE 0x094980     /* GetPlayerSlotState (Hplayer;)Hplayerslotstate; */
#define RVA_FOG_WORLD 0xd3b6f4             /* IsVisibleToPlayer's (0x9a4a0) world global; +0x34 is the fog map */
#define RVA_FOG_QUERY 0x09a420             /* thiscall fog->visible(x, y, z, player bits), callee cleans */
#define RVA_FOG_QUERY_Z 0xd2d88c           /* the z the native passes */
#define RVA_N_REMOVEPLAYER 0x0a51c0        /* (Hplayer;Hplayergameresult;)V: what melee victory and defeat call */
#define RVA_HANDLE_OBJECT 0x083940         /* cdecl(Hunit) -> unit object: the lookup every unit native starts with */
#define RVA_PLAYER_OBJECT 0x082770         /* cdecl(Hplayer) -> player object */
#define RVA_ENUM_OBJECTS 0x3f4ba0          /* (class, cb, ctx, 0): for every live object of a registry class */
#define RVA_UNIT_OWNER 0x292340
#define RVA_UNIT_STATE 0x2935c0
#define RVA_POS_UNPACK 0x3fcc50
#define RVA_ITEM_IN_SLOT 0x291820
#define RVA_PROTECTED_INT 0x3fb2b0
#define RVA_RESOLVE_PAIR 0x3fc7e0 /* (ecx = &ref pair) -> object or 0: items' owners, abilities */
#define RVA_VM_GLOBAL 0xd3b6f4    /* the JASS VM, set whenever a map is loaded */
#define RVA_VM_STR_CTX 0x069010   /* JassStringToC / the VM string context, for the script version */
#define RVA_SCRIPT_VERSION 0x304330
#define RVA_STORE_VTABLE 0xa78c2c /* CDataStore */
#define RVA_STORE_CTOR 0x0d7140
#define RVA_STORE_DTOR 0x0d7040
#define RVA_PUTBYTE 0x05ce80
#define RVA_PUTDATA 0x05d230
#define RVA_QUEUE_ACTION 0x1ae190      /* QueueLocalAction(store, player) */
#define RVA_SEND_LOOPBACK 0x1aab40     /* game->SendPacket(type, store, flags) */
#define RVA_REGISTER_PLAYER 0x1a5820   /* table->Register(wire id, name/info) -> 0x80 | (wire id - 1) */
#define RVA_ASSIGN_PLAYER 0x1b1990     /* table->Assign(waiting slot, game slot) */
#define RVA_UNREGISTER_PLAYER 0x1a5e70 /* table->Unregister(wire id) */
#define RVA_N_RESTARTGAME 0x0a5620
#define UNIT_CLASS 0x2b773375         /* '+w3u' */
#define DESTRUCTABLE_CLASS 0x2b773364 /* '+w3d' */
#define ITEM_CLASS 0x6974656d         /* 'item': what CreateItem's allocator (0x2c4d40) registers under */
#define ITEM_LIFE 0x160               /* GetWidgetLife's item getter (0x2c2e20) reads current life here */
/* second instance guard: CreateEventA("Warcraft III Game Application") 0x2e9ff -> 0x397c80.
 * game logger 0x3b8f0: hooking it stalls the simulation, do not. */

/* ---- shared state --------------------------------------------------------------------------- */
enum { ST_LAUNCHED, ST_IN_GAME, ST_ENDED }; /* the RPC lifecycle (docs/specs/protocol.md) */
extern volatile LONG g_status;
extern LONG g_step_frames_last;
extern char g_dir[MAX_PATH];
extern HANDLE g_log;
extern BYTE *g_base;
extern CRITICAL_SECTION g_lock; /* guards the pipe handle */

extern BYTE *g_game; /* GameUpdate's this, once seen */
extern DWORD g_game_tid;
extern volatile LONG g_stepmode, g_stepping, g_held;
extern DWORD g_step_target;
extern volatile LONG g_render, g_pktlog, g_steplog, g_wait_floor;
extern volatile LONG g_trace, g_trace_n;
extern int g_watch_skip_game, g_watch_hits;
extern double g_speed;
extern volatile LONG g_frozen;
extern LONGLONG g_qpc_freq;
extern DWORD g_exc_addr;

typedef VOID(WINAPI *SleepFn)(DWORD);
extern SleepFn Sleep_orig; /* the real Sleep, for the DLL's own threads */

/* ---- main.c ----------------------------------------------------------------------------------- */
void hook_log(const char *fmt, ...);
MH_STATUS require_hook(MH_STATUS status, const char *name);
#define MH_CreateHook(target, hook, original) require_hook(MH_CreateHook(target, hook, original), #target)
#define MH_CreateHookApi(module, name, hook, original)                                                                 \
    require_hook(MH_CreateHookApi(module, name, hook, original), name)

/* ---- transport.c ------------------------------------------------------------------------------ */
void pipe_send(const char *line);
DWORD WINAPI pipe_thread(LPVOID arg);

/* ---- clock.c ---------------------------------------------------------------------------------- */
LONGLONG real_qpc(void);
DWORD real_wait(HANDLE event, DWORD ms);
void clock_wake(void);
void clock_set_speed(double s);
void clock_freeze(int on);
void clock_init_hooks(void);

/* ---- step.c ----------------------------------------------------------------------------------- */
DWORD game_time(void);
int game_is_offline(void);
int game_is_replay(void);
void step_mode(int on);
const char *step_request(DWORD ms);
void step_end_after_turn(void);
void step_restart(void);
extern const char *g_step_reason;
void step_init_hooks(void);
extern volatile LONG g_frames_seen;

#include "obsbin.h"

/* ---- observe.c -------------------------------------------------------------------------------- */
typedef void(__cdecl *EnumObjectsFn)(int cls, void *cb, void *ctx, int zero);
typedef int(__cdecl *NativeI_V)(void);
typedef int(__cdecl *NativeI_I)(int);
typedef int(__cdecl *NativeI_II)(int, int);
typedef int(__cdecl *NativeI_III)(int, int, int);
#define NATIVE(rva, T) ((T)(g_base + (rva)))
extern BYTE *g_fn_owner, *g_fn_state, *g_fn_unpack, *g_vm_global, *g_fn_ctx, *g_fn_ver, *g_fn_slot, *g_fn_resolve_pair;
void accessors_init(void);
int __cdecl unit_xy_bits(BYTE *u, int idx);
int __cdecl unit_state_bits(BYTE *u, int state);
BYTE *__cdecl unit_owner(BYTE *u);
int __cdecl player_jass_id(BYTE *pl);
int __cdecl unit_visible(BYTE *u, int player_slot);
BYTE *__cdecl unit_item_in_slot(BYTE *u, int slot);
int __cdecl item_owned(BYTE *it);
int __cdecl script_version(void);
int player_slot(int jass_id);
int object_visible(BYTE *object, int cls, int player);
float bits_to_f(int i);
int unit_hero_level(BYTE *u);
unsigned obs_id(BYTE *o);                        /* the RPC's id of a unit or item: the game's own object id [o+0xc] */
void obs_write_json(JW *w, int player, int seq); /* game thread snapshot */
DWORD obs_write_binary(BYTE *dst, DWORD cap, int player, int seq); /* obsbin.h; 0 if over cap */
void obs_clear(void); /* the map is reloading */
const char *player_result(int jass_id);          /* "", "victory" or "defeat", from the game's own RemovePlayer */
void result_init_hooks(void);
void results_clear(void);
int agents_finished(void);

/* ---- metadata.c ----------------------------------------------------------------------------- */
void metadata_clear(void);
void metadata_write(JW *w, int observer, const int score[25]); /* score: metadata_scores */
void metadata_scores(int observer, int score[25]);

/* ---- setup.c -------------------------------------------------------------------------------- */
void setup_init_hooks(void);
void setup_clear(void);
void setup_hold(void);
const char *setup_error(void);
int setup_random_race(int player);
void setup_write(JW *w);

/* ---- events.c --------------------------------------------------------------------------------- */
unsigned events_gather(BB *out, int player); /* BinEvent records; returns the number lost to ring overflow */
void events_json(JW *w, const BinEvent *events, size_t n);
unsigned orders_gather(BB *out, int player); /* BinOrder records; returns the number lost to ring overflow */
void events_set_players(unsigned mask);  /* create_game: the players events are judged for; empties the log */
void events_init_hooks(void);

/* ---- act.c ------------------------------------------------------------------------------------ */
void act_flush(void);
const char *act_configure(unsigned agents);
void act_clear(void);
void players_init_hooks(void);
const char *players_configure(unsigned agents);
int player_is_agent(int player);
int player_wire_id(int player);
int player_prepared_agent(int player);
const char *player_arg(yyjson_val *args, int *handle);
void difficulty_init_hooks(void);
void difficulty_apply(int player);
const char *difficulty_read(yyjson_val *args, JW *w);
const char *overlay_show(yyjson_val *args, JW *w);
void chat_key(UINT msg, WPARAM wp); /* window thread: every key message the game window gets */
void chat_write_json(JW *w);        /* game thread: lines typed into the chat box since the last call */
void players_clear(void);
void players_unregister(void);
void act_init_hooks(void);
BYTE *object_by_rpc_id(long long id); /* game thread */
BYTE *unit_by_rpc_id(long long id);   /* unit-only lookup, game thread */

/* ---- stage.c ---------------------------------------------------------------------------------- */
typedef const char *(*StageOp)(yyjson_val *args, JW *w); /* game thread: NULL, or the bad_params detail */
StageOp stage_op(const char *name);
void stage_each_step(void);
void stage_clear(void);
int handle_of(BYTE *obj); /* game thread: the object's JASS handle, allocated if it has none */

/* ---- instance.c ------------------------------------------------------------------------------- */
void instance_init_hooks(void);

/* ---- allocator.c ------------------------------------------------------------------------------ */
void allocator_init_hooks(void);

/* ---- diag.c ----------------------------------------------------------------------------------- */
void mem_scan(const char *hex);
void mem_dump(DWORD addr, int n);
int watch_set(DWORD addr);
void profile(int n, DWORD lo, DWORD hi);
void where(void);
void log_addr(const char *prefix, DWORD addr);
void log_stack_from(const char *label, DWORD *frame);
void log_stack_here(const char *label);
int native_exc(EXCEPTION_POINTERS *e); /* the __except filter for game code called from the DLL */
void diag_init_hooks(void);

/* ---- rpc.c ------------------------------------------------------------------------------------ */
void rpc_init(void);
void rpc_handle(const char *line); /* one request line; replies on the pipe */
void rpc_service_jobs(void);       /* game thread, every frame */
int rpc_job_pending(void);
void rpc_service_jobs_between_frames(void); /* step.c: from the pacing wait, with the frame's TLS table installed */
void rpc_step_finished(void);
void rpc_set_ended(void);

typedef struct {
    long long unit_id;
    char command[16];
    int has_xy;
    double x, y;
    long long target;
    char type_id[8];
    char order[32];
    int slot;
    long long shop;
    char item_type[8];
    int reason, queued, auto_place;
} ActItem;
typedef struct {
    ActItem *items;
    int n;
    int player;
} ActJob;
enum { RJ_OK, RJ_NOT_YOURS, RJ_UNKNOWN_UNIT, RJ_UNKNOWN_COMMAND, RJ_BAD_ARGS, RJ_QUEUE_FULL, RJ_NO_BUILD_SITE };
void act_apply(ActJob *job); /* act.c: validate, encode and queue a batch; game thread */

/* Building search beneath the stock AI's 0x6d4970 wrapper; verified on 1.29.2. */
#define RVA_PLACEMENT_INIT 0x6d3de0     /* thiscall(scratch; slot, anchor x*, y*, type, density, worker) */
#define RVA_PLACEMENT_COPY 0x6d3c00     /* thiscall(destination; source scratch) */
#define RVA_PLACEMENT_SEARCH 0x6d41f0   /* thiscall(scratch; score*, site[3], primary bool) */
#define RVA_PLACEMENT_ENUM 0x28ea00     /* cdecl(callback, scratch, flags, radius, x, y, 2, 31) */
#define RVA_PLACEMENT_OBSTACLE 0x6d4c10 /* engine footprint callback for the search */
#define RVA_PLACEMENT_RADIUS 0xdb4b78   /* engine search radius, float bits */
#define RVA_PLACEMENT_NO_SITE 0xd2d804  /* engine failure score */
#define RVA_PLAYER_TYPE_COUNT 0x0bd4a0  /* thiscall(player; type, counter kind) */
#define RVA_UNIT_PENDING_TYPE 0x294700  /* thiscall(unit; type): pending production */
#define RVA_TYPE_IS_BUILDING 0x1cc9f0   /* cdecl(type) -> building predicate */
#define PLACEMENT_SEARCH_SIZE 0x1947c   /* native scratch structure size from 0x6d4970 */
/* A structure already ordered but not yet standing: another search must not choose its ground. */
typedef struct {
    unsigned type;
    float x, y;
    BYTE *worker;
} PendingSite;
/* `ignore`: the worker whose own pending site this order replaces (NULL when the order is queued). */
int placement_find(BYTE *worker, unsigned type, float *x, float *y, const PendingSite *pending, int n_pending,
                   const BYTE *ignore);
int placement_pending(int player, PendingSite *out, int max); /* the player's workers' current build orders */
int unit_order_point(BYTE *u, unsigned *id, float *x, float *y); /* observe.c: the current order's id and point */
