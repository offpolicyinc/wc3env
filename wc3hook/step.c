/* GameUpdate hook and step mode, the hold at a fixed start time, render off, turn packet delivery. */
#define COBJMACROS
#include "wc3hook.h"
#include <d3d9.h>

BYTE *g_game; /* GameUpdate's this, once seen */
static void *g_frame_tls;
DWORD g_game_tid;
volatile LONG g_stepmode;   /* 1: clock frozen between steps; GameUpdate feeds the sim */
volatile LONG g_render = 1; /* 0: skip drawing and presentation once the match is held */
volatile LONG g_pktlog;     /* `pktlog <n>`: log the next n turn-packet deliveries */
volatile LONG g_steplog;    /* `steplog <n>`: timeline of the next n steps */
static LONGLONG g_steplog_t0;
#define STEPLOG(fmt, ...)                                                                                 \
    do {                                                                                                  \
        if (g_steplog > 0)                                                                                \
            hook_log("steplog %7.3f ms " fmt, 1000.0 * (real_qpc() - g_steplog_t0) / g_qpc_freq, __VA_ARGS__); \
    } while (0)

/* ---- GameUpdate and step mode -----------------------------------------------------------
 * thiscall GameUpdate(this, now_ms) at RVA 0x1aefd0 runs once per frame. Layout (tools/disasm.py):
 *   [this+GAME_PREV_NOW] previous now_ms   [this+GAME_PENDING_MS] pending game ms
 *   [this+GAME_TURN_CREDIT] turn ms available   [this+GAME_TIME_MS] game time
 * It adds (now - prev) to pending, scaled by a lag controller (0x1af0dc, table 0xa8fb44: 0.25..4.0
 * by turn backlog) and clamped to 4000; then while pending >= 25 and a network turn is available it
 * consumes one turn (25 ms of game time).
 *
 * Step mode: the clock is frozen between steps, so nothing time-paced runs. A step records
 * target = gametime + ms and starts the clock; on every GameUpdate the hook writes
 * pending = target - gametime and passes now = prev (dt 0), so the game's own turn loop consumes
 * exactly the turns owed. Only after the frame flushes actions do we publish the producer
 * budget; the host supplies that much time and stops. At the target
 * the clock freezes again and the RPC's `step` is answered. Writing pending directly is what
 * makes this exact: feeding elapsed time goes through the lag controller and the clamp (a
 * 1000 ms jump on a frozen clock became ~200 ms of pending). */
#define STEP_STALL_S 5
typedef int(__fastcall *GameUpdateFn)(void *self, void *edx, DWORD now);
GameUpdateFn GameUpdate_orig;
volatile LONG g_stepping;
DWORD g_step_target;                    /* gametime at which the step completes */
static volatile DWORD g_produce_target; /* published only after the frame flushes actions */
static DWORD g_progress_time;
const char *g_step_reason = "target";
static LONGLONG g_step_started; /* real QPC at the request, for the stall guard */
static LONG g_step_frames;
LONG g_step_frames_last;
DWORD game_time(void) {
    return g_game ? *(DWORD *)(g_game + GAME_TIME_MS) : 0;
}
int game_is_replay(void) {
    return g_game && *(DWORD *)(g_game + GAME_CONNECTION) == 1;
}
int game_is_offline(void) {
    DWORD provider = g_game ? *(DWORD *)(g_game + 0x2650) : 0;
    return provider == 0x4c4f4f50 || provider == 0x4e4f4e45;
}

void step_mode(int on) {
    g_stepping = 0;
    g_stepmode = on ? 1 : 0;
    /* pending time accumulated while free-running must not drain into the first step */
    if (on && g_game)
        *(DWORD *)(g_game + GAME_PENDING_MS) = 0;
    clock_freeze(on);
    hook_log("stepmode %d (gametime %lu)", on, game_time());
}

static void step_to(DWORD target) {
    g_step_target = target;
    g_step_started = real_qpc();
    g_progress_time = game_time();
    g_step_frames = 0;
    g_step_reason = "running";
    g_steplog_t0 = g_step_started;
    STEPLOG("request target %lu tid %lu", target, GetCurrentThreadId());
    InterlockedExchange(&g_stepping, 1);
    clock_freeze(0);
    clock_wake();
}

/* pipe thread */
const char *step_request(DWORD ms) {
    if (!g_stepmode)
        return "not in step mode";
    if (!g_game)
        return "no game yet";
    if (ms == 0 || ms > MAX_STEP_MS || ms % 25)
        return "ms must be a positive multiple of 25 (one network turn)";
    if (g_stepping)
        return "a step is in flight";
    if (ms > 0xffffffffu - game_time())
        return "game clock exhausted; reset the episode";
    DWORD target = game_time() + ms;
    if (game_is_replay()) {
        DWORD end = *(DWORD *)(g_game + GAME_REPLAY_LENGTH);
        if (target > end)
            target = end;
    }
    step_to(target);
    return "ok";
}

/* ---- the start ----------------------------------------------------------------------------
 * On the first GameUpdate, the DLL enters step mode and steps to HOLD_AT_MS, so every game
 * waits for `create_game` frozen at the same game time (the simulation
 * advances only by lockstep turns, so the state there is the same every run; catching the game
 * anywhere else gave each run a different start). create_game leaves it there (stepping) or
 * releases it (realtime). The hold has no stall guard: the map may still be loading. */
#define HOLD_AT_MS 1000
volatile LONG g_held; /* 1 once the game sits at HOLD_AT_MS */
static int g_holding;
static int g_end_requested;
static int g_restarting;

/* CGameUI registers events 11/12/14 but its destructor removes only 11. Complete
 * the native unregistration while the
 * owning UI is still alive (48 bytes/reset). */
typedef void(__fastcall *GameUiDestroyFn)(void *, void *);
static GameUiDestroyFn GameUiDestroy_orig;
static void __fastcall GameUiDestroy_hook(void *self, void *edx) {
    typedef void(__cdecl * UnregisterFn)(int, void *, void *, int);
    UnregisterFn unregister = (UnregisterFn)(g_base + RVA_EVENT_UNREGISTER);
    unregister(12, g_base + RVA_GAME_UI_EVENT12, self, -1);
    unregister(14, g_base + RVA_GAME_UI_EVENT14, self, -1);
    GameUiDestroy_orig(self, edx);
}

/* Connection observers survive reload. Native teardown clears targets during
 * dispatch but leaves their nodes in
 * seldom-used buckets. At the new hold all
 * dispatches have unwound; native rehashing returns those nodes to their
 * pool. */
static void cleanup_game_observers(BYTE *game) {
    typedef void(__fastcall * RehashFn)(BYTE *, void *, void **);
    for (unsigned connection = 0; connection < 2; connection++) {
        for (unsigned observer = 0; observer < 2; observer++) {
            BYTE *table = *(BYTE **)(game + connection * 0x448 + observer * 12 + 0x14);
            if (!table || table[0])
                continue; /* active iterators must keep their links */
            void *nodes = NULL;
            ((RehashFn)(g_base + RVA_OBSERVER_COLLECT))(table, NULL, &nodes);
            ((RehashFn)(g_base + RVA_OBSERVER_REHASH))(table, NULL, &nodes);
        }
    }
}

void step_restart(void) {
    step_mode(0);
    g_held = g_holding = g_end_requested = 0;
    g_restarting = 1;
    g_produce_target = 0;
    g_game = NULL;
    g_frame_tls = NULL;
}

static void step_finish(BYTE *t, const char *why) {
    DWORD gt = *(DWORD *)(t + GAME_TIME_MS);
    if (!InterlockedExchange(&g_stepping, 0))
        return;
    if (!g_holding && game_is_replay() && gt >= *(DWORD *)(t + GAME_REPLAY_LENGTH)) {
        rpc_set_ended();
        why = "replay_end";
    }
    clock_freeze(1);
    g_step_reason = why ? why : "target";
    stage_each_step();
    if (why)
        hook_log("step ended early (%s): gametime %lu target %lu", why, gt, g_step_target);
    g_step_frames_last = g_step_frames;
    STEPLOG("finish gametime %lu frames %ld", gt, g_step_frames);
    if (g_steplog > 0 && !g_holding)
        InterlockedDecrement(&g_steplog);
    if (g_holding) {
        cleanup_game_observers(t);
        setup_hold();
        g_holding = 0;
        g_held = 1;
        hook_log("held at gametime %lu after %ld frames", gt, g_step_frames);
    } else
        rpc_step_finished();
}

/* Finish after GameUpdate unwinds: replying inside RemovePlayer exposes a partially completed
 * final turn, so successive players can observe different times or missing opponent results. */
void step_end_after_turn(void) {
    g_end_requested = 1;
    /* No more turns in this frame: the game subtracts a turn's 25 ms before running it and loops while 25 ms are
     * pending (0x1af22f, 0x1af408), so pending 0 ends the frame after the turn that ended the game. Otherwise the
     * frame ran on through whatever turns it had been given, and the final time and state depended on how the
     * step's turns happened to be split into frames (458.4 or 458.8 s in tests/e2e/extended/determinism.py). */
    if (g_game)
        *(DWORD *)(g_game + GAME_PENDING_MS) = 0;
}

int __fastcall GameUpdate_hook(void *self, void *edx, DWORD now) {
    BYTE *t = (BYTE *)self;
    if (g_restarting) {
        if (*(DWORD *)(t + GAME_TIME_MS) > 25)
            return GameUpdate_orig(self, edx, now);
        g_restarting = 0;
        hook_log("restart: new simulation %p", t);
    }
    g_game = t;
    g_game_tid = GetCurrentThreadId();
    rpc_service_jobs();
    if (g_restarting) {
        g_game = NULL;
        return GameUpdate_orig(self, edx, now);
    }
    if (g_status == ST_ENDED && g_stepmode && !g_stepping)
        return 0;
    act_flush();
    if (!g_stepmode && !g_held) {
        step_mode(1);
        if (*(DWORD *)(t + GAME_TIME_MS) < HOLD_AT_MS) {
            g_holding = 1;
            step_to(HOLD_AT_MS);
        } else {
            setup_hold();
            g_held = 1;
        }
    }
    if (g_stepmode) {
        DWORD gt = *(DWORD *)(t + GAME_TIME_MS);
        LONG need = g_stepping ? (LONG)(g_step_target - gt) : 0;
        *(DWORD *)(t + GAME_PENDING_MS) = need > 0 ? (DWORD)need : 0;
        now = *(DWORD *)(t + GAME_PREV_NOW); /* dt 0: the game adds nothing itself */
        if (g_stepping)
            g_step_frames++;
    }
    int r = GameUpdate_orig(self, edx, now);
    if (g_stepmode && g_stepping) {
        DWORD current = *(DWORD *)(t + GAME_TIME_MS);
        /* Replay EOF and a map's EndGame can tear down the simulation without a
         * player result. Invalidate
         * the episode before replying or touching units. */
        if (!g_holding && current < g_progress_time) {
            InterlockedExchange(&g_stepping, 0);
            clock_freeze(1);
            g_held = 0;
            g_status = ST_LAUNCHED;
            g_step_reason = "simulation_closed";
            g_step_frames_last = g_step_frames;
            hook_log("simulation closed during step at %lu ms", g_progress_time);
            rpc_step_finished();
            return r;
        }
        if (current != g_progress_time) {
            g_progress_time = current;
            g_step_started = real_qpc();
        }
        g_produce_target = g_step_target;
        if (g_end_requested) {
            g_end_requested = 0;
            step_finish(t, "game_over");
        } else if ((LONG)(*(DWORD *)(t + GAME_TIME_MS) - g_step_target) >= 0)
            step_finish(t, NULL);
        else if (!g_holding && real_qpc() - g_step_started > STEP_STALL_S * g_qpc_freq)
            step_finish(t, "stalled");
    }
    return r;
}

/* Render off suppresses Direct3D presentation while preserving Warcraft's frame cleanup,
 * which releases retired
 * textures. Draw calls and scene boundaries still run. */
typedef int(__cdecl *GxPresentFn)(int flags);
static GxPresentFn GxPresent_orig;
typedef HRESULT(WINAPI *D3dPresentFn)(void *, const RECT *, const RECT *, HWND, const RGNDATA *);
static D3dPresentFn D3dPresent_orig;
static HRESULT WINAPI D3dPresent_hook(void *device, const RECT *src, const RECT *dst, HWND window,
                                      const RGNDATA *dirty) {
    if (g_render)
        return D3dPresent_orig(device, src, dst, window, dirty);
    /* Finish GPU work even without presentation, so retired buffers can be reclaimed. */
    IDirect3DQuery9 *query = NULL;
    HRESULT hr = IDirect3DDevice9_CreateQuery((IDirect3DDevice9 *)device, D3DQUERYTYPE_EVENT, &query);
    if (FAILED(hr))
        return hr;
    hr = IDirect3DQuery9_Issue(query, D3DISSUE_END);
    if (SUCCEEDED(hr)) {
        do {
            hr = IDirect3DQuery9_GetData(query, NULL, 0, D3DGETDATA_FLUSH);
            if (hr == S_FALSE)
                Sleep_orig(0);
        } while (hr == S_FALSE);
    }
    IDirect3DQuery9_Release(query);
    return hr;
}
/* The D3D device exists only once frames start. Its COM method survives map reloads. */
static void hook_d3d_present(void) {
    BYTE *gx = *(BYTE **)(g_base + RVA_GX_DEVICE);
    if (!gx || *(BYTE **)gx != g_base + RVA_GX_D3D9_VTABLE || !*(void **)(gx + GX_D3D9_DEVICE)) {
        hook_log("render off requires the Direct3D 9 backend");
        ExitProcess(1);
    }
    void *device = *(void **)(gx + GX_D3D9_DEVICE);
    void *present = (*(void ***)device)[17]; /* IDirect3DDevice9::Present */
    MH_CreateHook(present, (void *)D3dPresent_hook, (void **)&D3dPresent_orig);
    require_hook(MH_EnableHook(present), "IDirect3DDevice9::Present");
}
volatile LONG g_frames_seen; /* heartbeat: frames run only while the clock moves */
/* The game keeps a per-thread component table in a TLS slot; the frame loop installs it before
 * each frame (0x5dff0) and the visibility check reads component 0xd through it (0x26d280).
 * Between steps the clock stands still, no frame runs, and the game thread sits in its pacing
 * wait with the slot empty, so an RPC job serviced there installs the table itself. */
static void frame_seen(void) {
    InterlockedIncrement(&g_frames_seen);
    g_frame_tls = TlsGetValue(*(DWORD *)(g_base + RVA_TLS_SLOT));
    rpc_service_jobs();
}
static int __cdecl GxPresent_hook(int flags) {
    frame_seen();
    if (!g_render && !D3dPresent_orig)
        hook_d3d_present();
    return GxPresent_orig(flags);
}
/* Render off skips the frame's paint handler once the match is held, keeping GxPresent (backend
 * cleanup, FPU control) at most 30 times a real second; docs/design.md#rendering-and-input. */
typedef int(__cdecl *PaintFn)(void);
static PaintFn Paint_orig;
static LONGLONG g_last_present;
static int __cdecl Paint_hook(void) {
    if (g_render || !g_held)
        return Paint_orig();
    LONGLONG now = real_qpc();
    if (now - g_last_present < g_qpc_freq / 30) {
        frame_seen();
        return 1;
    }
    g_last_present = now;
    GxPresent_hook(*(int *)(g_base + RVA_PAINT_PRESENT_FLAGS));
    return 1;
}

/* from the frame loop's pacing wait (clock.c): the one place the game thread passes between steps */
void rpc_service_jobs_between_frames(void) {
    if (!rpc_job_pending() || !g_frame_tls)
        return;
    DWORD slot = *(DWORD *)(g_base + RVA_TLS_SLOT);
    void *prev = TlsGetValue(slot);
    TlsSetValue(slot, g_frame_tls);
    rpc_service_jobs();
    TlsSetValue(slot, prev);
}

/* ---- turn packets -----------------------------------------------------------------------
 * The offline host builds F7/0C packets at 0x557be0. Bound their time before transmission;
 * Warcraft still owns sequencing, transport, parsing and acknowledgements. Zero-time F7/48
 * action fragments stay unchanged. Realtime and remote providers use native scheduling.
 * The host thread owns the sent total and resets it with the native packet counter.
 * `pktlog <n>` traces delivery through Storm event 0x18 to the game's packet queue. */
typedef int(__cdecl *PktDeliverFn)(BYTE *msg, void *arg);
static PktDeliverFn PktDeliver_orig;
static int __cdecl PktDeliver_hook(BYTE *msg, void *arg) {
    if (msg && g_stepping)
        STEPLOG("deliver type %02x tid %lu", msg[4], GetCurrentThreadId());
    if (msg && g_pktlog > 0) {
        InterlockedDecrement(&g_pktlog);
        BYTE *d = *(BYTE **)(msg + 8);
        DWORD n = *(DWORD *)(msg + 0xc);
        char hex[3 * 24 + 1] = {0};
        DWORD k = n < 24 ? n : 24;
        for (DWORD i = 0; i < k; i++)
            _snprintf(hex + 3 * i, 4, "%02x ", d[i]);
        hook_log("pkt: type=%02x flag=%02x len=%lu gametime=%lu avail=%lu stepping=%ld target=%lu data=%s", msg[4],
                 msg[5], n, game_time(), g_game ? *(DWORD *)(g_game + GAME_TURN_CREDIT) : 0, g_stepping, g_step_target,
                 hex);
    }
    return PktDeliver_orig(msg, arg);
}

static DWORD g_sent_ms, g_produce_ms;
typedef int(__cdecl *TurnProducerFn)(BYTE *, int);
static TurnProducerFn TurnProducer_orig;
typedef BYTE *(__cdecl *TurnTimeFn)(BYTE *, const WORD *);
static TurnTimeFn TurnTime_orig;
static BYTE *__cdecl TurnTime_hook(BYTE *store, const WORD *ms) {
    if ((BYTE *)_ReturnAddress() == g_base + RVA_TURN_TIME_RET) {
        WORD credit = *ms ? (WORD)g_produce_ms : 0;
        if (g_produce_ms)
            ms = &credit;
        g_sent_ms += *ms; /* account before transport can deliver the packet */
        return TurnTime_orig(store, ms);
    }
    return TurnTime_orig(store, ms);
}
static int __cdecl TurnProducer_hook(BYTE *host, int arg) {
    if (!*(DWORD *)(host + 0x2e8))
        g_sent_ms = 0;
    g_produce_ms = 0;
    if (g_stepmode && game_is_offline()) {
        if (!g_stepping || (LONG)(g_produce_target - g_sent_ms) <= 0)
            return 1;
        g_produce_ms = g_produce_target - g_sent_ms;
        if (g_produce_ms > 400)
            g_produce_ms = 400;
        STEPLOG("produce %lu ms tid %lu", g_produce_ms, GetCurrentThreadId());
    }
    return TurnProducer_orig(host, arg);
}

void step_init_hooks(void) {
    require_hook(MH_CreateHook(g_base + RVA_GAME_UI_DESTROY, GameUiDestroy_hook, (void **)&GameUiDestroy_orig),
                 "game UI event cleanup");
    MH_STATUS a = MH_CreateHook(g_base + RVA_TURN_PRODUCER, (void *)TurnProducer_hook, (void **)&TurnProducer_orig);
    MH_STATUS b = MH_CreateHook(g_base + RVA_TURN_TIME_WRITE, (void *)TurnTime_hook, (void **)&TurnTime_orig);
    hook_log("turn producer: %s, time writer: %s", MH_StatusToString(a), MH_StatusToString(b));
    MH_CreateHook(g_base + RVA_PKT_DELIVER, (void *)PktDeliver_hook, (void **)&PktDeliver_orig);
    MH_STATUS s = MH_CreateHook(g_base + RVA_GAMEUPDATE, (void *)GameUpdate_hook, (void **)&GameUpdate_orig);
    MH_STATUS p = MH_CreateHook(g_base + RVA_GXPRESENT, (void *)GxPresent_hook, (void **)&GxPresent_orig);
    require_hook(MH_CreateHook(g_base + RVA_PAINT, (void *)Paint_hook, (void **)&Paint_orig), "frame paint");
    hook_log("GameUpdate hook: %s, GxPresent hook: %s", MH_StatusToString(s), MH_StatusToString(p));
}
