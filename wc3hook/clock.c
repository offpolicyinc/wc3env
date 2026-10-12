/* The virtual clock: QPC, GetTickCount, FILETIME and rdtsc virtualised, Sleep and wait timeouts scaled, the wait floor.
 */
#include "wc3hook.h"
#include "clock_math.h"

/* ---- game clock ----------------------------------------------------------------------------
 * The exe imports QueryPerformanceCounter, GetTickCount and GetSystemTimeAsFileTime. All three
 * are detoured to a virtual clock that (a) runs g_speed times faster than wall time and
 * (b) stands still while the game is blocked in sync. The simulation then runs as fast as
 * the CPU allows to "catch up", which is exactly what we want. */
double g_speed = 1.0;
volatile LONG g_frozen = 0;
static CRITICAL_SECTION g_clock_lock; /* writers only (speed, freeze); readers never block */
LONGLONG g_qpc_freq;

/* Anchor of the virtual clock: virtual = virt0 + (real - real0) * speed, or a constant while
 * frozen. Published through a seqlock: the timer hooks run on every thread of the process,
 * from inside user32, the sound library and the loader, and a reader that could block on a
 * lock held by a suspended or slow thread stalls the game (seen with 8 instances booting at
 * once: the main thread waited forever under mss32 -> user32 -> QueryPerformanceCounter). */
typedef struct {
    LONGLONG real0, virt0, frozen_at;
    double speed;
    LONG frozen;
} ClockAnchor;
static ClockAnchor g_anchor;
static volatile LONG g_anchor_seq; /* odd while a writer is mid-update */

typedef BOOL(WINAPI *QpcFn)(LARGE_INTEGER *);
typedef DWORD(WINAPI *TickFn)(void);
typedef void(WINAPI *SysTimeFn)(LPFILETIME);
static QpcFn Qpc_orig;
static TickFn Tick_orig;
static SysTimeFn SysTime_orig;
static LONGLONG g_base_ft, g_base_v;

LONGLONG real_qpc(void) {
    LARGE_INTEGER li;
    Qpc_orig(&li);
    return li.QuadPart;
}

static ClockAnchor anchor_read(LONGLONG *now) {
    ClockAnchor a;
    LONG s1, s2;
    do {
        s1 = g_anchor_seq;
        _ReadWriteBarrier();
        a = g_anchor;
        if (now)
            *now = real_qpc();
        _ReadWriteBarrier();
        s2 = g_anchor_seq;
    } while ((s1 & 1) || s1 != s2);
    return a;
}
/* must hold g_clock_lock */
static void anchor_write(const ClockAnchor *a) {
    InterlockedIncrement(&g_anchor_seq);
    _ReadWriteBarrier();
    g_anchor = *a;
    _ReadWriteBarrier();
    InterlockedIncrement(&g_anchor_seq);
    g_speed = a->speed;
    g_frozen = a->frozen;
}

static LONGLONG virt_at(const ClockAnchor *a, LONGLONG now) {
    if (a->frozen)
        now = a->frozen_at;
    return a->virt0 + (LONGLONG)((now - a->real0) * a->speed);
}

static LONGLONG virt_qpc(void) {
    LONGLONG now;
    ClockAnchor a = anchor_read(&now);
    return virt_at(&a, now);
}

/* re-anchor so the virtual clock is continuous across speed changes and freezes */
static void reanchor(ClockAnchor *a) {
    LONGLONG now = a->frozen ? a->frozen_at : real_qpc();
    LONGLONG v = virt_at(a, now);
    a->real0 = now;
    a->virt0 = v;
}

void clock_set_speed(double s) {
    EnterCriticalSection(&g_clock_lock);
    ClockAnchor a = g_anchor;
    reanchor(&a);
    a.speed = s;
    anchor_write(&a);
    LeaveCriticalSection(&g_clock_lock);
}

/* Set while the clock runs, reset while it is frozen (manual reset: setting it releases every thread waiting on
 * it). Waits made while frozen sleep their full length on it, so a step's start wakes them all at once. */
static HANDLE g_running;

void clock_freeze(int on) {
    EnterCriticalSection(&g_clock_lock);
    ClockAnchor a = g_anchor;
    if (on && !a.frozen) {
        a.frozen_at = real_qpc();
        a.frozen = 1;
        anchor_write(&a);
        if (g_running)
            ResetEvent(g_running);
    } else if (!on && a.frozen) {
        reanchor(&a);
        a.frozen = 0;
        a.real0 = real_qpc();
        anchor_write(&a);
        if (g_running)
            SetEvent(g_running);
    }
    LeaveCriticalSection(&g_clock_lock);
}

static BOOL WINAPI Qpc_hook(LARGE_INTEGER *out) {
    out->QuadPart = virt_qpc();
    return TRUE;
}
static DWORD WINAPI Tick_hook(void) {
    return (DWORD)clock_units(virt_qpc(), g_qpc_freq, 1000);
}
static void WINAPI SysTime_hook(LPFILETIME ft) {
    /* wall clock offset by the same virtual delta (100 ns units) */
    LONGLONG t = g_base_ft + clock_units(virt_qpc() - g_base_v, g_qpc_freq, 10000000);
    ft->dwLowDateTime = (DWORD)t;
    ft->dwHighDateTime = (DWORD)(t >> 32);
}

/* The game's precise timer is not QPC but `rdtsc` wrapped in a 3-byte helper (RVA 0x396c80:
 * rdtsc; ret), calibrated against QPC once at startup (RVA 0x396bd0). We overwrite the helper
 * with a jump to a routine that returns virtual QPC scaled by the measured TSC/QPC ratio, so
 * both clocks stay consistent and both obey speed and freeze. */
static double g_tsc_per_qpc = 1.0;

static ULONGLONG __cdecl virt_tsc(void) {
    return (ULONGLONG)(virt_qpc() * g_tsc_per_qpc);
}
static __declspec(naked) void rdtsc_stub(void) {
    /* returns 64-bit in edx:eax like rdtsc; virt_tsc already returns in edx:eax (cdecl) */
    __asm { jmp virt_tsc }
}
static void patch_rdtsc(void) {
    LARGE_INTEGER q0, q1;
    ULONGLONG t0, t1;
    QueryPerformanceCounter(&q0);
    t0 = __rdtsc();
    Sleep(20);
    QueryPerformanceCounter(&q1);
    t1 = __rdtsc();
    g_tsc_per_qpc = (double)(t1 - t0) / (double)(q1.QuadPart - q0.QuadPart);
    BYTE *p = g_base + RVA_RDTSC_HELPER;
    DWORD old;
    if (p[0] != 0x0f || p[1] != 0x31 || p[2] != 0xc3) {
        hook_log("unsupported rdtsc helper");
        ExitProcess(1);
    }
    if (!VirtualProtect(p, 8, PAGE_EXECUTE_READWRITE, &old))
        ExitProcess(1);
    p[0] = 0xe9;
    *(DWORD *)(p + 1) = (DWORD)((BYTE *)rdtsc_stub - (p + 5));
    if (!VirtualProtect(p, 8, old, &old) || !FlushInstructionCache(GetCurrentProcess(), p, 8))
        ExitProcess(1);
    hook_log("rdtsc helper patched; tsc/qpc = %.3f", g_tsc_per_qpc);
}

/* Sleep is the frame limiter's tool; when running faster than realtime, a wall-clock sleep
 * wastes the whole speedup. Scale it down by the speed factor (never below a yield). */
typedef VOID(WINAPI *SleepFn)(DWORD);
SleepFn Sleep_orig;
static DWORD frozen_wait(DWORD n, const HANDLE *h, BOOL all, DWORD ms, BOOL alertable);
static VOID WINAPI Sleep_hook(DWORD ms) {
    ClockAnchor a = anchor_read(NULL);
    if (a.frozen && ms > 0 && ms != INFINITE && g_running) {
        frozen_wait(0, NULL, FALSE, ms, FALSE);
        return;
    }
    if (a.speed > 1.0 && ms > 0 && ms != INFINITE) {
        DWORD s = (DWORD)(ms / a.speed);
        Sleep_orig(s);
        return;
    }
    Sleep_orig(ms);
}

/* Timed waits are the other way a thread paces itself against wall time. Finite timeouts are
 * divided by the speed factor so a producer thread that sleeps in WaitForSingleObject(h, 25)
 * wakes `speed` times as often. INFINITE and 0 pass through (docs/design.md, Time and command delivery). */
volatile LONG g_wait_floor; /* `waitfloor <ms>`: least scaled wait timeout; 0 lets waits round to zero and spin */
static DWORD scale_timeout(DWORD ms) {
    ClockAnchor a = anchor_read(NULL);
    if (a.speed <= 1.0 || ms == 0 || ms == INFINITE)
        return ms;
    DWORD s = (DWORD)(ms / a.speed);
    /* Below 1 ms the wait returns at once: the turn thread then spins, producing a turn per
     * 25 virtual ms as fast as the virtual clock advances, at the cost of one busy core (2.5
     * cores per game measured at 32x). A 1 ms floor caps one game at 1000 turns/s = 25x
     * realtime, which is the right trade once several games share the machine: the pool sets
     * it, a lone game leaves it at 0. */
    return s < (DWORD)g_wait_floor ? (DWORD)g_wait_floor : s;
}
typedef DWORD(WINAPI *WFSOFn)(HANDLE, DWORD);
typedef DWORD(WINAPI *WFSOExFn)(HANDLE, DWORD, BOOL);
typedef DWORD(WINAPI *WFMOFn)(DWORD, const HANDLE *, BOOL, DWORD);
typedef DWORD(WINAPI *WFMOExFn)(DWORD, const HANDLE *, BOOL, DWORD, BOOL);
typedef DWORD(WINAPI *SleepExFn)(DWORD, BOOL);
typedef DWORD(WINAPI *MsgWaitFn)(DWORD, const HANDLE *, BOOL, DWORD, DWORD);
static WFSOFn WFSO_orig;
static WFSOExFn WFSOEx_orig;
static WFMOFn WFMO_orig;
static WFMOExFn WFMOEx_orig;
static SleepExFn SleepEx_orig;
static MsgWaitFn MsgWait_orig;

/* A timed wait made while the clock is frozen (between steps in step mode). Cut by the speed factor it would be
 * ~1 ms, and every game thread would wake ~1,000 times a second to look at a clock that does not move: ~0.15 of a
 * core per waiting game under Wine (docs/training/9-benchmarks.md). Instead it waits its full length on its own
 * handles and on g_running, which the clock starting sets: the step's start ends it at once. The clock starting
 * reads as a timeout (always a possible outcome of a timed wait; the caller looks at its clock again). Waits for
 * all handles, or with no room for one more handle, keep the old 1 ms cut. */
/* Least length of a wait made while frozen. Some game threads wait 1 ms at a time on their own (two per game, ~800
 * wakes a second each under Wine, and each wake a wineserver round trip); g_running and the threads' own handles
 * still end such a wait as soon as there is something to do. */
#define FROZEN_WAIT_MIN_MS 100

static int frozen_now(DWORD ms) {
    return ms != 0 && ms != INFINITE && g_running && anchor_read(NULL).frozen;
}
static DWORD frozen_wait(DWORD n, const HANDLE *h, BOOL all, DWORD ms, BOOL alertable) {
    HANDLE both[MAXIMUM_WAIT_OBJECTS];
    if (ms < FROZEN_WAIT_MIN_MS)
        ms = FROZEN_WAIT_MIN_MS;
    if (all || n >= MAXIMUM_WAIT_OBJECTS)
        return WFMOEx_orig(n, h, all, 1, alertable);
    for (DWORD i = 0; i < n; i++)
        both[i] = h[i];
    both[n] = g_running;
    DWORD r = WFMOEx_orig(n + 1, both, FALSE, ms, alertable);
    return r == WAIT_OBJECT_0 + n ? WAIT_TIMEOUT : r;
}
DWORD real_wait(HANDLE event, DWORD ms) {
    return WFSO_orig(event, ms);
}
/* Auto-reset: a step started or a job was posted. The frame loop's pacing wait also waits on it,
 * so the game thread starts at once instead of at the end of its timeout. */
static HANDLE g_wake;
void clock_wake(void) {
    if (g_wake)
        SetEvent(g_wake);
}
/* Between steps the frame loop sits in its pacing wait (the wrapper 0x3c2ed0, called from the
 * loop at 0x44574a). RPC jobs are serviced there, and only there: any other wait is inside game
 * code with its own state half-built. The wrapper has other callers too (0x575dcb waits there
 * with no timeout in the middle of game code; an observation served from it faulted in the Player
 * native after 15-25 minutes of realtime games), so its own caller is checked as well: the wrapper
 * pushed ebp, its two arguments and our return address, so its return address is 16 bytes up. */
static DWORD WINAPI WFSO_hook(HANDLE h, DWORD ms) {
    if ((BYTE *)_ReturnAddress() == g_base + RVA_FRAME_WAIT_RET && g_game_tid && GetCurrentThreadId() == g_game_tid &&
        *(BYTE **)((BYTE *)_AddressOfReturnAddress() + 16) == g_base + RVA_FRAME_LOOP_WAIT_RET) {
        rpc_service_jobs_between_frames();
        /* frozen: the full wait (a step's start or a posted job sets g_wake); running: cut by the speed factor */
        DWORD timeout = frozen_now(ms) ? (ms < FROZEN_WAIT_MIN_MS ? FROZEN_WAIT_MIN_MS : ms) : scale_timeout(ms);
        if (!timeout)
            return WFSO_orig(h, 0);
        HANDLE both[2] = {h, g_wake};
        DWORD r = WFMO_orig(2, both, FALSE, timeout);
        if (r != WAIT_OBJECT_0 + 1)
            return r;
        rpc_service_jobs_between_frames();
        return WAIT_TIMEOUT; /* the loop checks the clock again */
    }
    if (frozen_now(ms))
        return frozen_wait(1, &h, FALSE, ms, FALSE);
    return WFSO_orig(h, scale_timeout(ms));
}
static DWORD WINAPI WFSOEx_hook(HANDLE h, DWORD ms, BOOL a) {
    if (frozen_now(ms))
        return frozen_wait(1, &h, FALSE, ms, a);
    return WFSOEx_orig(h, scale_timeout(ms), a);
}
static DWORD WINAPI WFMO_hook(DWORD n, const HANDLE *h, BOOL all, DWORD ms) {
    if (frozen_now(ms))
        return frozen_wait(n, h, all, ms, FALSE);
    return WFMO_orig(n, h, all, scale_timeout(ms));
}
static DWORD WINAPI WFMOEx_hook(DWORD n, const HANDLE *h, BOOL all, DWORD ms, BOOL a) {
    if (frozen_now(ms))
        return frozen_wait(n, h, all, ms, a);
    return WFMOEx_orig(n, h, all, scale_timeout(ms), a);
}
static DWORD WINAPI SleepEx_hook(DWORD ms, BOOL a) {
    if (frozen_now(ms)) {
        DWORD r = frozen_wait(0, NULL, FALSE, ms, a);
        return r == WAIT_IO_COMPLETION ? r : 0;
    }
    return SleepEx_orig(scale_timeout(ms), a);
}
static DWORD WINAPI MsgWait_hook(DWORD n, const HANDLE *h, BOOL all, DWORD ms, DWORD mask) {
    if (frozen_now(ms) && !all && n < MAXIMUM_WAIT_OBJECTS) {
        HANDLE both[MAXIMUM_WAIT_OBJECTS];
        for (DWORD i = 0; i < n; i++)
            both[i] = h[i];
        both[n] = g_running;
        DWORD r = MsgWait_orig(n + 1, both, FALSE, ms < FROZEN_WAIT_MIN_MS ? FROZEN_WAIT_MIN_MS : ms, mask);
        if (r == WAIT_OBJECT_0 + n)
            return WAIT_TIMEOUT;
        return r == WAIT_OBJECT_0 + n + 1 ? WAIT_OBJECT_0 + n : r; /* input arrived: its index without ours */
    }
    return MsgWait_orig(n, h, all, scale_timeout(ms), mask);
}
static void wait_init_hooks(void) {
    MH_CreateHookApi(L"kernel32", "WaitForSingleObject", (void *)WFSO_hook, (void **)&WFSO_orig);
    MH_CreateHookApi(L"kernel32", "WaitForSingleObjectEx", (void *)WFSOEx_hook, (void **)&WFSOEx_orig);
    MH_CreateHookApi(L"kernel32", "WaitForMultipleObjects", (void *)WFMO_hook, (void **)&WFMO_orig);
    MH_CreateHookApi(L"kernel32", "WaitForMultipleObjectsEx", (void *)WFMOEx_hook, (void **)&WFMOEx_orig);
    MH_CreateHookApi(L"kernel32", "SleepEx", (void *)SleepEx_hook, (void **)&SleepEx_orig);
    MH_CreateHookApi(L"user32", "MsgWaitForMultipleObjects", (void *)MsgWait_hook, (void **)&MsgWait_orig);
}

void clock_init_hooks(void) {
    /* Waits end on the system timer tick: 15.6 ms by default on Windows, which paced every hop of a
     * step (the frozen game thread, the turn producer) at up to one tick. Wine already uses 1 ms.
     * Windows 11 ignores the request, and may move the process to efficiency cores, for a process
     * whose window is hidden; opt out of both (the call is absent before Windows 10 1709). */
    typedef BOOL(WINAPI * SetInfoFn)(HANDLE, int, void *, DWORD);
    SetInfoFn set_info = (SetInfoFn)GetProcAddress(GetModuleHandleA("kernel32"), "SetProcessInformation");
    if (set_info) {
        DWORD throttling[3] = {1, 0x1 | 0x4, 0}; /* version, EXECUTION_SPEED | IGNORE_TIMER_RESOLUTION, off */
        set_info(GetCurrentProcess(), 4 /* ProcessPowerThrottling */, throttling, sizeof throttling);
    }
    timeBeginPeriod(1);
    g_wake = CreateEventA(NULL, FALSE, FALSE, NULL);
    g_running = CreateEventA(NULL, TRUE, TRUE, NULL); /* manual reset; the clock starts running */
    LARGE_INTEGER f;
    QueryPerformanceFrequency(&f);
    g_qpc_freq = f.QuadPart;
    InitializeCriticalSection(&g_clock_lock);
    wait_init_hooks();
    MH_STATUS a = MH_CreateHookApi(L"kernel32", "QueryPerformanceCounter", (void *)Qpc_hook, (void **)&Qpc_orig);
    MH_STATUS b = MH_CreateHookApi(L"kernel32", "GetTickCount", (void *)Tick_hook, (void **)&Tick_orig);
    MH_STATUS c =
        MH_CreateHookApi(L"kernel32", "GetSystemTimeAsFileTime", (void *)SysTime_hook, (void **)&SysTime_orig);
    MH_STATUS d = MH_CreateHookApi(L"kernel32", "Sleep", (void *)Sleep_hook, (void **)&Sleep_orig);
    {
        ClockAnchor a = {0};
        a.real0 = a.virt0 = real_qpc();
        a.speed = 1.0;
        anchor_write(&a);
    }
    {
        FILETIME ft;
        SysTime_orig(&ft);
        g_base_ft = ((LONGLONG)ft.dwHighDateTime << 32) | ft.dwLowDateTime;
        g_base_v = virt_qpc();
    }
    hook_log("clock hooks: qpc=%s tick=%s systime=%s sleep=%s", MH_StatusToString(a), MH_StatusToString(b),
             MH_StatusToString(c), MH_StatusToString(d));
    patch_rdtsc();
}
