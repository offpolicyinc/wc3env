/* The RPC (docs/specs/protocol.md): JSON lines over the pipe. One request, one reply, in
 * order; `status` on every reply; a closed set of error codes. Requests are parsed with yyjson
 * (vendored, wc3hook/yyjson); replies use the streaming writer in json.h.
 *
 * Threads. The pipe thread parses and validates. Anything that reads game state or queues an
 * order runs on the game thread: the pipe thread posts a job, and the game thread runs it from
 * its hooks (the render hook every frame; the frame loop's pacing wait between steps, when no
 * frame runs). `step` starts the step and waits for step_finish to signal it. */
#include "wc3hook.h"

/* ---- status ---------------------------------------------------------------------------------- */
static const char *STATUS_NAMES[] = {"launched", "in_game", "ended"};
volatile LONG g_status = ST_LAUNCHED;
static int g_mode;               /* 0 none, 1 stepping, 2 realtime */
static int g_obs_seq[16];        /* observe sequence per player */
static char g_map[4 * MAX_PATH]; /* UTF-8 path passed to -loadfile by the launcher */
static char g_exe_hash[65];
static LONGLONG g_request_work_ticks;

void rpc_set_ended(void) {
    if (g_status == ST_IN_GAME)
        g_status = ST_ENDED;
}

/* ---- replies ----------------------------------------------------------------------------------- */
static void reply_head(JW *w, LONGLONG id, int ok) {
    jw_open(w, '{');
    jw_key(w, "protocol_version");
    jw_int(w, 1);
    jw_key(w, "id");
    jw_int(w, id);
    jw_key(w, "ok");
    jw_bool(w, ok);
    jw_key(w, "status");
    jw_string(w, STATUS_NAMES[g_status]);
}
static void reply_error(JW *w, LONGLONG id, const char *code, const char *detail) {
    jw_reset(w);
    reply_head(w, id, 0);
    jw_key(w, "error");
    jw_string(w, code);
    jw_key(w, "detail");
    jw_string(w, detail ? detail : "");
    jw_close(w, '}');
}

/* ---- jobs on the game thread ------------------------------------------------------------------- */
static CRITICAL_SECTION g_job_lock;
static HANDLE g_job_done;
static void (*g_job_fn)(void *);
static void *g_job_arg;
static volatile LONG g_job_pending;
static int g_job_failed;

int rpc_job_pending(void) {
    return g_job_pending != 0;
}
/* g_job_pending: 0 idle, 1 posted, 2 running. Called from the game thread every frame (GxPresent
 * hook), from GameUpdate, and from the frame loop's pacing wait between steps (step.c installs
 * the frame's TLS table first). */
void rpc_service_jobs(void) {
    if (InterlockedCompareExchange(&g_job_pending, 2, 1) != 1)
        return;
    void (*fn)(void *) = g_job_fn;
    void *arg = g_job_arg;
    LONGLONG started = real_qpc();
    __try {
        fn(arg);
    } __except (native_exc(GetExceptionInformation())) {
        g_job_failed = 1;
        hook_log("rpc: job faulted");
    }
    g_request_work_ticks += real_qpc() - started;
    InterlockedExchange(&g_job_pending, 0);
    SetEvent(g_job_done);
}
static int run_on_game_thread(void (*fn)(void *), void *arg, DWORD timeout_ms) {
    if (GetCurrentThreadId() == g_game_tid) {
        fn(arg);
        return 1;
    }
    EnterCriticalSection(&g_job_lock);
    ResetEvent(g_job_done);
    g_job_fn = fn;
    g_job_arg = arg;
    g_job_failed = 0;
    InterlockedExchange(&g_job_pending, 1);
    clock_wake();
    DWORD r = real_wait(g_job_done, timeout_ms);
    if (r != WAIT_OBJECT_0) {
        hook_log("rpc: the game thread did not service a job in %lu ms (gametime %lu, frozen %ld, stepping %ld)",
                 timeout_ms, game_time(), g_frozen, g_stepping);
        /* not started: withdraw it. Started: its argument is the caller's stack, so wait it out */
        if (InterlockedCompareExchange(&g_job_pending, 0, 1) != 1)
            real_wait(g_job_done, INFINITE);
    }
    int ok = r == WAIT_OBJECT_0 && !g_job_failed;
    LeaveCriticalSection(&g_job_lock);
    return ok;
}

/* ---- observe ----------------------------------------------------------------------------------- */
typedef struct {
    JW *w;
    int player;
    int ok;
} ObserveJob;
static void observe_job(void *a) {
    ObserveJob *j = (ObserveJob *)a;
    obs_write_json(j->w, j->player, ++g_obs_seq[j->player] - 1);
    j->ok = 1;
}

/* ---- binary observations: records in a shared memory mapping that the host reads in place ------
 * (obsbin.h). The mapping is Local\wc3hook-obs-<pid>, OBS_MAP_BYTES long, created at the first
 * binary request; each request overwrites it from offset 0, one observation per player, 8-byte aligned. */
static BYTE *g_obs_view;
static int obs_map_ready(void) {
    if (g_obs_view)
        return 1;
    char name[64];
    _snprintf(name, sizeof name, "Local\\wc3hook-obs-%lu", GetCurrentProcessId());
    HANDLE map = CreateFileMappingA(INVALID_HANDLE_VALUE, NULL, PAGE_READWRITE, 0, OBS_MAP_BYTES, name);
    if (!map)
        return 0;
    g_obs_view = (BYTE *)MapViewOfFile(map, FILE_MAP_WRITE, 0, 0, OBS_MAP_BYTES);
    return g_obs_view != NULL; /* the mapping lives as long as the process */
}
typedef struct {
    int n, ok;
    int players[16];
    DWORD offset[16], size[16];
} BinaryJob;
static void binary_job(void *a) {
    BinaryJob *j = (BinaryJob *)a;
    DWORD at = 0;
    for (int i = 0; i < j->n; i++) {
        int p = j->players[i];
        DWORD size = obs_write_binary(g_obs_view + at, OBS_MAP_BYTES - at, p, ++g_obs_seq[p] - 1);
        if (!size)
            return; /* over the mapping: j->ok stays 0 */
        j->offset[i] = at;
        j->size[i] = size;
        at += (size + 7) & ~7u;
    }
    j->ok = 1;
}
/* the players of a job, each at most once; NULL is an error detail */
static const char *binary_players(BinaryJob *j, const int *players, int n) {
    memset(j, 0, sizeof *j);
    for (int k = 0; k < n; k++) {
        for (int i = 0; i < j->n; i++)
            if (j->players[i] == players[k])
                return "observe lists a player twice";
        j->players[j->n++] = players[k];
    }
    return j->n && !obs_map_ready() ? "the binary observation mapping could not be created" : NULL;
}
/* step's `observe`: a list of player slots */
static const char *binary_list(yyjson_val *list, BinaryJob *j) {
    int players[16], n = 0;
    if (list && (!yyjson_is_arr(list) || yyjson_arr_size(list) > 16))
        return "observe must be a list of player slots 0..15";
    for (int k = 0; list && k < (int)yyjson_arr_size(list); k++) {
        LONGLONG p;
        if (!jr_is_int(yyjson_arr_get(list, k), &p) || p < 0 || p >= 16)
            return "observe must be a list of player slots 0..15";
        players[n++] = (int)p;
    }
    return binary_players(j, players, n);
}
static void write_binary(JW *w, const BinaryJob *j) {
    jw_key(w, "observations");
    jw_open(w, '[');
    for (int i = 0; i < j->n; i++) {
        jw_open(w, '{');
        jw_key(w, "player");
        jw_int(w, j->players[i]);
        jw_key(w, "offset");
        jw_uint(w, j->offset[i]);
        jw_key(w, "size");
        jw_uint(w, j->size[i]);
        jw_close(w, '}');
    }
    jw_close(w, ']');
}

/* ---- act --------------------------------------------------------------------------------------- */
static const char *REASONS[] = {"",           "not_your_unit", "unknown_unit", "unknown_command", "bad_arguments",
                                "queue_full", "no_build_site"};
static void act_job(void *a) {
    act_apply((ActJob *)a);
}

/* ---- the methods ------------------------------------------------------------------------------- */
static const char *STATUS_ERR(const char *m) {
    static char b[96];
    _snprintf(b, sizeof b, "%s is not valid in %s", m, STATUS_NAMES[g_status]);
    return b;
}

static void m_info(JW *w, LONGLONG id) {
    reply_head(w, id, 1);
    jw_key(w, "result");
    jw_open(w, '{');
    jw_key(w, "exe_hash");
    jw_string(w, g_exe_hash);
    jw_key(w, "game_version");
    jw_string(w, "1.29.2.9231");
    jw_key(w, "dll_version");
    jw_string(w, DLL_VERSION);
    jw_key(w, "mode");
    if (g_mode == 1)
        jw_string(w, "stepping");
    else if (g_mode == 2)
        jw_string(w, "realtime");
    else
        jw_null(w);
    jw_key(w, "game_time_ms");
    jw_uint(w, game_time());
    jw_key(w, "instance");
    jw_uint(w, GetCurrentProcessId());
    jw_key(w, "map");
    jw_string(w, g_map);
    setup_write(w);
    jw_close(w, '}');
    jw_close(w, '}');
}

static const char *RACES[] = {"random", "human", "orc", "undead", "night_elf"};
static int race_code(const char *name) {
    for (int i = 0; i < 5; i++)
        if (strcmp(name, RACES[i]) == 0)
            return i;
    return -1;
}
static int map_matches(const char *requested) {
    char path[4 * MAX_PATH];
    strcpy(path, requested);
    for (char *p = path; *p; p++)
        if (*p == '/')
            *p = '\\';
    const char *base = strrchr(g_map, '\\');
    base = base ? base + 1 : g_map;
    return g_map[0] && (_stricmp(path, g_map) == 0 || (!strchr(path, '\\') && _stricmp(path, base) == 0));
}
typedef struct {
    int mode;
    unsigned players;
    int races[16], controls[16];
    char error[160];
} CreateJob;
static void restart_job(void *a) {
    const char **error = (const char **)a;
    if (game_is_replay()) {
        *error = "replay reset requires a new process";
        return;
    }
    if (!game_is_offline()) {
        *error = "reset requires an offline game";
        return;
    }
    players_unregister();
    NATIVE(RVA_N_RESTARTGAME, NativeI_I)(0); /* RestartGame(false): schedule Warcraft's map reload */
    events_set_players(0);
    act_clear();
    players_clear();
    results_clear();
    stage_clear();
    metadata_clear();
    obs_clear();
    setup_clear();
    g_status = ST_LAUNCHED;
    g_mode = 0;
    memset(g_obs_seq, 0, sizeof g_obs_seq);
    step_restart();
}
static const char *restart_game(void) {
    const char *error = NULL;
    if (!run_on_game_thread(restart_job, &error, 5000))
        return "the game thread did not restart the game";
    if (error)
        return error;
    for (int i = 0; i < 6000 && !g_held; i++)
        Sleep_orig(5);
    return g_held ? NULL : "the restarted map did not start in 30 s";
}
static void m_reset(JW *w, LONGLONG id) {
    if (g_status == ST_LAUNCHED) {
        reply_error(w, id, "bad_status", STATUS_ERR("reset"));
        return;
    }
    const char *error = restart_game();
    if (error) {
        reply_error(w, id, "bad_status", error);
        return;
    }
    reply_head(w, id, 1);
    jw_key(w, "result");
    jw_open(w, '{');
    jw_close(w, '}');
    jw_close(w, '}');
}
/* The engine's own save: close the recorder and move Replay/TempReplay.w3g to the path. It is
 * what leaving a game runs for LastReplay.w3g, so recording ends for this episode. */
typedef int(__fastcall *SaveReplayFn)(BYTE *game, void *edx, const char *path);
typedef struct {
    char path[MAX_PATH];
    int ok;
} ReplayJob;
static void replay_job(void *a) {
    ReplayJob *j = (ReplayJob *)a;
    j->ok = g_game && ((SaveReplayFn)(g_base + RVA_SAVE_REPLAY))(g_game, NULL, j->path);
}
static void m_save_replay(JW *w, LONGLONG id, yyjson_val *params) {
    ReplayJob j = {0};
    if (g_status == ST_LAUNCHED || game_is_replay()) {
        reply_error(w, id, "bad_status", "save_replay needs a recorded game that has started");
        return;
    }
    if (!jr_str(yyjson_obj_get(params, "path"), j.path, sizeof j.path) || !j.path[0]) {
        reply_error(w, id, "bad_params", "path must be a string shorter than 260 bytes");
        return;
    }
    if (!run_on_game_thread(replay_job, &j, 5000) || !j.ok) {
        reply_error(w, id, "bad_status", "the game could not save its replay; it records once per episode");
        return;
    }
    reply_head(w, id, 1);
    jw_key(w, "result");
    jw_open(w, '{');
    jw_key(w, "game_time_ms");
    jw_uint(w, game_time());
    jw_close(w, '}');
    jw_close(w, '}');
}
static void create_job(void *a) {
    CreateJob *j = (CreateJob *)a;
    if (game_is_replay() && j->mode != 1) {
        strcpy(j->error, "replay playback requires stepping mode");
        return;
    }
    const char *setup_problem = setup_error();
    if (setup_problem) {
        _snprintf(j->error, sizeof j->error, "%s", setup_problem);
        return;
    }
    unsigned agents = 0;
    for (int p = 0; p < 16; p++) {
        if (!(j->players & (1u << p)))
            continue;
        int handle = NATIVE(RVA_N_PLAYER, NativeI_I)(p);
        int race = NATIVE(RVA_N_PLAYERRACE, NativeI_I)(handle);
        int control = NATIVE(RVA_N_PLAYERCONTROL, NativeI_I)(handle);
        int state = NATIVE(RVA_N_PLAYERSLOTSTATE, NativeI_I)(handle);
        /* a replay recorded by an observer (tournament replays) reads every player as left (2) */
        if ((state != 1 && !(state == 2 && game_is_replay())) || race < 1 || race > 4 ||
            (j->races[p] >= 0 && race != j->races[p] && !(j->races[p] == 0 && setup_random_race(p))) ||
            (control != j->controls[p] && !(control == 1 && j->controls[p] == 0 && player_prepared_agent(p)))) {
            _snprintf(j->error, sizeof j->error,
                      "player %d setup is unsupported: loaded race=%d controller=%d state=%d; requested race=%d "
                      "controller=%d",
                      p, race, control, state, j->races[p], j->controls[p]);
            return;
        }
        j->races[p] = race;
        if (j->controls[p] == 0)
            agents |= 1u << p;
    }
    const char *error = act_configure(agents);
    if (error) {
        _snprintf(j->error, sizeof j->error, "%s", error);
        return;
    }
    memset(g_obs_seq, 0, sizeof g_obs_seq);
    events_set_players(j->players);
    g_mode = j->mode;
    g_status = ST_IN_GAME;
    if (g_mode == 2)
        step_mode(0); /* release only after the event readers are ready */
}

static void m_create_game(JW *w, LONGLONG id, yyjson_val *params) {
    char map[4 * MAX_PATH], mode[16];
    if (g_status == ST_IN_GAME) {
        reply_error(w, id, "bad_status", STATUS_ERR("create_game"));
        return;
    }
    if (!jr_str(yyjson_obj_get(params, "map"), map, sizeof map)) {
        reply_error(w, id, "bad_params", "map must be a string");
        return;
    }
    if (!map[0]) {
        reply_error(w, id, "bad_params", "map path must not be empty");
        return;
    }
    yyjson_val *players = yyjson_obj_get(params, "players");
    if (players == NULL || !yyjson_is_arr(players)) {
        reply_error(w, id, "bad_params", "players must be a list");
        return;
    }
    if (!(int)yyjson_arr_size(players)) {
        reply_error(w, id, "bad_params", "players must not be empty");
        return;
    }
    CreateJob j = {0};
    for (int k = 0; k < (int)yyjson_arr_size(players); k++) {
        yyjson_val *pl = yyjson_arr_get(players, k);
        LONGLONG slot;
        char race[32], control[32];
        if (!yyjson_is_obj(pl) || !jr_is_int(yyjson_obj_get(pl, "slot"), &slot) || slot < 0 || slot >= 16 ||
            !jr_str(yyjson_obj_get(pl, "control"), control, sizeof control)) {
            reply_error(w, id, "bad_params", "players must be a list of {slot 0..15, race, control}");
            return;
        }
        yyjson_val *rt = yyjson_obj_get(pl, "race");
        int rc = -1;
        if (rt != NULL && (!jr_str(rt, race, sizeof race) || (rc = race_code(race)) < 0)) {
            reply_error(w, id, "bad_params", "unknown race");
            return;
        }
        int cc = strcmp(control, "agent") == 0 ? 0 : strcmp(control, "computer") == 0 ? 1 : -1;
        if (cc < 0 || (j.players & (1u << slot))) {
            reply_error(w, id, "bad_params", "unknown control or duplicate player slot");
            return;
        }
        j.players |= 1u << slot;
        j.races[slot] = rc;
        j.controls[slot] = cc;
    }
    if (!jr_str(yyjson_obj_get(params, "mode"), mode, sizeof mode) ||
        (strcmp(mode, "stepping") && strcmp(mode, "realtime"))) {
        reply_error(w, id, "bad_params", "mode must be stepping or realtime");
        return;
    }
    if (!map_matches(map)) {
        reply_error(w, id, "unsupported_config", "requested map differs from the file loaded at launch");
        return;
    }
    if (g_status == ST_ENDED) {
        const char *error = restart_game();
        if (error) {
            reply_error(w, id, "bad_status", error);
            return;
        }
    }
    /* Launch and reload hold at 1000 ms; configure before releasing a realtime game. */
    for (int i = 0; i < 1200 && !g_held; i++)
        Sleep_orig(100);
    if (!g_held) {
        reply_error(w, id, "bad_status", "the map did not start");
        return;
    }
    j.mode = strcmp(mode, "stepping") == 0 ? 1 : 2;
    if (!run_on_game_thread(create_job, &j, 5000)) {
        reply_error(w, id, "bad_status", "the game thread did not initialize the game");
        return;
    }
    if (j.error[0]) {
        reply_error(w, id, "unsupported_config", j.error);
        return;
    }
    reply_head(w, id, 1);
    jw_key(w, "result");
    jw_open(w, '{');
    jw_key(w, "game_time_ms");
    jw_uint(w, game_time());
    jw_key(w, "map");
    jw_string(w, g_map);
    setup_write(w);
    jw_key(w, "players");
    jw_open(w, '[');
    for (int p = 0; p < 16; p++)
        if (j.players & (1u << p)) {
            jw_open(w, '{');
            jw_key(w, "slot");
            jw_int(w, p);
            jw_key(w, "race");
            jw_string(w, RACES[j.races[p]]);
            jw_key(w, "control");
            jw_string(w, j.controls[p] == 0 ? "agent" : "computer");
            jw_close(w, '}');
        }
    jw_close(w, ']');
    jw_close(w, '}');
    jw_close(w, '}');
}

static HANDLE g_stepped; /* signalled by step_finish */
/* The binary observations the running step writes as it finishes, on the game thread before it signals:
 * no second trip to the game thread. Static, so a step that finishes after its request timed out writes
 * nowhere stale. */
static BinaryJob g_step_binary;
void rpc_step_finished(void) {
    if (g_step_binary.n && g_status != ST_LAUNCHED)
        binary_job(&g_step_binary);
    if (g_stepped)
        SetEvent(g_stepped);
}

static void m_step(JW *w, LONGLONG id, yyjson_val *params) {
    LONGLONG ms;
    if (g_status != ST_IN_GAME) {
        reply_error(w, id, "bad_status", STATUS_ERR("step"));
        return;
    }
    if (g_mode != 1) {
        reply_error(w, id, "not_stepping", "step is stepping mode only");
        return;
    }
    if (!jr_is_int(yyjson_obj_get(params, "ms"), &ms) || ms <= 0 || ms > MAX_STEP_MS || ms % 25) {
        reply_error(w, id, "bad_params", "ms must be a multiple of 25 in 25..60000");
        return;
    }
    const char *invalid = binary_list(yyjson_obj_get(params, "observe"), &g_step_binary);
    if (invalid) {
        reply_error(w, id, "bad_params", invalid);
        return;
    }
    DWORD before = game_time();
    ResetEvent(g_stepped);
    const char *err = step_request((DWORD)ms);
    if (strcmp(err, "ok") != 0) {
        reply_error(w, id, "bad_status", err);
        return;
    }
    if (real_wait(g_stepped, 120000) != WAIT_OBJECT_0) {
        reply_error(w, id, "bad_status", "the step did not finish in 120 s");
        return;
    }
    if (!strcmp(g_step_reason, "simulation_closed")) {
        reply_error(w, id, "bad_status", "simulation closed during step; launch a new process");
        return;
    }
    reply_head(w, id, 1);
    jw_key(w, "result");
    jw_open(w, '{');
    jw_key(w, "game_time_ms");
    jw_uint(w, game_time());
    jw_key(w, "frames");
    jw_int(w, g_step_frames_last);
    jw_key(w, "elapsed_ms");
    jw_uint(w, game_time() - before);
    jw_key(w, "reason");
    jw_string(w, g_step_reason);
    if (g_step_binary.n) {
        if (!g_step_binary.ok) {
            reply_error(w, id, "bad_status", "the binary observations did not fit the mapping");
            return;
        }
        write_binary(w, &g_step_binary);
    }
    jw_close(w, '}');
    jw_close(w, '}');
}

static int player_param(yyjson_val *params, int required, LONGLONG *out) {
    yyjson_val *p = yyjson_obj_get(params, "player");
    if (p == NULL) {
        if (required)
            return 0;
        *out = 0;
        return 1;
    }
    return jr_is_int(p, out) && *out >= 0 && *out < 16;
}

static void m_observe(JW *w, LONGLONG id, yyjson_val *params) {
    LONGLONG player;
    if (g_status == ST_LAUNCHED) {
        reply_error(w, id, "bad_status", STATUS_ERR("observe"));
        return;
    }
    if (!player_param(params, 0, &player)) {
        reply_error(w, id, "bad_params", "player must be a slot 0..15");
        return;
    }
    yyjson_val *format_val = yyjson_obj_get(params, "format");
    const char *format = format_val ? yyjson_get_str(format_val) : "json";
    if (!format || (strcmp(format, "json") != 0 && strcmp(format, "binary") != 0)) {
        reply_error(w, id, "bad_params", "format must be json or binary");
        return;
    }
    if (!strcmp(format, "binary")) {
        BinaryJob binary;
        int one = (int)player;
        const char *error = binary_players(&binary, &one, 1);
        if (error || !run_on_game_thread(binary_job, &binary, 5000) || !binary.ok) {
            reply_error(w, id, "bad_status", error ? error : "the binary observation did not fit or was not written");
            return;
        }
        reply_head(w, id, 1);
        jw_key(w, "result");
        jw_open(w, '{');
        write_binary(w, &binary);
        jw_close(w, '}');
        jw_close(w, '}');
        return;
    }
    reply_head(w, id, 1);
    jw_key(w, "result");
    ObserveJob j = {w, (int)player, 0};
    if (!run_on_game_thread(observe_job, &j, 5000) || !j.ok) {
        reply_error(w, id, "bad_status", "the game thread did not answer");
        return;
    }
    jw_close(w, '}');
}

static void m_act(JW *w, LONGLONG id, yyjson_val *params) {
    ActJob job = {0};
    LONGLONG player;
    if (g_status != ST_IN_GAME) {
        reply_error(w, id, "bad_status", STATUS_ERR("act"));
        return;
    }
    if (game_is_replay()) {
        reply_error(w, id, "bad_status", "replay actions come from the recording");
        return;
    }
    if (!player_param(params, 1, &player)) {
        reply_error(w, id, "bad_params", "player must be a slot 0..15");
        return;
    }
    if (!player_is_agent((int)player)) {
        reply_error(w, id, "bad_params", "player is not configured as an agent");
        return;
    }
    yyjson_val *arr = yyjson_obj_get(params, "actions");
    if (arr == NULL || !yyjson_is_arr(arr)) {
        reply_error(w, id, "bad_params", "actions must be a list");
        return;
    }
    job.player = (int)player;
    job.n = (int)yyjson_arr_size(arr);
    job.items = calloc(job.n, sizeof *job.items);
    if (job.n && !job.items) {
        reply_error(w, id, "bad_status", "could not allocate action batch");
        return;
    }
    for (int k = 0; k < job.n; k++) {
        yyjson_val *a = yyjson_arr_get(arr, k);
        ActItem *it = &job.items[k];
        char detail[64];
        _snprintf(detail, sizeof detail, "actions[%d] must be {unit_id, command, arguments}", k);
        if (!yyjson_is_obj(a) || !jr_is_int(yyjson_obj_get(a, "unit_id"), &it->unit_id) ||
            !jr_str(yyjson_obj_get(a, "command"), it->command, sizeof it->command)) {
            reply_error(w, id, "bad_params", detail);
            goto cleanup;
        }
        yyjson_val *args = yyjson_obj_get(a, "arguments");
        if (args != NULL && !yyjson_is_obj(args)) {
            reply_error(w, id, "bad_params", detail);
            goto cleanup;
        }
        yyjson_val *queued = yyjson_obj_get(args, "queued");
        if (queued && !yyjson_is_bool(queued))
            it->reason = RJ_BAD_ARGS;
        it->queued = yyjson_get_bool(queued);
        yyjson_val *auto_place = yyjson_obj_get(args, "auto_place");
        if (auto_place && (!yyjson_is_bool(auto_place) || strcmp(it->command, "build")))
            it->reason = RJ_BAD_ARGS;
        it->auto_place = yyjson_get_bool(auto_place);
        double x, y;
        LONGLONG t, slot;
        if (jr_is_num(yyjson_obj_get(args, "x"), &x) && jr_is_num(yyjson_obj_get(args, "y"), &y)) {
            it->has_xy = 1;
            it->x = x;
            it->y = y;
        } else if (yyjson_obj_get(args, "x") != NULL || yyjson_obj_get(args, "y") != NULL)
            it->reason = RJ_BAD_ARGS;
        it->target = jr_is_int(yyjson_obj_get(args, "target_id"), &t) ? t : 0;
        it->shop = jr_is_int(yyjson_obj_get(args, "shop_id"), &t) ? t : 0;
        it->slot = jr_is_int(yyjson_obj_get(args, "slot"), &slot) && slot >= 0 && slot <= 5 ? (int)slot : -1;
        jr_str(yyjson_obj_get(args, "type_id"), it->type_id, sizeof it->type_id);
        jr_str(yyjson_obj_get(args, "item_type_id"), it->item_type, sizeof it->item_type);
        if (!strcmp(it->command, "learn"))
            jr_str(yyjson_obj_get(args, "ability_id"), it->type_id, sizeof it->type_id);
        jr_str(yyjson_obj_get(args, "order"), it->order, sizeof it->order);
        if ((it->has_xy && (fabs(it->x) > 1000000 || fabs(it->y) > 1000000)) ||
            (yyjson_obj_get(args, "target_id") != NULL &&
             (!jr_is_int(yyjson_obj_get(args, "target_id"), &t) || t <= 0 || t > 0xffffffffLL)) ||
            (yyjson_obj_get(args, "shop_id") != NULL &&
             (!jr_is_int(yyjson_obj_get(args, "shop_id"), &t) || t <= 0 || t > 0xffffffffLL)) ||
            (yyjson_obj_get(args, "slot") != NULL && it->slot < 0))
            it->reason = RJ_BAD_ARGS;
        const char *fields[] = {"type_id", "item_type_id", "ability_id", "order"};
        for (int q = 0; q < 4; q++) {
            yyjson_val *token = yyjson_obj_get(args, fields[q]);
            char value[32];
            if (token != NULL && (!jr_str(token, value, sizeof value) || (q < 3 && strlen(value) != 4)))
                it->reason = RJ_BAD_ARGS;
        }
        static const char *KNOWN[] = {"move",     "stop",  "attack", "smart",    "harvest",   "build",  "train",
                                      "research", "learn", "cast",   "use_item", "drop_item", "select", "buy",
                                      "revive"};
        int known = 0;
        for (size_t q = 0; q < sizeof KNOWN / sizeof KNOWN[0]; q++)
            if (strcmp(it->command, KNOWN[q]) == 0)
                known = 1;
        if (!known) {
            it->reason = RJ_UNKNOWN_COMMAND;
            continue;
        }
        if (it->queued &&
            (!strcmp(it->command, "train") || !strcmp(it->command, "research") || !strcmp(it->command, "learn") ||
             !strcmp(it->command, "select") || !strcmp(it->command, "buy") || !strcmp(it->command, "revive") ||
             !strcmp(it->command, "drop_item")))
            it->reason = RJ_BAD_ARGS;
    }
    if (!run_on_game_thread(act_job, &job, 5000)) {
        reply_error(w, id, "bad_status", "the game thread did not answer");
        goto cleanup;
    }
    reply_head(w, id, 1);
    jw_key(w, "result");
    jw_open(w, '{');
    jw_key(w, "rejected");
    jw_open(w, '[');
    for (int k = 0; k < job.n; k++)
        if (job.items[k].reason) {
            jw_open(w, '{');
            jw_key(w, "index");
            jw_int(w, k);
            jw_key(w, "reason");
            jw_string(w, REASONS[job.items[k].reason]);
            jw_close(w, '}');
        }
    jw_close(w, ']');
    jw_key(w, "placements");
    jw_open(w, '[');
    for (int k = 0; k < job.n; k++) {
        ActItem *a = &job.items[k];
        if (a->auto_place && !a->reason) {
            jw_open(w, '{');
            jw_key(w, "index");
            jw_int(w, k);
            jw_key(w, "x");
            jw_num(w, a->x);
            jw_key(w, "y");
            jw_num(w, a->y);
            jw_close(w, '}');
        }
    }
    jw_close(w, ']');
    jw_close(w, '}');
    jw_close(w, '}');
cleanup:
    free(job.items);
}

typedef struct {
    StageOp op;
    yyjson_val *args;
    JW *w;
    const char *err;
} StageJob;
static void stage_job(void *a) {
    StageJob *j = (StageJob *)a;
    j->err = j->op(j->args, j->w);
}

static void m_debug(JW *w, LONGLONG id, yyjson_val *params) {
    char op[32];
    if (g_status != ST_IN_GAME) {
        reply_error(w, id, "bad_status", STATUS_ERR("debug"));
        return;
    }
    if (!jr_str(yyjson_obj_get(params, "op"), op, sizeof op)) {
        reply_error(w, id, "bad_params", "op must be a string");
        return;
    }
    yyjson_val *args = yyjson_obj_get(params, "args");
    if (args != NULL && !yyjson_is_obj(args)) {
        reply_error(w, id, "bad_params", "args must be an object");
        return;
    }
    if (strcmp(op, "speed") == 0) {
        double s;
        if (!jr_is_num(yyjson_obj_get(args, "factor"), &s) || s <= 0 || s > MAX_SPEED) {
            reply_error(w, id, "bad_params", "speed needs finite args.factor in (0, 2048]");
            return;
        }
        clock_set_speed(s);
    } else if (strcmp(op, "waitfloor") == 0) {
        LONGLONG f;
        if (!jr_is_int(yyjson_obj_get(args, "ms"), &f) || f < 0 || f > MAX_WAIT_FLOOR_MS) {
            reply_error(w, id, "bad_params", "waitfloor needs args.ms in 0..1000");
            return;
        }
        g_wait_floor = (LONG)f;
    } else if (strcmp(op, "render") == 0) {
        LONGLONG on;
        if (!jr_is_int(yyjson_obj_get(args, "on"), &on)) {
            reply_error(w, id, "bad_params", "render needs args.on");
            return;
        }
        g_render = on ? 1 : 0;
    } else if (stage_op(op)) { /* a staging native on the game thread (wc3hook/stage.c) */
        StageJob j = {stage_op(op), args, w, NULL};
        reply_head(w, id, 1);
        jw_key(w, "result");
        jw_open(w, '{');
        int answered = run_on_game_thread(stage_job, &j, 5000);
        hook_log("rpc: debug %s -> %s", op, !answered ? "no answer" : j.err ? j.err : "ok");
        if (!answered) {
            reply_error(w, id, "bad_status", "the game thread did not answer");
            return;
        }
        if (j.err) {
            reply_error(w, id, "bad_params", j.err);
            return;
        }
        jw_close(w, '}');
        jw_close(w, '}');
        return;
    } else {
        reply_error(w, id, "bad_params", "unknown debug op");
        return;
    }
    reply_head(w, id, 1);
    jw_key(w, "result");
    jw_open(w, '{');
    jw_close(w, '}');
    jw_close(w, '}');
}

static int g_quit;
static void m_quit(JW *w, LONGLONG id) {
    reply_head(w, id, 1);
    jw_key(w, "result");
    jw_open(w, '{');
    jw_close(w, '}');
    jw_close(w, '}');
    g_quit = 1;
}

/* ---- entry: one request line -> one reply line ------------------------------------------------ */
void rpc_handle(const char *line) {
    static JW w;
    LONGLONG started = real_qpc();
    g_request_work_ticks = 0;
    yyjson_doc *doc = jr_read(line);
    yyjson_val *root = yyjson_doc_get_root(doc);
    jw_reset(&w);
    LONGLONG id = 0;
    int have_id = jr_is_int(yyjson_obj_get(root, "id"), &id);
    if (!yyjson_is_obj(root))
        reply_error(&w, 0, "bad_request", "not a JSON object within request limits");
    else {
        LONGLONG ver;
        char method[32];
        if (!jr_is_int(yyjson_obj_get(root, "protocol_version"), &ver) || ver != 1)
            reply_error(&w, id, "bad_version", "protocol_version must be 1");
        else if (!have_id)
            reply_error(&w, id, "bad_request", "id must be an integer");
        else if (!jr_str(yyjson_obj_get(root, "method"), method, sizeof method))
            reply_error(&w, id, "bad_request", "method must be a string");
        else {
            yyjson_val *params = yyjson_obj_get(root, "params");
            if (params && !yyjson_is_obj(params))
                reply_error(&w, id, "bad_params", "params must be an object");
            else if (strcmp(method, "info") == 0)
                m_info(&w, id);
            else if (strcmp(method, "create_game") == 0)
                m_create_game(&w, id, params);
            else if (strcmp(method, "reset") == 0)
                m_reset(&w, id);
            else if (strcmp(method, "save_replay") == 0)
                m_save_replay(&w, id, params);
            else if (strcmp(method, "step") == 0)
                m_step(&w, id, params);
            else if (strcmp(method, "observe") == 0)
                m_observe(&w, id, params);
            else if (strcmp(method, "act") == 0)
                m_act(&w, id, params);
            else if (strcmp(method, "debug") == 0)
                m_debug(&w, id, params);
            else if (strcmp(method, "quit") == 0)
                m_quit(&w, id);
            else
                reply_error(&w, id, "unknown_method", method);
        }
    }
    yyjson_doc_free(doc);
    /* Each handler produced one complete object. Append transport metadata outside
     * result so observations and
     * gameplay remain independent of wall-clock timing. */
    w.p[--w.n] = 0;
    jw_key(&w, "timing_ms");
    jw_open(&w, '{');
    jw_key(&w, "server");
    jw_num(&w, 1000.0 * (real_qpc() - started) / g_qpc_freq);
    jw_key(&w, "game_thread");
    jw_num(&w, 1000.0 * g_request_work_ticks / g_qpc_freq);
    jw_close(&w, '}');
    jw_close(&w, '}');
    pipe_send(w.p);
    if (g_quit) {
        hook_log("rpc: quit");
        Sleep_orig(50);
        ExitProcess(0);
    }
}

void rpc_init(void) {
    wchar_t map[MAX_PATH];
    DWORD n = GetEnvironmentVariableW(L"WC3HOOK_MAP", map, MAX_PATH);
    if (n && n < MAX_PATH)
        WideCharToMultiByte(CP_UTF8, WC_ERR_INVALID_CHARS, map, -1, g_map, sizeof g_map, NULL, NULL);
    GetEnvironmentVariableA("WC3HOOK_EXE_HASH", g_exe_hash, sizeof g_exe_hash);
    InitializeCriticalSection(&g_job_lock);
    g_job_done = CreateEventA(NULL, TRUE, FALSE, NULL);
    g_stepped = CreateEventA(NULL, TRUE, FALSE, NULL);
    if (!g_job_done || !g_stepped) {
        hook_log("RPC event initialization failed");
        ExitProcess(1);
    }
}
