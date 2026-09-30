/* Observer relationships, map bounds and own scores.
 * RVAs/prototypes verified with tools/natives.py on the supported executable. */
#include "wc3hook.h"

#define N_NEUTRAL_AGGRESSIVE 0x094680
#define N_PLAYER_TEAM 0x094b80
#define N_PLAYER_ALLIANCE 0x0944c0
#define N_PLAYER_ALLY 0x099510
#define N_PLAYER_ENEMY 0x099580
#define N_PLAYER_SCORE 0x094710

/* GetWorldBounds (0x97ef0) copies the world rectangle with 0x3036f0(float out[4], 0), then wraps it in a
 * new rect handle. Only the copy is taken: creating and removing even one object reorders the engine's
 * reused object ids, and a replay's recorded orders name units by those ids. */
#define RVA_WORLD_BOUNDS_COPY 0x3036f0 /* out = {min y, min x, max y, max x} (checked against the native) */
typedef void(__cdecl *BoundsCopyFn)(float *out, int zero);
typedef int(__cdecl *N0)(void);
typedef int(__cdecl *N3)(int, int, int);

static int g_bounds_ready;
static float g_bounds[4];
void metadata_clear(void) {
    g_bounds_ready = 0;
}
static void bounds_init(void) {
    if (g_bounds_ready)
        return;
    float r[4];
    NATIVE(RVA_WORLD_BOUNDS_COPY, BoundsCopyFn)(r, 0);
    g_bounds[0] = r[1];
    g_bounds[1] = r[0];
    g_bounds[2] = r[3];
    g_bounds[3] = r[2];
    g_bounds_ready = 1;
}

static const char *SCORES[] = {"units_trained",
                               "units_killed",
                               "structures_built",
                               "structures_razed",
                               "tech_percent",
                               "food_max_produced",
                               "food_max_used",
                               "heroes_killed",
                               "items_gained",
                               "mercenaries_hired",
                               "gold_mined",
                               "gold_mined_upkeep",
                               "gold_lost_upkeep",
                               "gold_lost_tax",
                               "gold_given",
                               "gold_received",
                               "lumber_total",
                               "lumber_lost_upkeep",
                               "lumber_lost_tax",
                               "lumber_given",
                               "lumber_received",
                               "unit_total",
                               "hero_total",
                               "resource_total",
                               "total"}; /* common.j PLAYER_SCORE_* enum order, 0..24 */

void metadata_write(JW *w, int observer, const int score[25]) {
    int own = NATIVE(RVA_N_PLAYER, NativeI_I)(observer);
    int neutral = NATIVE(N_NEUTRAL_AGGRESSIVE, N0)();
    jw_key(w, "players");
    jw_open(w, '[');
    for (int p = 0; p < neutral + 4; p++) {
        int handle = NATIVE(RVA_N_PLAYER, NativeI_I)(p);
        int state = NATIVE(RVA_N_PLAYERSLOTSTATE, NativeI_I)(handle);
        if (state == 0 && p < neutral && p != observer)
            continue;
        int control = NATIVE(RVA_N_PLAYERCONTROL, NativeI_I)(handle);
        const char *relation = p == observer                                     ? "self"
                               : NATIVE(N_PLAYER_ENEMY, NativeI_II)(handle, own) ? "enemy"
                               : NATIVE(N_PLAYER_ALLY, NativeI_II)(handle, own)  ? "ally"
                                                                                 : "neutral";
        jw_open(w, '{');
        jw_key(w, "id");
        jw_int(w, p);
        jw_key(w, "kind");
        jw_string(w, p >= neutral ? "neutral" : "player");
        jw_key(w, "relation");
        jw_string(w, relation);
        jw_key(w, "team");
        jw_int(w, NATIVE(N_PLAYER_TEAM, NativeI_I)(handle));
        jw_key(w, "active");
        jw_bool(w, state == 1);
        jw_key(w, "controller");
        jw_string(w, p < 16 && player_is_agent(p) ? "agent"
                     : control == 0               ? "human"
                     : control == 1               ? "computer"
                                                  : "none");
        jw_key(w, "shares_vision");
        jw_bool(w, NATIVE(N_PLAYER_ALLIANCE, N3)(handle, own, 5));
        jw_key(w, "shares_control");
        jw_bool(w, NATIVE(N_PLAYER_ALLIANCE, N3)(handle, own, 6));
        jw_close(w, '}');
    }
    jw_close(w, ']');
    bounds_init();
    jw_key(w, "map");
    jw_open(w, '{');
    jw_key(w, "bounds");
    jw_open(w, '{');
    static const char *KEYS[] = {"min_x", "min_y", "max_x", "max_y"};
    for (int i = 0; i < 4; i++) {
        jw_key(w, KEYS[i]);
        jw_num(w, g_bounds[i]);
    }
    jw_close(w, '}');
    jw_close(w, '}');
    jw_key(w, "score");
    jw_open(w, '{');
    for (int i = 0; i < 25; i++) {
        jw_key(w, SCORES[i]);
        jw_int(w, score[i]);
    }
    jw_close(w, '}');
}
void metadata_scores(int observer, int score[25]) {
    int own = NATIVE(RVA_N_PLAYER, NativeI_I)(observer);
    for (int i = 0; i < 25; i++)
        score[i] = NATIVE(N_PLAYER_SCORE, NativeI_II)(own, i);
}
