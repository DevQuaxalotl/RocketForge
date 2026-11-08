#include "rocketsim_ffi.h"
#include <stddef.h>

_Static_assert(sizeof(struct RsfVec3) == 3 * sizeof(float), "C vector layout");
_Static_assert(sizeof(((struct RsfCarState*)0)->jumpTicks) == sizeof(uint32_t), "Integer jump ticks");

uint32_t rocketsim_c_header_smoke(struct RsfArena* arena) {
    struct RsfBallState ball = {0};
    return rsf_arena_get_ball_state(arena, &ball) == 0 ? rsf_abi_version() : 0;
}
