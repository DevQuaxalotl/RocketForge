#ifndef ROCKETSIM_FFI_H
#define ROCKETSIM_FFI_H

/* Generated from src/lib.rs by cbindgen. Do not edit by hand. */

#include <stdarg.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>

#define RSF_EVENT_CAR_HIT_BALL 0

#define RSF_EVENT_CAR_HIT_CAR 1

#define RSF_EVENT_CAR_PICKUP_BOOST 2

#define RSF_EVENT_BALL_HIT_WORLD 3

#define RSF_EVENT_CAR_HIT_WORLD 4

#define RSF_EVENT_CAR_LANDED 5

#define RSF_TILE_FULL 0

#define RSF_TILE_DAMAGED 1

#define RSF_TILE_BROKEN 2

/**
 * Opaque owned arena. Each handle requires exclusive access, including reads.
 * Independent handles may be used on different threads after initialization.
 */
typedef struct RsfArena RsfArena;

/**
 * Owned ball-only arena for predicting ball trajectories. Each handle requires exclusive access.
 */
typedef struct RsfBallPredictor RsfBallPredictor;

typedef struct RsfVec3 {
  float x;
  float y;
  float z;
} RsfVec3;

typedef struct RsfWheelPairConfig {
  float wheelRadius;
  float suspensionRestLength;
  struct RsfVec3 connectionPointOffset;
} RsfWheelPairConfig;

typedef struct RsfCarConfig {
  struct RsfVec3 hitboxSize;
  struct RsfVec3 hitboxPosOffset;
  struct RsfWheelPairConfig frontWheels;
  struct RsfWheelPairConfig backWheels;
  uint8_t threeWheels;
  float dodgeDeadzone;
} RsfCarConfig;

typedef struct RsfRaycastHit {
  struct RsfVec3 point;
  struct RsfVec3 normal;
  float fraction;
  /**
   * 0=world, 1=car, 2=ball, 3=dropshot tile.
   */
  uint32_t objectType;
} RsfRaycastHit;

/**
 * kind-dependent fields:
 *  CAR_HIT_BALL:     a=car_idx,            pos=contact_point, extra=extra_hit_vel
 *  CAR_HIT_CAR:      a=bumper, b=victim,   pos=contact_point, is_demo
 *  CAR_PICKUP_BOOST: a=car_idx, b=pad_idx
 *  BALL_HIT_WORLD:   pos=contact_point,    extra=contact_normal
 *  CAR_HIT_WORLD:    a=car_idx, pos=contact_point, extra=contact_normal
 */
typedef struct RsfEvent {
  uint32_t kind;
  uint32_t a;
  uint32_t b;
  uint8_t isDemo;
  uint64_t tick;
  struct RsfVec3 pos;
  struct RsfVec3 extra;
  /**
   * Ball center after the contact tick, populated for CAR_HIT_BALL.
   */
  struct RsfVec3 ballPosition;
  struct RsfRaycastHit wheels[4];
  uint8_t wheelContacts[4];
} RsfEvent;

/**
 * Axes are forward/right/up.
 * RocketSim stores Mat3A with x_axis=forward, y_axis=right, z_axis=up.
 */
typedef struct RsfRotMat {
  struct RsfVec3 forward;
  struct RsfVec3 right;
  struct RsfVec3 up;
} RsfRotMat;

typedef struct RsfPhysState {
  struct RsfVec3 pos;
  struct RsfRotMat rotMat;
  struct RsfVec3 vel;
  struct RsfVec3 angVel;
} RsfPhysState;

typedef struct RsfCarControls {
  float throttle;
  float steer;
  float pitch;
  float yaw;
  float roll;
  uint8_t jump;
  uint8_t boost;
  uint8_t handbrake;
} RsfCarControls;

typedef struct RsfCarState {
  struct RsfPhysState phys;
  struct RsfCarControls controls;
  struct RsfCarControls prevControls;
  uint8_t isOnGround;
  uint8_t wheelsWithContact[4];
  struct RsfRaycastHit wheelContacts[4];
  uint8_t hasJumped;
  uint8_t hasDoubleJumped;
  uint8_t hasFlipped;
  struct RsfVec3 flipRelTorque;
  uint32_t jumpTicks;
  uint8_t jumpHoldBroken;
  float flipTime;
  uint8_t isFlipping;
  uint8_t isJumping;
  float airTime;
  float airTimeSinceJump;
  float boost;
  float timeSinceBoosted;
  uint8_t isBoosting;
  float boostingTime;
  uint8_t isSupersonic;
  float supersonicGraceTimer;
  float handbrakeVal;
  uint8_t isAutoFlipping;
  float autoFlipTimer;
  float autoFlipTorqueScale;
  float bumpCooldownTimer;
  uint8_t hasWorldContact;
  struct RsfVec3 worldContactNormal;
  uint8_t isDemoed;
  float demoRespawnTimer;
  uint32_t bumpCooldownVictim;
  uint8_t hasBumpCooldownVictim;
  uint64_t lastExtraHitTick;
  uint8_t hasLastExtraHitTick;
} RsfCarState;

typedef struct RsfBallState {
  struct RsfPhysState phys;
  int8_t hsTarget;
  float hsSpeed;
  float hsTime;
  uint8_t dsCharge;
  float dsForce;
  int8_t dsTarget;
  uint64_t dsLastDamageTick;
  uint8_t dsHasDamageTick;
  uint64_t tickCountSinceKickoff;
} RsfBallState;

typedef struct RsfPadConfig {
  struct RsfVec3 pos;
  uint8_t isBig;
} RsfPadConfig;

typedef struct RsfPadState {
  float cooldown;
  uint8_t isActive;
} RsfPadState;

/**
 * Per-car touch and ground summary of one rsf_arena_step_snapshot call.
 * Ticks are relative to the step: 0 is the first tick the call simulated.
 */
typedef struct RsfStepCarResult {
  /**
   * 1 when this car touched the ball during the step; the other hit fields are zero otherwise.
   */
  uint8_t hitBall;
  /**
   * Step-relative tick of the car's last touch.
   */
  uint64_t hitTick;
  /**
   * Ball center at the start of the touch tick, before that tick's contact resolved.
   */
  struct RsfVec3 hitBallPos;
  /**
   * World-space contact point on the ball of the last touch.
   */
  struct RsfVec3 hitContactPoint;
  /**
   * Extra hit velocity the engine reported for the last touch (CarHitBall extra_hit_vel).
   */
  struct RsfVec3 hitExtraVel;
  /**
   * OR of isOnGround over the step-start state and the state after every tick.
   */
  uint8_t wasOnGround;
} RsfStepCarResult;

/**
 * One car-car contact the engine counted as a bump or demo.
 */
typedef struct RsfBump {
  /**
   * Car index (add order) of the attacker.
   */
  uint32_t bumper;
  /**
   * Car index (add order) of the car that was bumped or demolished.
   */
  uint32_t victim;
  /**
   * 1 when the contact demolished the victim.
   */
  uint8_t isDemo;
  /**
   * Step-relative tick of the contact.
   */
  uint64_t tick;
} RsfBump;

/**
 * Caller-allocated output of rsf_arena_step_snapshot.
 * The caller sets every pointer and count; the call fills the buffers and writes bumpsWritten and isBallScored.
 * Counts must equal the arena's: numCars = rsf_arena_num_cars, numPads = rsf_arena_num_pads,
 * numTiles = rsf_arena_num_tiles (0 outside Dropshot). A buffer pointer may be null only when its count is 0.
 */
typedef struct RsfStepOutput {
  /**
   * numCars entries: each car's full state after the last tick, in add order.
   */
  struct RsfCarState *cars;
  /**
   * numCars entries: each car's touch and ground summary for the step.
   */
  struct RsfStepCarResult *carResults;
  uint32_t numCars;
  /**
   * Ball state after the last tick. Never null.
   */
  struct RsfBallState *ball;
  /**
   * numPads entries: pad states after the last tick, in rsf_arena_get_pad_config order.
   */
  struct RsfPadState *pads;
  uint32_t numPads;
  /**
   * numTiles RSF_TILE_* values after the last tick, Blue's 70 tiles then Orange's.
   */
  uint8_t *tiles;
  uint32_t numTiles;
  /**
   * Room for bumpsCap bumps, stored in the order they happened.
   */
  struct RsfBump *bumps;
  uint32_t bumpsCap;
  /**
   * Count of every bump during the step. Can exceed bumpsCap; only the first bumpsCap are stored.
   */
  uint32_t bumpsWritten;
  /**
   * rsf_arena_is_ball_scored after the last tick.
   */
  uint8_t isBallScored;
} RsfStepOutput;

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

/**
 * ABI revision for matching generated headers and libraries.
 */
uint32_t rsf_abi_version(void);

/**
 * Thread-local message, valid until the next API call on the same thread.
 */
const char *rsf_last_error(void);

/**
 * All input pointers must be aligned/live for the call; strings are UTF-8 and NUL terminated.
 * Output buffers must be writable and disjoint from inputs and arena storage.
 * Status functions return 0 on success, -1 on error. Null free is allowed.
 */
int rsf_init(const char *path, int silent);

int rsf_is_initialized(void);

/**
 * Whether the public engine has loaded collision meshes for a mode (TheVoid needs none).
 */
int rsf_has_meshes(uint32_t mode);

/**
 * Memory mode: 0=Heavy, 1=Balanced, 2=Light. Negative seed uses a random seed.
 * Null return indicates failure; inspect rsf_last_error(). No engine memory crosses the ABI.
 */
struct RsfArena *rsf_arena_new(uint32_t mode,
                               int64_t seed,
                               uint8_t no_ball_rot,
                               uint32_t memory_mode);

void rsf_arena_free(struct RsfArena *p);

/**
 * Clone the public engine's full physics/RNG state; the two handles have independent ownership.
 */
struct RsfArena *rsf_arena_clone(struct RsfArena *p);

/**
 * Returns UINT32_MAX on failure. Null configuration selects the native Octane preset.
 */
uint32_t rsf_arena_add_car(struct RsfArena *p, uint32_t team, const struct RsfCarConfig *config);

uint32_t rsf_arena_num_cars(struct RsfArena *p);

/**
 * Events accumulate across all requested ticks. Zero ticks preserves state AND events.
 */
int rsf_arena_step(struct RsfArena *p, uint32_t ticks);

/**
 * Returns written count, or UINT32_MAX on error. Query total with cap=0 and out=NULL.
 */
uint32_t rsf_arena_get_events(struct RsfArena *p,
                              struct RsfEvent *out,
                              uint32_t cap,
                              uint32_t *total);

int rsf_arena_get_car_state(struct RsfArena *p, uint32_t idx, struct RsfCarState *out);

/**
 * State transfer follows upstream teleport semantics. Use reset_contacts for an episode reset;
 * clone is the operation for exact physics copies (state structs do not contain solver caches).
 */
int rsf_arena_set_car_state(struct RsfArena *p, uint32_t idx, const struct RsfCarState *state);

int rsf_arena_reset_contacts(struct RsfArena *p, uint32_t idx);

int rsf_arena_set_car_controls(struct RsfArena *p, uint32_t idx, const struct RsfCarControls *c);

/**
 * Applies controls for every car in add order. Count must match; validates before any mutation.
 */
int rsf_arena_set_controls(struct RsfArena *p, const struct RsfCarControls *c, uint32_t count);

int rsf_arena_respawn_car(struct RsfArena *p, uint32_t idx);

int rsf_arena_get_ball_state(struct RsfArena *p, struct RsfBallState *out);

int rsf_arena_set_ball_state(struct RsfArena *p, const struct RsfBallState *s);

int rsf_arena_reset_kickoff(struct RsfArena *p, int64_t seed);

uint64_t rsf_arena_tick_count(struct RsfArena *p);

int rsf_arena_is_ball_scored(struct RsfArena *p);

uint32_t rsf_arena_num_pads(struct RsfArena *p);

int rsf_arena_get_pad_config(struct RsfArena *p, uint32_t idx, struct RsfPadConfig *out);

int rsf_arena_get_pad_state(struct RsfArena *p, uint32_t idx, struct RsfPadState *out);

int rsf_arena_set_pad_state(struct RsfArena *p, uint32_t idx, const struct RsfPadState *s);

/**
 * Sets every car's controls, steps `ticks` ticks and writes a full snapshot into `out`.
 * `controls` holds numCars entries in add order and may be null only without cars.
 * Inputs are validated before any mutation. Zero ticks applies the controls and reports the current state.
 * Snapshot steps do not record into the rsf_arena_get_events buffer; a nonzero step clears it.
 */
int rsf_arena_step_snapshot(struct RsfArena *p,
                            const struct RsfCarControls *controls,
                            uint32_t ticks,
                            struct RsfStepOutput *out);

/**
 * Teleports a car like rsf_arena_set_car_state, but keeps the native state callers usually cannot map:
 * jumpHoldBroken, lastExtraHitTick (with hasLastExtraHitTick),
 * and the hit details of each wheel that `mapped` marks in contact and that already was.
 * wheelsWithContact comes from `mapped`, as does every other field;
 * a wheel newly marked in contact takes its hit details from `mapped`.
 */
int rsf_arena_set_car_state_preserving(struct RsfArena *p,
                                       uint32_t idx,
                                       const struct RsfCarState *mapped);

/**
 * Teleports the ball like rsf_arena_set_ball_state, but keeps the native tickCountSinceKickoff,
 * so a teleport does not restart the Hoops/Dropshot launch delay.
 * dsLastDamageTick is on rsf_arena_tick_count's clock and must not be later than it.
 */
int rsf_arena_set_ball_state_preserving(struct RsfArena *p, const struct RsfBallState *s);

/**
 * Number of Dropshot tiles: 140 in Dropshot, 0 otherwise. UINT32_MAX on failure.
 */
uint32_t rsf_arena_num_tiles(struct RsfArena *p);

/**
 * Copies rsf_arena_num_tiles RSF_TILE_* values, Blue's 70 tiles then Orange's; `cap` must hold them all.
 */
int rsf_arena_get_tiles(struct RsfArena *p,
                        uint8_t *out,
                        uint32_t cap);

/**
 * Replaces every tile state (count 140, values RSF_TILE_*, rsf_arena_get_tiles order) and updates tile collision.
 * Fails outside Dropshot.
 */
int rsf_arena_set_tiles(struct RsfArena *p,
                        const uint8_t *states,
                        uint32_t count);

/**
 * Creates a ball predictor for a mode; no_ball_rot must match the arena whose ball it predicts.
 * Null on failure. A Dropshot predictor starts every prediction from all-full tiles until rsf_ball_predictor_set_tiles.
 */
struct RsfBallPredictor *rsf_ball_predictor_new(uint32_t mode,
                                                uint8_t no_ball_rot);

void rsf_ball_predictor_free(struct RsfBallPredictor *p);

/**
 * Sets the Dropshot tiles (count 140, values RSF_TILE_*) that every later prediction starts from.
 * Fails for predictors of other modes.
 */
int rsf_ball_predictor_set_tiles(struct RsfBallPredictor *p, const uint8_t *states, uint32_t count);

/**
 * Predicts `num_ticks` ball states from `start`; out[i] is the state after i + 1 ticks.
 * Each prediction is independent: it starts from `start` and the predictor's saved tiles.
 * The predictor has no game clock, so dsLastDamageTick is an age in both `start` and `out`:
 * ticks since the ball last damaged tiles (the game's rsf_arena_tick_count minus its dsLastDamageTick).
 * Returns the first index whose state scores, -1 if none does, or INT32_MIN on failure.
 */
int32_t rsf_ball_predict(struct RsfBallPredictor *p,
                         const struct RsfBallState *start,
                         uint32_t num_ticks,
                         struct RsfBallState *out);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  /* ROCKETSIM_FFI_H */
