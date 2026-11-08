//! Standalone C ABI for public RocketSim v3. See include/rocketsim_ffi.h and README.md.
//! Adapted from the initial RocketForge v3 bridge; no private engine APIs are required.
#![deny(unsafe_op_in_unsafe_fn)]
use glam::{Mat3A, Vec3A};
use rocketsim::{
    Arena, ArenaConfig, ArenaEvent, BallState, BoostPadState, CarBodyConfig, CarControls, CarState,
    GameMode, RaycastHitInfo, Team, TileDamageState, TileStates, UserInfoType, consts,
};
use std::cell::RefCell;
use std::ffi::CStr;
#[cfg(test)]
use std::ffi::CString;
use std::os::raw::{c_char, c_int};
use std::panic::{AssertUnwindSafe, catch_unwind};

#[repr(C)]
#[derive(Clone, Copy, Default, Debug, PartialEq)]
pub struct RsfVec3 {
    pub x: f32,
    pub y: f32,
    pub z: f32,
}

impl From<Vec3A> for RsfVec3 {
    fn from(v: Vec3A) -> Self {
        Self {
            x: v.x,
            y: v.y,
            z: v.z,
        }
    }
}
impl From<RsfVec3> for Vec3A {
    fn from(v: RsfVec3) -> Self {
        Vec3A::new(v.x, v.y, v.z)
    }
}

/// Axes are forward/right/up.
/// RocketSim stores Mat3A with x_axis=forward, y_axis=right, z_axis=up.
#[repr(C)]
#[derive(Clone, Copy, Default, Debug, PartialEq)]
pub struct RsfRotMat {
    pub forward: RsfVec3,
    pub right: RsfVec3,
    pub up: RsfVec3,
}

impl From<Mat3A> for RsfRotMat {
    fn from(m: Mat3A) -> Self {
        Self {
            forward: m.x_axis.into(),
            right: m.y_axis.into(),
            up: m.z_axis.into(),
        }
    }
}
impl From<RsfRotMat> for Mat3A {
    fn from(m: RsfRotMat) -> Self {
        Mat3A::from_cols(m.forward.into(), m.right.into(), m.up.into())
    }
}

#[repr(C)]
#[derive(Clone, Copy, Default, Debug, PartialEq)]
pub struct RsfPhysState {
    pub pos: RsfVec3,
    pub rot_mat: RsfRotMat,
    pub vel: RsfVec3,
    pub ang_vel: RsfVec3,
}

#[repr(C)]
#[derive(Clone, Copy, Default, Debug, PartialEq)]
pub struct RsfCarControls {
    pub throttle: f32,
    pub steer: f32,
    pub pitch: f32,
    pub yaw: f32,
    pub roll: f32,
    pub jump: u8,
    pub boost: u8,
    pub handbrake: u8,
}

impl From<CarControls> for RsfCarControls {
    fn from(c: CarControls) -> Self {
        Self {
            throttle: c.throttle,
            steer: c.steer,
            pitch: c.pitch,
            yaw: c.yaw,
            roll: c.roll,
            jump: c.jump as u8,
            boost: c.boost as u8,
            handbrake: c.handbrake as u8,
        }
    }
}
impl From<RsfCarControls> for CarControls {
    fn from(c: RsfCarControls) -> Self {
        Self {
            throttle: c.throttle,
            steer: c.steer,
            pitch: c.pitch,
            yaw: c.yaw,
            roll: c.roll,
            jump: c.jump != 0,
            boost: c.boost != 0,
            handbrake: c.handbrake != 0,
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Default, Debug, PartialEq)]
pub struct RsfWheelPairConfig {
    pub wheel_radius: f32,
    pub suspension_rest_length: f32,
    pub connection_point_offset: RsfVec3,
}

#[repr(C)]
#[derive(Clone, Copy, Default, Debug, PartialEq)]
pub struct RsfCarConfig {
    pub hitbox_size: RsfVec3,
    pub hitbox_pos_offset: RsfVec3,
    pub front_wheels: RsfWheelPairConfig,
    pub back_wheels: RsfWheelPairConfig,
    pub three_wheels: u8,
    pub dodge_deadzone: f32,
}

impl From<&RsfCarConfig> for CarBodyConfig {
    fn from(c: &RsfCarConfig) -> Self {
        rocketsim::CarBodyConfig {
            hitbox_size: c.hitbox_size.into(),
            hitbox_pos_offset: c.hitbox_pos_offset.into(),
            front_wheels: rocketsim::WheelPairConfig {
                wheel_radius: c.front_wheels.wheel_radius,
                suspension_rest_length: c.front_wheels.suspension_rest_length,
                connection_point_offset: c.front_wheels.connection_point_offset.into(),
            },
            back_wheels: rocketsim::WheelPairConfig {
                wheel_radius: c.back_wheels.wheel_radius,
                suspension_rest_length: c.back_wheels.suspension_rest_length,
                connection_point_offset: c.back_wheels.connection_point_offset.into(),
            },
            three_wheels: c.three_wheels != 0,
            dodge_deadzone: c.dodge_deadzone,
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RsfCarState {
    pub phys: RsfPhysState,
    pub controls: RsfCarControls,
    pub prev_controls: RsfCarControls,
    pub is_on_ground: u8,
    pub wheels_with_contact: [u8; 4],
    pub wheel_contacts: [RsfRaycastHit; 4],
    pub has_jumped: u8,
    pub has_double_jumped: u8,
    pub has_flipped: u8,
    pub flip_rel_torque: RsfVec3,
    pub jump_ticks: u32,
    pub jump_hold_broken: u8,
    pub flip_time: f32,
    pub is_flipping: u8,
    pub is_jumping: u8,
    pub air_time: f32,
    pub air_time_since_jump: f32,
    pub boost: f32,
    pub time_since_boosted: f32,
    pub is_boosting: u8,
    pub boosting_time: f32,
    pub is_supersonic: u8,
    pub supersonic_grace_timer: f32,
    pub handbrake_val: f32,
    pub is_auto_flipping: u8,
    pub auto_flip_timer: f32,
    pub auto_flip_torque_scale: f32,
    pub bump_cooldown_timer: f32,
    pub has_world_contact: u8,
    pub world_contact_normal: RsfVec3,
    pub is_demoed: u8,
    pub demo_respawn_timer: f32,
    pub bump_cooldown_victim: u32,
    pub has_bump_cooldown_victim: u8,
    pub last_extra_hit_tick: u64,
    pub has_last_extra_hit_tick: u8,
}

impl From<&CarState> for RsfCarState {
    fn from(s: &CarState) -> Self {
        Self {
            phys: RsfPhysState {
                pos: s.phys.pos.into(),
                rot_mat: s.phys.rot_mat.into(),
                vel: s.phys.vel.into(),
                ang_vel: s.phys.ang_vel.into(),
            },
            controls: s.controls.into(),
            prev_controls: s.prev_controls.into(),
            is_on_ground: s.is_on_ground as u8,
            wheels_with_contact: s.wheels_with_contact.map(|b| b.is_some() as u8),
            wheel_contacts: s.wheels_with_contact.map(|b| b.unwrap_or_default().into()),
            has_jumped: s.has_jumped as u8,
            has_double_jumped: s.has_double_jumped as u8,
            has_flipped: s.has_flipped as u8,
            flip_rel_torque: s.flip_rel_torque.into(),
            jump_ticks: s.jump_ticks,
            jump_hold_broken: s.jump_hold_broken as _,
            flip_time: s.flip_time,
            is_flipping: s.is_flipping as u8,
            is_jumping: s.is_jumping as u8,
            air_time: s.air_time,
            air_time_since_jump: s.air_time_since_jump,
            boost: s.boost,
            time_since_boosted: s.time_since_boosted,
            is_boosting: s.is_boosting as u8,
            boosting_time: s.boosting_time,
            is_supersonic: s.is_supersonic as u8,
            supersonic_grace_timer: s.supersonic_grace_timer,
            handbrake_val: s.handbrake_val,
            is_auto_flipping: s.is_auto_flipping as u8,
            auto_flip_timer: s.auto_flip_timer,
            auto_flip_torque_scale: s.auto_flip_torque_scale,
            bump_cooldown_timer: s.bump_cooldown_timer,
            has_world_contact: s.world_contact_normal.is_some() as u8,
            world_contact_normal: s.world_contact_normal.unwrap_or(Vec3A::ZERO).into(),
            is_demoed: s.is_demoed as u8,
            demo_respawn_timer: s.demo_respawn_timer,
            bump_cooldown_victim: s.bump_cooldown_victim.unwrap_or(0) as u32,
            has_bump_cooldown_victim: s.bump_cooldown_victim.is_some() as u8,
            last_extra_hit_tick: s.last_extra_hit_tick.unwrap_or(0),
            has_last_extra_hit_tick: s.last_extra_hit_tick.is_some() as u8,
        }
    }
}

impl From<&RsfCarState> for CarState {
    fn from(s: &RsfCarState) -> Self {
        CarState {
            phys: rocketsim::PhysState {
                pos: s.phys.pos.into(),
                rot_mat: s.phys.rot_mat.into(),
                vel: s.phys.vel.into(),
                ang_vel: s.phys.ang_vel.into(),
            },
            controls: s.controls.into(),
            prev_controls: s.prev_controls.into(),
            is_on_ground: s.is_on_ground != 0,
            wheels_with_contact: std::array::from_fn(|i| {
                (s.wheels_with_contact[i] != 0).then(|| s.wheel_contacts[i].into())
            }),
            has_jumped: s.has_jumped != 0,
            has_double_jumped: s.has_double_jumped != 0,
            has_flipped: s.has_flipped != 0,
            flip_rel_torque: s.flip_rel_torque.into(),
            jump_ticks: s.jump_ticks,
            jump_hold_broken: s.jump_hold_broken != 0,
            flip_time: s.flip_time,
            is_flipping: s.is_flipping != 0,
            is_jumping: s.is_jumping != 0,
            air_time: s.air_time,
            air_time_since_jump: s.air_time_since_jump,
            boost: s.boost,
            time_since_boosted: s.time_since_boosted,
            is_boosting: s.is_boosting != 0,
            boosting_time: s.boosting_time,
            is_supersonic: s.is_supersonic != 0,
            supersonic_grace_timer: s.supersonic_grace_timer,
            handbrake_val: s.handbrake_val,
            is_auto_flipping: s.is_auto_flipping != 0,
            auto_flip_timer: s.auto_flip_timer,
            auto_flip_torque_scale: s.auto_flip_torque_scale,
            bump_cooldown_timer: s.bump_cooldown_timer,
            bump_cooldown_victim: (s.has_bump_cooldown_victim != 0)
                .then_some(s.bump_cooldown_victim as usize),
            last_extra_hit_tick: (s.has_last_extra_hit_tick != 0).then_some(s.last_extra_hit_tick),
            world_contact_normal: (s.has_world_contact != 0).then(|| s.world_contact_normal.into()),
            is_demoed: s.is_demoed != 0,
            demo_respawn_timer: s.demo_respawn_timer,
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Default, Debug, PartialEq)]
pub struct RsfBallState {
    pub phys: RsfPhysState,
    pub hs_target: i8,
    pub hs_speed: f32,
    pub hs_time: f32,
    pub ds_charge: u8,
    pub ds_force: f32,
    pub ds_target: i8,
    pub ds_last_damage_tick: u64,
    pub ds_has_damage_tick: u8,
    pub tick_count_since_kickoff: u64,
}

impl From<&BallState> for RsfBallState {
    fn from(s: &BallState) -> Self {
        Self {
            phys: RsfPhysState {
                pos: s.pos.into(),
                rot_mat: s.rot_mat.into(),
                vel: s.vel.into(),
                ang_vel: s.ang_vel.into(),
            },
            hs_target: s.hs_info.y_target_dir,
            hs_speed: s.hs_info.cur_target_speed,
            hs_time: s.hs_info.time_since_hit,
            ds_charge: s.ds_info.charge_level,
            ds_force: s.ds_info.accumulated_hit_force,
            ds_target: s.ds_info.y_target_dir,
            ds_last_damage_tick: s.ds_info.last_damage_tick.unwrap_or(0),
            ds_has_damage_tick: s.ds_info.last_damage_tick.is_some() as u8,
            tick_count_since_kickoff: s.tick_count_since_kickoff,
        }
    }
}

impl From<&RsfBallState> for BallState {
    fn from(s: &RsfBallState) -> Self {
        BallState {
            phys: rocketsim::PhysState {
                pos: s.phys.pos.into(),
                rot_mat: s.phys.rot_mat.into(),
                vel: s.phys.vel.into(),
                ang_vel: s.phys.ang_vel.into(),
            },
            hs_info: rocketsim::HeatseekerInfo {
                y_target_dir: s.hs_target,
                cur_target_speed: s.hs_speed,
                time_since_hit: s.hs_time,
            },
            ds_info: rocketsim::DropshotInfo {
                charge_level: s.ds_charge,
                accumulated_hit_force: s.ds_force,
                y_target_dir: s.ds_target,
                last_damage_tick: (s.ds_has_damage_tick != 0).then_some(s.ds_last_damage_tick),
            },
            tick_count_since_kickoff: s.tick_count_since_kickoff,
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Default, Debug, PartialEq)]
pub struct RsfPadConfig {
    pub pos: RsfVec3,
    pub is_big: u8,
}

#[repr(C)]
#[derive(Clone, Copy, Default, Debug, PartialEq)]
pub struct RsfPadState {
    pub cooldown: f32,
    pub is_active: u8,
}

pub const RSF_EVENT_CAR_HIT_BALL: u32 = 0;
pub const RSF_EVENT_CAR_HIT_CAR: u32 = 1;
pub const RSF_EVENT_CAR_PICKUP_BOOST: u32 = 2;
pub const RSF_EVENT_BALL_HIT_WORLD: u32 = 3;
pub const RSF_EVENT_CAR_HIT_WORLD: u32 = 4;

/// kind-dependent fields:
///  CAR_HIT_BALL:     a=car_idx,            pos=contact_point, extra=extra_hit_vel
///  CAR_HIT_CAR:      a=bumper, b=victim,   pos=contact_point, is_demo
///  CAR_PICKUP_BOOST: a=car_idx, b=pad_idx
///  BALL_HIT_WORLD:   pos=contact_point,    extra=contact_normal
///  CAR_HIT_WORLD:    a=car_idx, pos=contact_point, extra=contact_normal
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RsfEvent {
    pub kind: u32,
    pub a: u32,
    pub b: u32,
    pub is_demo: u8,
    pub tick: u64,
    pub pos: RsfVec3,
    pub extra: RsfVec3,
    /// Ball center after the contact tick, populated for CAR_HIT_BALL.
    pub ball_position: RsfVec3,
    pub wheels: [RsfRaycastHit; 4],
    pub wheel_contacts: [u8; 4],
}

#[repr(C)]
#[derive(Clone, Copy, Default, Debug, PartialEq)]
pub struct RsfRaycastHit {
    pub point: RsfVec3,
    pub normal: RsfVec3,
    pub fraction: f32,
    /// 0=world, 1=car, 2=ball, 3=dropshot tile.
    pub object_type: u32,
}
impl From<RaycastHitInfo> for RsfRaycastHit {
    fn from(h: RaycastHitInfo) -> Self {
        Self {
            point: h.hit_point.into(),
            normal: h.hit_normal.into(),
            fraction: h.hit_fraction,
            object_type: match h.user_info {
                UserInfoType::None => 0,
                UserInfoType::Car => 1,
                UserInfoType::Ball => 2,
                UserInfoType::DropshotTile => 3,
            },
        }
    }
}
impl From<RsfRaycastHit> for RaycastHitInfo {
    fn from(h: RsfRaycastHit) -> Self {
        Self {
            hit_point: h.point.into(),
            hit_normal: h.normal.into(),
            hit_fraction: h.fraction,
            user_info: match h.object_type {
                1 => UserInfoType::Car,
                2 => UserInfoType::Ball,
                3 => UserInfoType::DropshotTile,
                _ => UserInfoType::None,
            },
        }
    }
}

/// Opaque owned arena. Each handle requires exclusive access, including reads.
/// Independent handles may be used on different threads after initialization.
pub struct FfiArena {
    arena: Arena,
    events: Vec<RsfEvent>,
}

thread_local! { static LAST_ERROR: RefCell<Vec<u8>> = RefCell::new(vec![0]); }
fn guarded<T>(fallback: T, f: impl FnOnce() -> Result<T, String>) -> T {
    LAST_ERROR.with(|e| {
        let mut buffer = e.borrow_mut();
        buffer.clear();
        buffer.push(0);
    });
    match catch_unwind(AssertUnwindSafe(f)) {
        Ok(Ok(value)) => value,
        other => {
            let message = match other {
                Ok(Err(message)) => message,
                Err(_) => "RocketSim panicked; discard this arena before continuing".to_owned(),
                _ => unreachable!(),
            };
            LAST_ERROR.with(|e| {
                let mut buffer = e.borrow_mut();
                buffer.clear();
                buffer.extend(message.bytes().map(|c| if c == 0 { b' ' } else { c }));
                buffer.push(0);
            });
            fallback
        }
    }
}
/// ABI revision for matching generated headers and libraries.
#[unsafe(no_mangle)]
pub extern "C" fn rsf_abi_version() -> u32 {
    2
}
/// Thread-local message, valid until the next API call on the same thread.
#[unsafe(no_mangle)]
pub extern "C" fn rsf_last_error() -> *const c_char {
    LAST_ERROR.with(|e| e.borrow().as_ptr().cast())
}

unsafe fn arena<'a>(p: *mut FfiArena) -> Result<&'a mut FfiArena, String> {
    unsafe { p.as_mut() }.ok_or_else(|| "null arena".to_owned())
}
unsafe fn input<'a, T>(p: *const T) -> Result<&'a T, String> {
    unsafe { p.as_ref() }.ok_or_else(|| "null input".to_owned())
}
unsafe fn output<T>(p: *mut T, value: T) -> Result<c_int, String> {
    if p.is_null() {
        return Err("null output".to_owned());
    }
    unsafe { p.write(value) };
    Ok(0)
}
fn car_index(a: &FfiArena, index: u32) -> Result<usize, String> {
    if (index as usize) < a.arena.num_cars() {
        Ok(index as usize)
    } else {
        Err("car index out of range".to_owned())
    }
}
fn pad_index(a: &FfiArena, index: u32) -> Result<usize, String> {
    if (index as usize) < a.arena.num_boost_pads() {
        Ok(index as usize)
    } else {
        Err("pad index out of range".to_owned())
    }
}
fn game_mode(value: u32) -> Result<GameMode, String> {
    match value {
        0 => Ok(GameMode::Soccar),
        1 => Ok(GameMode::Hoops),
        2 => Ok(GameMode::Heatseeker),
        3 => Ok(GameMode::Snowday),
        4 => Ok(GameMode::Dropshot),
        5 => Ok(GameMode::TheVoid),
        _ => Err("unknown game mode".to_owned()),
    }
}
fn require_meshes(mode: GameMode) -> Result<(), String> {
    if mode == GameMode::TheVoid {
        return Ok(());
    }
    if !rocketsim::is_initialized() {
        return Err("initialize collision meshes first".to_owned());
    }
    if rocketsim::get_arena_collision_mesh_files(mode).is_empty() {
        return Err("collision meshes for the requested mode are not loaded".to_owned());
    }
    Ok(())
}
fn pad_state(a: &Arena, index: usize) -> RsfPadState {
    let cooldown = a.get_boost_pad_state(index).cooldown;
    RsfPadState {
        cooldown,
        is_active: (cooldown <= 0.0) as u8,
    }
}
/// All input pointers must be aligned/live for the call; strings are UTF-8 and NUL terminated.
/// Output buffers must be writable and disjoint from inputs and arena storage.
/// Status functions return 0 on success, -1 on error. Null free is allowed.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rsf_init(path: *const c_char, silent: c_int) -> c_int {
    guarded(-1, || {
        if path.is_null() {
            return Err("null mesh path".to_owned());
        }
        let path = unsafe { CStr::from_ptr(path) }
            .to_str()
            .map_err(|e| e.to_string())?;
        rocketsim::init(path, silent != 0).map_err(|e| e.to_string())?;
        Ok(0)
    })
}
#[unsafe(no_mangle)]
pub extern "C" fn rsf_is_initialized() -> c_int {
    rocketsim::is_initialized() as c_int
}
/// Whether the public engine has loaded collision meshes for a mode (TheVoid needs none).
#[unsafe(no_mangle)]
pub extern "C" fn rsf_has_meshes(mode: u32) -> c_int {
    guarded(-1, || {
        let mode = game_mode(mode)?;
        if mode == GameMode::TheVoid {
            return Ok(1);
        }
        let mesh_mode = match mode {
            GameMode::Heatseeker | GameMode::Snowday => GameMode::Soccar,
            _ => mode,
        };
        Ok((!rocketsim::get_arena_collision_mesh_files(mesh_mode).is_empty()) as c_int)
    })
}

/// Memory mode: 0=Heavy, 1=Balanced, 2=Light. Negative seed uses a random seed.
/// Null return indicates failure; inspect rsf_last_error(). No engine memory crosses the ABI.
#[unsafe(no_mangle)]
pub extern "C" fn rsf_arena_new(
    mode: u32,
    seed: i64,
    no_ball_rot: u8,
    memory_mode: u32,
) -> *mut FfiArena {
    guarded(std::ptr::null_mut(), || {
        let mode = game_mode(mode)?;
        require_meshes(mode)?;
        let memory = match memory_mode {
            0 => rocketsim::ArenaMemWeightMode::Heavy,
            1 => rocketsim::ArenaMemWeightMode::Balanced,
            2 => rocketsim::ArenaMemWeightMode::Light,
            _ => return Err("unknown memory mode".to_owned()),
        };
        let config = ArenaConfig {
            rng_seed: (seed >= 0).then_some(seed as u64),
            no_ball_rot: no_ball_rot != 0,
            mem_weight_mode: memory,
            ..ArenaConfig::new(mode)
        };
        Ok(Box::into_raw(Box::new(FfiArena {
            arena: Arena::new_with_config(config),
            events: Vec::new(),
        })))
    })
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rsf_arena_free(p: *mut FfiArena) {
    if !p.is_null() {
        drop(unsafe { Box::from_raw(p) });
    }
}
/// Clone the public engine's full physics/RNG state; the two handles have independent ownership.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rsf_arena_clone(p: *mut FfiArena) -> *mut FfiArena {
    guarded(std::ptr::null_mut(), || {
        let a = unsafe { arena(p)? };
        Ok(Box::into_raw(Box::new(FfiArena {
            arena: a.arena.clone(),
            events: a.events.clone(),
        })))
    })
}
/// Returns UINT32_MAX on failure. Null configuration selects the native Octane preset.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rsf_arena_add_car(
    p: *mut FfiArena,
    team: u32,
    config: *const RsfCarConfig,
) -> u32 {
    guarded(u32::MAX, || {
        let team = match team {
            0 => Team::Blue,
            1 => Team::Orange,
            _ => return Err("unknown team".to_owned()),
        };
        let body = if config.is_null() {
            CarBodyConfig::OCTANE
        } else {
            unsafe { input(config)? }.into()
        };
        if !body.hitbox_size.is_finite() || body.hitbox_size.min_element() <= 0.0 {
            return Err("invalid hitbox".to_owned());
        }
        Ok(unsafe { arena(p)? }.arena.add_car(team, body) as u32)
    })
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rsf_arena_num_cars(p: *mut FfiArena) -> u32 {
    guarded(
        u32::MAX,
        || Ok(unsafe { arena(p)? }.arena.num_cars() as u32),
    )
}
fn convert_event(ev: &ArenaEvent, tick: u64) -> RsfEvent {
    let mut out = RsfEvent {
        kind: 0,
        a: 0,
        b: 0,
        is_demo: 0,
        tick,
        pos: RsfVec3::default(),
        extra: RsfVec3::default(),
        ball_position: RsfVec3::default(),
        wheels: [RsfRaycastHit::default(); 4],
        wheel_contacts: [0; 4],
    };
    match ev {
        ArenaEvent::CarHitBall(e) => {
            out.kind = 0;
            out.a = e.car_idx as u32;
            out.pos = e.contact_point.into();
            out.extra = e.extra_hit_vel.into();
        }
        ArenaEvent::CarHitCar(e) => {
            out.kind = 1;
            out.a = e.bumper_car_idx as u32;
            out.b = e.victim_car_idx as u32;
            out.pos = e.contact_point.into();
            out.is_demo = e.is_demo as u8;
        }
        ArenaEvent::CarPickupBoost(e) => {
            out.kind = 2;
            out.a = e.car_idx as u32;
            out.b = e.boost_pad_idx as u32;
        }
        ArenaEvent::BallHitWorld(e) => {
            out.kind = 3;
            out.pos = e.contact_point.into();
            out.extra = e.contact_normal.into();
        }
        ArenaEvent::CarHitWorld(e) => {
            out.kind = 4;
            out.a = e.car_idx as u32;
            out.pos = e.contact_point.into();
            out.extra = e.contact_normal.into();
        }
        ArenaEvent::CarLanded(e) => {
            out.kind = 5;
            out.a = e.car_idx as u32;
            out.wheel_contacts = e.wheels.map(|w| w.is_some() as u8);
            out.wheels = e.wheels.map(|w| w.unwrap_or_default().into());
        }
    }
    out
}
pub const RSF_EVENT_CAR_LANDED: u32 = 5;
/// Events accumulate across all requested ticks. Zero ticks preserves state AND events.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rsf_arena_step(p: *mut FfiArena, ticks: u32) -> c_int {
    guarded(-1, || {
        let a = unsafe { arena(p)? };
        if ticks == 0 {
            return Ok(0);
        }
        a.events.clear();
        for _ in 0..ticks {
            let tick = a.arena.tick_count();
            let first = a.events.len();
            for ev in a.arena.step_tick() {
                a.events.push(convert_event(ev, tick));
            }
            for event in &mut a.events[first..] {
                if event.kind == RSF_EVENT_CAR_HIT_BALL {
                    event.ball_position = a.arena.get_ball_state().pos.into();
                }
            }
        }
        Ok(0)
    })
}
/// Returns written count, or UINT32_MAX on error. Query total with cap=0 and out=NULL.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rsf_arena_get_events(
    p: *mut FfiArena,
    out: *mut RsfEvent,
    cap: u32,
    total: *mut u32,
) -> u32 {
    guarded(u32::MAX, || {
        let a = unsafe { arena(p)? };
        let n = a.events.len().min(cap as usize);
        if n > 0 && out.is_null() {
            return Err("null event buffer".to_owned());
        }
        if !total.is_null() {
            unsafe { total.write(a.events.len() as u32) }
        }
        if n > 0 {
            unsafe { std::ptr::copy_nonoverlapping(a.events.as_ptr(), out, n) }
        }
        Ok(n as u32)
    })
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rsf_arena_get_car_state(
    p: *mut FfiArena,
    idx: u32,
    out: *mut RsfCarState,
) -> c_int {
    guarded(-1, || {
        let a = unsafe { arena(p)? };
        let i = car_index(a, idx)?;
        unsafe { output(out, a.arena.get_car_state(i).into()) }
    })
}
/// State transfer follows upstream teleport semantics. Use reset_contacts for an episode reset;
/// clone is the operation for exact physics copies (state structs do not contain solver caches).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rsf_arena_set_car_state(
    p: *mut FfiArena,
    idx: u32,
    state: *const RsfCarState,
) -> c_int {
    guarded(-1, || {
        let a = unsafe { arena(p)? };
        let i = car_index(a, idx)?;
        let state = unsafe { input(state)? };
        a.arena.set_car_state(i, state.into());
        Ok(0)
    })
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rsf_arena_reset_contacts(p: *mut FfiArena, idx: u32) -> c_int {
    guarded(-1, || {
        let a = unsafe { arena(p)? };
        let i = car_index(a, idx)?;
        a.arena.reset_car_transient_contacts(i);
        a.arena.clear_persistent_manifolds();
        a.events.clear();
        Ok(0)
    })
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rsf_arena_set_car_controls(
    p: *mut FfiArena,
    idx: u32,
    c: *const RsfCarControls,
) -> c_int {
    guarded(-1, || {
        let a = unsafe { arena(p)? };
        let i = car_index(a, idx)?;
        let c = unsafe { input(c)? };
        a.arena.set_car_controls(i, (*c).into());
        Ok(0)
    })
}
/// Applies controls for every car in add order. Count must match; validates before any mutation.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rsf_arena_set_controls(
    p: *mut FfiArena,
    c: *const RsfCarControls,
    count: u32,
) -> c_int {
    guarded(-1, || {
        let a = unsafe { arena(p)? };
        if count as usize != a.arena.num_cars() || (count > 0 && c.is_null()) {
            return Err("invalid controls buffer/count".to_owned());
        }
        if count > 0 {
            for (i, c) in unsafe { std::slice::from_raw_parts(c, count as usize) }
                .iter()
                .enumerate()
            {
                a.arena.set_car_controls(i, (*c).into());
            }
        }
        Ok(0)
    })
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rsf_arena_respawn_car(p: *mut FfiArena, idx: u32) -> c_int {
    guarded(-1, || {
        let a = unsafe { arena(p)? };
        let i = car_index(a, idx)?;
        a.arena.respawn_car(i);
        Ok(0)
    })
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rsf_arena_get_ball_state(
    p: *mut FfiArena,
    out: *mut RsfBallState,
) -> c_int {
    guarded(-1, || {
        let a = unsafe { arena(p)? };
        unsafe { output(out, a.arena.get_ball_state().into()) }
    })
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rsf_arena_set_ball_state(
    p: *mut FfiArena,
    s: *const RsfBallState,
) -> c_int {
    guarded(-1, || {
        let a = unsafe { arena(p)? };
        let s = unsafe { input(s)? };
        a.arena.set_ball_state(s.into());
        Ok(0)
    })
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rsf_arena_reset_kickoff(p: *mut FfiArena, seed: i64) -> c_int {
    guarded(-1, || {
        let a = unsafe { arena(p)? };
        a.arena
            .reset_to_random_kickoff((seed >= 0).then_some(seed as u64));
        a.events.clear();
        Ok(0)
    })
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rsf_arena_tick_count(p: *mut FfiArena) -> u64 {
    guarded(u64::MAX, || Ok(unsafe { arena(p)? }.arena.tick_count()))
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rsf_arena_is_ball_scored(p: *mut FfiArena) -> c_int {
    guarded(-1, || {
        Ok(unsafe { arena(p)? }.arena.is_ball_scored() as c_int)
    })
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rsf_arena_num_pads(p: *mut FfiArena) -> u32 {
    guarded(u32::MAX, || {
        Ok(unsafe { arena(p)? }.arena.num_boost_pads() as u32)
    })
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rsf_arena_get_pad_config(
    p: *mut FfiArena,
    idx: u32,
    out: *mut RsfPadConfig,
) -> c_int {
    guarded(-1, || {
        let a = unsafe { arena(p)? };
        let i = pad_index(a, idx)?;
        let s = a.arena.get_boost_pad_config(i);
        unsafe {
            output(
                out,
                RsfPadConfig {
                    pos: s.pos.into(),
                    is_big: s.is_big as u8,
                },
            )
        }
    })
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rsf_arena_get_pad_state(
    p: *mut FfiArena,
    idx: u32,
    out: *mut RsfPadState,
) -> c_int {
    guarded(-1, || {
        let a = unsafe { arena(p)? };
        let i = pad_index(a, idx)?;
        unsafe { output(out, pad_state(&a.arena, i)) }
    })
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rsf_arena_set_pad_state(
    p: *mut FfiArena,
    idx: u32,
    s: *const RsfPadState,
) -> c_int {
    guarded(-1, || {
        let a = unsafe { arena(p)? };
        let i = pad_index(a, idx)?;
        let s = unsafe { input(s)? };
        if !s.cooldown.is_finite() || s.cooldown < 0.0 || (s.is_active != 0) != (s.cooldown <= 0.0)
        {
            return Err("inconsistent pad state".to_owned());
        }
        a.arena.set_boost_pad_state(
            i,
            BoostPadState {
                cooldown: s.cooldown,
            },
        );
        Ok(0)
    })
}

pub const RSF_TILE_FULL: u8 = 0;
pub const RSF_TILE_DAMAGED: u8 = 1;
pub const RSF_TILE_BROKEN: u8 = 2;

/// Per-car touch and ground summary of one rsf_arena_step_snapshot call.
/// Ticks are relative to the step: 0 is the first tick the call simulated.
#[repr(C)]
#[derive(Clone, Copy, Default, Debug, PartialEq)]
pub struct RsfStepCarResult {
    /// 1 when this car touched the ball during the step; the other hit fields are zero otherwise.
    pub hit_ball: u8,
    /// Step-relative tick of the car's last touch.
    pub hit_tick: u64,
    /// Ball center at the start of the touch tick, before that tick's contact resolved.
    pub hit_ball_pos: RsfVec3,
    /// World-space contact point on the ball of the last touch.
    pub hit_contact_point: RsfVec3,
    /// Extra hit velocity the engine reported for the last touch (CarHitBall extra_hit_vel).
    pub hit_extra_vel: RsfVec3,
    /// OR of isOnGround over the step-start state and the state after every tick.
    pub was_on_ground: u8,
}

/// One car-car contact the engine counted as a bump or demo.
#[repr(C)]
#[derive(Clone, Copy, Default, Debug, PartialEq)]
pub struct RsfBump {
    /// Car index (add order) of the attacker.
    pub bumper: u32,
    /// Car index (add order) of the car that was bumped or demolished.
    pub victim: u32,
    /// 1 when the contact demolished the victim.
    pub is_demo: u8,
    /// Step-relative tick of the contact.
    pub tick: u64,
}

/// Caller-allocated output of rsf_arena_step_snapshot.
/// The caller sets every pointer and count; the call fills the buffers and writes bumpsWritten and isBallScored.
/// Counts must equal the arena's: numCars = rsf_arena_num_cars, numPads = rsf_arena_num_pads,
/// numTiles = rsf_arena_num_tiles (0 outside Dropshot). A buffer pointer may be null only when its count is 0.
#[repr(C)]
pub struct RsfStepOutput {
    /// numCars entries: each car's full state after the last tick, in add order.
    pub cars: *mut RsfCarState,
    /// numCars entries: each car's touch and ground summary for the step.
    pub car_results: *mut RsfStepCarResult,
    pub num_cars: u32,
    /// Ball state after the last tick. Never null.
    pub ball: *mut RsfBallState,
    /// numPads entries: pad states after the last tick, in rsf_arena_get_pad_config order.
    pub pads: *mut RsfPadState,
    pub num_pads: u32,
    /// numTiles RSF_TILE_* values after the last tick, Blue's 70 tiles then Orange's.
    pub tiles: *mut u8,
    pub num_tiles: u32,
    /// Room for bumpsCap bumps, stored in the order they happened.
    pub bumps: *mut RsfBump,
    pub bumps_cap: u32,
    /// Count of every bump during the step. Can exceed bumpsCap; only the first bumpsCap are stored.
    pub bumps_written: u32,
    /// rsf_arena_is_ball_scored after the last tick.
    pub is_ball_scored: u8,
}

/// Owned ball-only arena for predicting ball trajectories. Each handle requires exclusive access.
pub struct FfiBallPredictor {
    arena: Arena,
    /// Dropshot only: the tiles every prediction starts from, since a prediction can damage them.
    tiles: Option<TileStates>,
}

/// Ticks after which a damage age no longer changes physics: Dropshot rate-limits damage only within MIN_DAMAGE_INTERVAL.
/// The + 2 covers the f32 product truncating below 12 and the rate limit's inclusive `<=` boundary.
const DAMAGE_AGE_LIMIT: u64 = (consts::dropshot::MIN_DAMAGE_INTERVAL * consts::TICK_RATE) as u64 + 2;

fn tile_state(value: u8) -> Result<TileDamageState, String> {
    match value {
        RSF_TILE_FULL => Ok(TileDamageState::Full),
        RSF_TILE_DAMAGED => Ok(TileDamageState::Damaged),
        RSF_TILE_BROKEN => Ok(TileDamageState::Broken),
        _ => Err("unknown tile state".to_owned()),
    }
}
unsafe fn tile_states_from(states: *const u8, count: u32) -> Result<TileStates, String> {
    let total = consts::dropshot::NUM_TILES_PER_TEAM * 2;
    if count as usize != total || states.is_null() {
        return Err("tile buffer must hold every Dropshot tile".to_owned());
    }
    let values = unsafe { std::slice::from_raw_parts(states, total) };
    let mut tiles = TileStates::DEFAULT;
    for (slot, value) in tiles.states.iter_mut().flatten().zip(values) {
        *slot = tile_state(*value)?;
    }
    Ok(tiles)
}
fn write_tiles(tiles: &TileStates, out: &mut [u8]) {
    for (slot, state) in out.iter_mut().zip(tiles.states.iter().flatten()) {
        *slot = *state as u8;
    }
}
unsafe fn slice_out<'a, T>(p: *mut T, count: u32) -> Result<&'a mut [T], String> {
    if count == 0 {
        return Ok(&mut []);
    }
    if p.is_null() {
        return Err("null output buffer".to_owned());
    }
    Ok(unsafe { std::slice::from_raw_parts_mut(p, count as usize) })
}

/// Sets every car's controls, steps `ticks` ticks and writes a full snapshot into `out`.
/// `controls` holds numCars entries in add order and may be null only without cars.
/// Inputs are validated before any mutation. Zero ticks applies the controls and reports the current state.
/// Snapshot steps do not record into the rsf_arena_get_events buffer; a nonzero step clears it.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rsf_arena_step_snapshot(
    p: *mut FfiArena,
    controls: *const RsfCarControls,
    ticks: u32,
    out: *mut RsfStepOutput,
) -> c_int {
    guarded(-1, || {
        let a = unsafe { arena(p)? };
        let out = unsafe { out.as_mut() }.ok_or_else(|| "null output".to_owned())?;
        let num_cars = a.arena.num_cars();
        if out.num_cars as usize != num_cars
            || out.num_pads as usize != a.arena.num_boost_pads()
            || out.num_tiles as usize != a.arena.num_tiles()
        {
            return Err("output counts do not match the arena".to_owned());
        }
        if num_cars > 0 && controls.is_null() {
            return Err("null controls".to_owned());
        }
        let cars = unsafe { slice_out(out.cars, out.num_cars)? };
        let results = unsafe { slice_out(out.car_results, out.num_cars)? };
        let pads = unsafe { slice_out(out.pads, out.num_pads)? };
        let tiles = unsafe { slice_out(out.tiles, out.num_tiles)? };
        let bumps = unsafe { slice_out(out.bumps, out.bumps_cap)? };
        let ball = unsafe { out.ball.as_mut() }.ok_or_else(|| "null ball output".to_owned())?;

        if num_cars > 0 {
            let controls = unsafe { std::slice::from_raw_parts(controls, num_cars) };
            for (i, c) in controls.iter().enumerate() {
                a.arena.set_car_controls(i, (*c).into());
            }
        }
        for (i, result) in results.iter_mut().enumerate() {
            *result = RsfStepCarResult {
                was_on_ground: a.arena.get_car_state(i).is_on_ground as u8,
                ..Default::default()
            };
        }
        if ticks > 0 {
            a.events.clear();
        }

        let mut bump_count = 0u32;
        for tick in 0..u64::from(ticks) {
            let ball_pos_before: RsfVec3 = a.arena.get_ball_state().pos.into();
            for event in a.arena.step_tick() {
                match event {
                    ArenaEvent::CarHitBall(hit) => {
                        let result = &mut results[hit.car_idx];
                        result.hit_ball = 1;
                        result.hit_tick = tick;
                        result.hit_ball_pos = ball_pos_before;
                        result.hit_contact_point = hit.contact_point.into();
                        result.hit_extra_vel = hit.extra_hit_vel.into();
                    }
                    ArenaEvent::CarHitCar(bump) => {
                        if let Some(slot) = bumps.get_mut(bump_count as usize) {
                            *slot = RsfBump {
                                bumper: bump.bumper_car_idx as u32,
                                victim: bump.victim_car_idx as u32,
                                is_demo: bump.is_demo as u8,
                                tick,
                            };
                        }
                        bump_count += 1;
                    }
                    _ => {}
                }
            }
            for (i, result) in results.iter_mut().enumerate() {
                result.was_on_ground |= a.arena.get_car_state(i).is_on_ground as u8;
            }
        }

        for (i, car) in cars.iter_mut().enumerate() {
            *car = a.arena.get_car_state(i).into();
        }
        *ball = a.arena.get_ball_state().into();
        for (i, pad) in pads.iter_mut().enumerate() {
            *pad = pad_state(&a.arena, i);
        }
        if !tiles.is_empty() {
            write_tiles(a.arena.get_tile_states(), tiles);
        }
        out.bumps_written = bump_count;
        out.is_ball_scored = a.arena.is_ball_scored() as u8;
        Ok(0)
    })
}

/// Teleports a car like rsf_arena_set_car_state, but keeps the native state callers usually cannot map:
/// jumpHoldBroken, lastExtraHitTick (with hasLastExtraHitTick),
/// and the hit details of each wheel that `mapped` marks in contact and that already was.
/// wheelsWithContact comes from `mapped`, as does every other field;
/// a wheel newly marked in contact takes its hit details from `mapped`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rsf_arena_set_car_state_preserving(
    p: *mut FfiArena,
    idx: u32,
    mapped: *const RsfCarState,
) -> c_int {
    guarded(-1, || {
        let a = unsafe { arena(p)? };
        let i = car_index(a, idx)?;
        let mapped = unsafe { input(mapped)? };
        let native = *a.arena.get_car_state(i);
        let mut state = CarState::from(mapped);
        state.jump_hold_broken = native.jump_hold_broken;
        state.last_extra_hit_tick = native.last_extra_hit_tick;
        for (wheel, native_wheel) in state.wheels_with_contact.iter_mut().zip(native.wheels_with_contact) {
            if wheel.is_some() && native_wheel.is_some() {
                *wheel = native_wheel;
            }
        }
        a.arena.set_car_state(i, state);
        Ok(0)
    })
}

/// Teleports the ball like rsf_arena_set_ball_state, but keeps the native tickCountSinceKickoff,
/// so a teleport does not restart the Hoops/Dropshot launch delay.
/// dsLastDamageTick is on rsf_arena_tick_count's clock and must not be later than it.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rsf_arena_set_ball_state_preserving(
    p: *mut FfiArena,
    s: *const RsfBallState,
) -> c_int {
    guarded(-1, || {
        let a = unsafe { arena(p)? };
        let s = unsafe { input(s)? };
        if s.ds_has_damage_tick != 0 && s.ds_last_damage_tick > a.arena.tick_count() {
            return Err("ds_last_damage_tick is later than the arena tick count".to_owned());
        }
        let mut state = BallState::from(s);
        state.tick_count_since_kickoff = a.arena.get_ball_state().tick_count_since_kickoff;
        a.arena.set_ball_state(state);
        Ok(0)
    })
}

/// Number of Dropshot tiles: 140 in Dropshot, 0 otherwise. UINT32_MAX on failure.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rsf_arena_num_tiles(p: *mut FfiArena) -> u32 {
    guarded(u32::MAX, || Ok(unsafe { arena(p)? }.arena.num_tiles() as u32))
}

/// Copies rsf_arena_num_tiles RSF_TILE_* values, Blue's 70 tiles then Orange's; `cap` must hold them all.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rsf_arena_get_tiles(p: *mut FfiArena, out: *mut u8, cap: u32) -> c_int {
    guarded(-1, || {
        let a = unsafe { arena(p)? };
        let count = a.arena.num_tiles() as u32;
        if cap < count {
            return Err("tile buffer too small".to_owned());
        }
        let out = unsafe { slice_out(out, count)? };
        if count > 0 {
            write_tiles(a.arena.get_tile_states(), out);
        }
        Ok(0)
    })
}

/// Replaces every tile state (count 140, values RSF_TILE_*, rsf_arena_get_tiles order) and updates tile collision.
/// Fails outside Dropshot.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rsf_arena_set_tiles(
    p: *mut FfiArena,
    states: *const u8,
    count: u32,
) -> c_int {
    guarded(-1, || {
        let a = unsafe { arena(p)? };
        if a.arena.game_mode() != GameMode::Dropshot {
            return Err("tiles exist only in Dropshot".to_owned());
        }
        let tiles = unsafe { tile_states_from(states, count)? };
        a.arena.set_tile_states(tiles);
        Ok(0)
    })
}

/// Creates a ball predictor for a mode; no_ball_rot must match the arena whose ball it predicts.
/// Null on failure. A Dropshot predictor starts every prediction from all-full tiles until rsf_ball_predictor_set_tiles.
#[unsafe(no_mangle)]
pub extern "C" fn rsf_ball_predictor_new(mode: u32, no_ball_rot: u8) -> *mut FfiBallPredictor {
    guarded(std::ptr::null_mut(), || {
        let mode = game_mode(mode)?;
        require_meshes(mode)?;
        let config = ArenaConfig {
            no_ball_rot: no_ball_rot != 0,
            mem_weight_mode: rocketsim::ArenaMemWeightMode::Light,
            rng_seed: Some(0),
            ..ArenaConfig::new(mode)
        };
        let mut arena = Arena::new_with_config(config);
        let tiles = (mode == GameMode::Dropshot).then_some(TileStates::DEFAULT);

        // Damage ticks are rebased onto this arena's clock, which must first run past any age that still matters.
        if tiles.is_some() {
            for _ in 0..DAMAGE_AGE_LIMIT {
                arena.step_tick();
            }
        }
        Ok(Box::into_raw(Box::new(FfiBallPredictor { arena, tiles })))
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn rsf_ball_predictor_free(p: *mut FfiBallPredictor) {
    if !p.is_null() {
        drop(unsafe { Box::from_raw(p) });
    }
}

/// Sets the Dropshot tiles (count 140, values RSF_TILE_*) that every later prediction starts from.
/// Fails for predictors of other modes.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rsf_ball_predictor_set_tiles(
    p: *mut FfiBallPredictor,
    states: *const u8,
    count: u32,
) -> c_int {
    guarded(-1, || {
        let predictor = unsafe { p.as_mut() }.ok_or_else(|| "null predictor".to_owned())?;
        let Some(saved) = predictor.tiles.as_mut() else {
            return Err("tiles exist only in Dropshot".to_owned());
        };
        *saved = unsafe { tile_states_from(states, count)? };
        Ok(0)
    })
}

/// Predicts `num_ticks` ball states from `start`; out[i] is the state after i + 1 ticks.
/// Each prediction is independent: it starts from `start` and the predictor's saved tiles.
/// The predictor has no game clock, so dsLastDamageTick is an age in both `start` and `out`:
/// ticks since the ball last damaged tiles (the game's rsf_arena_tick_count minus its dsLastDamageTick).
/// Returns the first index whose state scores, -1 if none does, or INT32_MIN on failure.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rsf_ball_predict(
    p: *mut FfiBallPredictor,
    start: *const RsfBallState,
    num_ticks: u32,
    out: *mut RsfBallState,
) -> i32 {
    guarded(i32::MIN, || {
        let predictor = unsafe { p.as_mut() }.ok_or_else(|| "null predictor".to_owned())?;
        let start = unsafe { input(start)? };
        if num_ticks > i32::MAX as u32 {
            return Err("too many prediction ticks".to_owned());
        }
        let out = unsafe { slice_out(out, num_ticks)? };
        let arena = &mut predictor.arena;

        // Contacts cached by an earlier prediction would otherwise leak into this one.
        arena.clear_persistent_manifolds();
        if let Some(tiles) = predictor.tiles {
            arena.set_tile_states(tiles);
        }

        let start_age = start.ds_last_damage_tick;
        let start_damage_tick = (start.ds_has_damage_tick != 0)
            .then(|| arena.tick_count().saturating_sub(start_age.min(DAMAGE_AGE_LIMIT)));
        let mut state = BallState::from(start);
        state.ds_info.last_damage_tick = start_damage_tick;
        arena.set_ball_state(state);

        let mut first_scored = -1;
        for (i, slot) in out.iter_mut().enumerate() {
            arena.step_tick();
            let ball = arena.get_ball_state();
            *slot = ball.into();
            slot.ds_last_damage_tick = match ball.ds_info.last_damage_tick {
                Some(tick) if Some(tick) == start_damage_tick => start_age.saturating_add(i as u64 + 1),
                Some(tick) => arena.tick_count() - tick,
                None => 0,
            };
            if first_scored < 0 && arena.is_ball_scored() {
                first_scored = i as i32;
            }
        }
        Ok(first_scored)
    })
}

#[cfg(test)]
mod tests;
