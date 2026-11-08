use super::*;
use std::sync::Once;

fn initialize() {
    static ONCE: Once = Once::new();
    ONCE.call_once(|| {
        let meshes = std::env::var("RSF_TEST_MESHES")
            .expect("Set RSF_TEST_MESHES to public collision meshes");
        let path = CString::new(meshes).unwrap();
        assert_eq!(unsafe { rsf_init(path.as_ptr(), 1) }, 0);
    });
}
fn make_arena(mode: u32) -> *mut FfiArena {
    initialize();
    let p = rsf_arena_new(mode, 7, 0, 2);
    assert!(!p.is_null());
    p
}
unsafe fn state(p: *mut FfiArena, car: u32) -> RsfCarState {
    let mut state = std::mem::MaybeUninit::uninit();
    assert_eq!(
        unsafe { rsf_arena_get_car_state(p, car, state.as_mut_ptr()) },
        0
    );
    unsafe { state.assume_init() }
}
#[test]
fn stepped_state_and_events_match_native_public_api() {
    let p = make_arena(0);
    let mut native = Arena::new_with_config(
        ArenaConfig::new(GameMode::Soccar)
            .with_rng_seed(7)
            .with_mem_weight_mode(rocketsim::ArenaMemWeightMode::Light),
    );
    native.add_car(Team::Blue, CarBodyConfig::OCTANE);
    unsafe {
        assert_eq!(rsf_arena_add_car(p, 0, std::ptr::null()), 0);
        for tick in 0..360 {
            let controls = CarControls {
                throttle: 1.0,
                steer: 0.25,
                jump: (30..45).contains(&tick),
                boost: tick > 60,
                ..CarControls::DEFAULT
            };
            native.set_car_controls(0, controls);
            assert_eq!(rsf_arena_set_car_controls(p, 0, &controls.into()), 0);
            let mut events: Vec<_> = native
                .step_tick()
                .iter()
                .map(|e| convert_event(e, tick))
                .collect();
            for event in &mut events {
                if event.kind == RSF_EVENT_CAR_HIT_BALL {
                    event.ball_position = native.get_ball_state().pos.into();
                }
            }
            assert_eq!(rsf_arena_step(p, 1), 0);
            assert_eq!(state(p, 0), native.get_car_state(0).into());
            assert_eq!((*p).events, events);
            assert_eq!(rsf_arena_tick_count(p), tick + 1);
        }
        rsf_arena_free(p);
    }
}
#[test]
fn clone_preserves_contact_history_and_is_independent() {
    let p = make_arena(0);
    unsafe {
        rsf_arena_add_car(p, 0, std::ptr::null());
        rsf_arena_set_car_controls(
            p,
            0,
            &RsfCarControls {
                throttle: 1.0,
                ..Default::default()
            },
        );
        assert_eq!(rsf_arena_step(p, 120), 0);
        let copy = rsf_arena_clone(p);
        assert!(!copy.is_null());
        for _ in 0..120 {
            assert_eq!(rsf_arena_step(p, 1), 0);
            assert_eq!(rsf_arena_step(copy, 1), 0);
            assert_eq!(state(p, 0), state(copy, 0));
        }
        rsf_arena_free(p);
        assert_eq!(rsf_arena_step(copy, 1), 0);
        rsf_arena_free(copy);
    }
}
#[test]
fn zero_ticks_preserves_events_and_clock() {
    let p = make_arena(0);
    unsafe {
        assert_eq!(rsf_arena_step(p, 120), 0);
        let events = (*p).events.clone();
        let tick = rsf_arena_tick_count(p);
        assert_eq!(rsf_arena_step(p, 0), 0);
        assert_eq!((*p).events, events);
        assert_eq!(rsf_arena_tick_count(p), tick);
        rsf_arena_free(p);
    }
}
#[test]
fn car_state_preserves_every_native_field() {
    let mut native = CarState::DEFAULT;
    native.jump_ticks = 19;
    native.jump_hold_broken = true;
    native.wheels_with_contact[2] = Some(RaycastHitInfo {
        hit_point: Vec3A::new(1.0, 2.0, 3.0),
        hit_normal: Vec3A::Y,
        hit_fraction: 0.4,
        user_info: UserInfoType::Ball,
    });
    native.bump_cooldown_victim = Some(5);
    native.last_extra_hit_tick = Some(12345);
    let c = RsfCarState::from(&native);
    let back = CarState::from(&c);
    assert_eq!(RsfCarState::from(&back), c);
    assert_eq!(back.wheels_with_contact, native.wheels_with_contact);
}
#[test]
fn ball_mode_state_and_pad_cooldowns_round_trip() {
    // Soccar, Heatseeker and Snowday share the available Soccar meshes.
    for mode in [0, 2, 3] {
        let p = make_arena(mode);
        unsafe {
            let mut s = RsfBallState::default();
            assert_eq!(rsf_arena_get_ball_state(p, &mut s), 0);
            s.hs_target = -1;
            s.hs_speed = 3100.0;
            s.hs_time = 1.25;
            s.ds_charge = 7;
            s.ds_force = 4.5;
            s.ds_target = 1;
            s.ds_has_damage_tick = 1;
            s.ds_last_damage_tick = 123;
            s.tick_count_since_kickoff = 456;
            assert_eq!(rsf_arena_set_ball_state(p, &s), 0);
            let mut out = RsfBallState::default();
            assert_eq!(rsf_arena_get_ball_state(p, &mut out), 0);
            assert_eq!(s, out);
            if rsf_arena_num_pads(p) > 0 {
                let mut pad = RsfPadState {
                    cooldown: 2.0,
                    is_active: 0,
                };
                assert_eq!(rsf_arena_set_pad_state(p, 0, &pad), 0);
                assert_eq!(rsf_arena_get_pad_state(p, 0, &mut pad), 0);
                assert!((pad.cooldown - 2.0).abs() < 0.01);
                assert_eq!(pad.is_active, 0);
            }
            rsf_arena_free(p);
        }
    }
}
#[test]
fn invalid_inputs_return_errors_without_touching_state() {
    assert!(rsf_arena_new(99, 7, 0, 0).is_null());
    let p = make_arena(5);
    unsafe {
        rsf_arena_add_car(p, 0, std::ptr::null());
        let before = state(p, 0);
        assert_eq!(rsf_arena_get_car_state(p, 99, std::ptr::null_mut()), -1);
        assert!(!CStr::from_ptr(rsf_last_error()).to_bytes().is_empty());
        assert_eq!(rsf_arena_set_controls(p, std::ptr::null(), 2), -1);
        assert_eq!(state(p, 0), before);
        assert_eq!(rsf_arena_step(std::ptr::null_mut(), 1), -1);
        rsf_arena_free(p);
        rsf_arena_free(std::ptr::null_mut());
    }
}
#[test]
fn bulk_controls_match_scalar_controls() {
    let p = make_arena(0);
    unsafe {
        for i in 0..6 {
            rsf_arena_add_car(p, i % 2, std::ptr::null());
        }
        let copy = rsf_arena_clone(p);
        let controls: Vec<_> = (0..6)
            .map(|i| RsfCarControls {
                throttle: 1.0,
                steer: i as f32 / 6.0,
                ..Default::default()
            })
            .collect();
        assert_eq!(rsf_arena_set_controls(p, controls.as_ptr(), 6), 0);
        for (i, c) in controls.iter().enumerate() {
            assert_eq!(rsf_arena_set_car_controls(copy, i as u32, c), 0);
        }
        assert_eq!(rsf_arena_step(p, 60), 0);
        assert_eq!(rsf_arena_step(copy, 60), 0);
        for i in 0..6 {
            assert_eq!(state(p, i), state(copy, i));
        }
        rsf_arena_free(p);
        rsf_arena_free(copy);
    }
}

/// Caller-owned buffers behind one RsfStepOutput, sized from the arena like a real consumer.
struct StepBuffers {
    cars: Vec<RsfCarState>,
    results: Vec<RsfStepCarResult>,
    ball: RsfBallState,
    pads: Vec<RsfPadState>,
    tiles: Vec<u8>,
    bumps: Vec<RsfBump>,
    bumps_written: u32,
    is_ball_scored: u8,
}

unsafe fn step_snapshot(
    p: *mut FfiArena,
    controls: &[RsfCarControls],
    ticks: u32,
    bumps_cap: u32,
) -> StepBuffers {
    let num_cars = unsafe { rsf_arena_num_cars(p) } as usize;
    let mut buffers = StepBuffers {
        cars: vec![RsfCarState::from(&CarState::DEFAULT); num_cars],
        results: vec![RsfStepCarResult::default(); num_cars],
        ball: RsfBallState::default(),
        pads: vec![RsfPadState::default(); unsafe { rsf_arena_num_pads(p) } as usize],
        tiles: vec![0; unsafe { rsf_arena_num_tiles(p) } as usize],
        bumps: vec![RsfBump::default(); bumps_cap as usize],
        bumps_written: 0,
        is_ball_scored: 0,
    };
    let mut out = RsfStepOutput {
        cars: buffers.cars.as_mut_ptr(),
        car_results: buffers.results.as_mut_ptr(),
        num_cars: num_cars as u32,
        ball: &mut buffers.ball,
        pads: buffers.pads.as_mut_ptr(),
        num_pads: buffers.pads.len() as u32,
        tiles: buffers.tiles.as_mut_ptr(),
        num_tiles: buffers.tiles.len() as u32,
        bumps: buffers.bumps.as_mut_ptr(),
        bumps_cap,
        bumps_written: u32::MAX,
        is_ball_scored: u8::MAX,
    };
    assert_eq!(
        unsafe { rsf_arena_step_snapshot(p, controls.as_ptr(), ticks, &mut out) },
        0
    );
    buffers.bumps_written = out.bumps_written;
    buffers.is_ball_scored = out.is_ball_scored;
    buffers
}

/// What a step snapshot must report, recorded by stepping a native arena one public tick at a time.
struct ReferenceStep {
    results: Vec<RsfStepCarResult>,
    bumps: Vec<RsfBump>,
    ball_pos_after_hit_tick: Vec<Option<RsfVec3>>,
}

fn reference_step(native: &mut Arena, controls: &[RsfCarControls], ticks: u32) -> ReferenceStep {
    let num_cars = native.num_cars();
    for (i, c) in controls.iter().enumerate() {
        native.set_car_controls(i, (*c).into());
    }
    let mut reference = ReferenceStep {
        results: (0..num_cars)
            .map(|i| RsfStepCarResult {
                was_on_ground: native.get_car_state(i).is_on_ground as u8,
                ..Default::default()
            })
            .collect(),
        bumps: Vec::new(),
        ball_pos_after_hit_tick: vec![None; num_cars],
    };
    for tick in 0..ticks as u64 {
        let ball_pos_before: RsfVec3 = native.get_ball_state().pos.into();
        let events = native.step_tick().to_vec();
        for event in events {
            match event {
                ArenaEvent::CarHitBall(hit) => {
                    let result = &mut reference.results[hit.car_idx];
                    result.hit_ball = 1;
                    result.hit_tick = tick;
                    result.hit_ball_pos = ball_pos_before;
                    result.hit_contact_point = hit.contact_point.into();
                    result.hit_extra_vel = hit.extra_hit_vel.into();
                    reference.ball_pos_after_hit_tick[hit.car_idx] =
                        Some(native.get_ball_state().pos.into());
                }
                ArenaEvent::CarHitCar(bump) => reference.bumps.push(RsfBump {
                    bumper: bump.bumper_car_idx as u32,
                    victim: bump.victim_car_idx as u32,
                    is_demo: bump.is_demo as u8,
                    tick,
                }),
                _ => {}
            }
        }
        for (i, result) in reference.results.iter_mut().enumerate() {
            result.was_on_ground |= native.get_car_state(i).is_on_ground as u8;
        }
    }
    reference
}

fn tile_bytes(tiles: &TileStates) -> Vec<u8> {
    tiles.states.iter().flatten().map(|t| *t as u8).collect()
}

/// Asserts the snapshot's state block equals the native arena after the same ticks.
fn assert_snapshot_matches(buffers: &StepBuffers, native: &Arena) {
    for (i, car) in buffers.cars.iter().enumerate() {
        assert_eq!(*car, RsfCarState::from(native.get_car_state(i)));
    }
    assert_eq!(buffers.ball, RsfBallState::from(native.get_ball_state()));
    for (i, pad) in buffers.pads.iter().enumerate() {
        let cooldown = native.get_boost_pad_state(i).cooldown;
        assert_eq!(pad.cooldown, cooldown);
        assert_eq!(pad.is_active, (cooldown <= 0.0) as u8);
    }
    if native.num_tiles() > 0 {
        assert_eq!(buffers.tiles, tile_bytes(native.get_tile_states()));
    }
    assert_eq!(buffers.is_ball_scored, native.is_ball_scored() as u8);
}

/// Teleports a car onto the field's center line, facing +Y or -Y and driving that way at `speed`.
unsafe fn place_car(p: *mut FfiArena, car: u32, y: f32, forward_y: f32, speed: f32) {
    let mut s = unsafe { state(p, car) };
    s.phys.pos = RsfVec3 { x: 0.0, y, z: 17.0 };
    s.phys.rot_mat = RsfRotMat {
        forward: RsfVec3 { x: 0.0, y: forward_y, z: 0.0 },
        right: RsfVec3 { x: -forward_y, y: 0.0, z: 0.0 },
        up: RsfVec3 { x: 0.0, y: 0.0, z: 1.0 },
    };
    s.phys.vel = RsfVec3 { x: 0.0, y: forward_y * speed, z: 0.0 };
    s.phys.ang_vel = RsfVec3::default();
    assert_eq!(unsafe { rsf_arena_set_car_state(p, car, &s) }, 0);
}

fn has_dropshot_meshes() -> bool {
    initialize();
    let present = rsf_has_meshes(4) == 1;
    if !present {
        eprintln!("Dropshot meshes are missing from RSF_TEST_MESHES; skipping the Dropshot arena checks");
    }
    present
}

#[test]
fn step_snapshot_records_ball_pos_before_touch_tick() {
    let p = make_arena(0);
    unsafe {
        rsf_arena_add_car(p, 0, std::ptr::null());
        rsf_arena_add_car(p, 1, std::ptr::null());
        place_car(p, 0, -1200.0, 1.0, 0.0);
        place_car(p, 1, 1200.0, -1.0, 0.0);
        let mut native = (*p).arena.clone();
        let controls = [RsfCarControls {
            throttle: 1.0,
            boost: 1,
            ..Default::default()
        }; 2];
        let mut touched = false;
        for _ in 0..30 {
            let buffers = step_snapshot(p, &controls, 8, 4);
            let reference = reference_step(&mut native, &controls, 8);
            assert_eq!(buffers.results, reference.results);
            assert_snapshot_matches(&buffers, &native);
            for (result, after) in buffers.results.iter().zip(&reference.ball_pos_after_hit_tick) {
                if let Some(after) = after {
                    touched = true;
                    assert_eq!(result.hit_ball, 1);
                    assert_ne!(result.hit_ball_pos, *after);
                }
            }
        }
        assert!(touched, "kickoff drive never touched the ball");
        rsf_arena_free(p);
    }
}

#[test]
fn step_snapshot_ors_ground_contact_including_start() {
    let p = make_arena(0);
    unsafe {
        rsf_arena_add_car(p, 0, std::ptr::null());
        let mut airborne = state(p, 0);
        airborne.phys.pos = RsfVec3 { x: 0.0, y: 0.0, z: 1200.0 };
        airborne.phys.vel = RsfVec3::default();
        for start_on_ground in [1u8, 0] {
            airborne.is_on_ground = start_on_ground;
            assert_eq!(rsf_arena_set_car_state(p, 0, &airborne), 0);
            let mut native = (*p).arena.clone();
            let controls = [RsfCarControls::default()];
            let buffers = step_snapshot(p, &controls, 4, 0);
            let reference = reference_step(&mut native, &controls, 4);
            assert!(!native.get_car_state(0).is_on_ground);
            assert_eq!(buffers.results[0].was_on_ground, start_on_ground);
            assert_eq!(buffers.results, reference.results);
            assert_snapshot_matches(&buffers, &native);
        }

        // Landing mid-step reports ground contact even though the step started airborne.
        airborne.phys.pos.z = 40.0;
        airborne.is_on_ground = 0;
        assert_eq!(rsf_arena_set_car_state(p, 0, &airborne), 0);
        let mut native = (*p).arena.clone();
        let controls = [RsfCarControls::default()];
        let buffers = step_snapshot(p, &controls, 30, 0);
        let reference = reference_step(&mut native, &controls, 30);
        assert_eq!(buffers.results[0].was_on_ground, 1);
        assert_eq!(buffers.results, reference.results);
        assert_snapshot_matches(&buffers, &native);
        rsf_arena_free(p);
    }
}

#[test]
fn step_snapshot_reports_relative_bump_ticks() {
    let p = make_arena(0);
    unsafe {
        rsf_arena_add_car(p, 0, std::ptr::null());
        rsf_arena_add_car(p, 1, std::ptr::null());
        let idle = [RsfCarControls::default(); 2];
        step_snapshot(p, &idle, 50, 0);
        place_car(p, 0, -400.0, 1.0, 1400.0);
        place_car(p, 1, 400.0, -1.0, 1400.0);
        assert!(rsf_arena_tick_count(p) > 0);
        let mut native = (*p).arena.clone();
        let truncated = rsf_arena_clone(p);
        let drive = [RsfCarControls {
            throttle: 1.0,
            ..Default::default()
        }; 2];
        let buffers = step_snapshot(p, &drive, 60, 8);
        let reference = reference_step(&mut native, &drive, 60);
        assert!(!reference.bumps.is_empty(), "cars never bumped");
        assert_eq!(buffers.bumps_written as usize, reference.bumps.len());
        assert_eq!(&buffers.bumps[..reference.bumps.len()], reference.bumps.as_slice());
        assert_snapshot_matches(&buffers, &native);

        // A short buffer still reports the full count so callers can detect truncation.
        let short = step_snapshot(truncated, &drive, 60, 0);
        assert_eq!(short.bumps_written as usize, reference.bumps.len());
        rsf_arena_free(truncated);
        rsf_arena_free(p);
    }
}

#[test]
fn set_car_state_preserving_keeps_jump_hold_and_extra_hit_tick() {
    let p = make_arena(0);
    unsafe {
        rsf_arena_add_car(p, 0, std::ptr::null());
        step_snapshot(p, &[RsfCarControls::default()], 30, 0);
        let mut native_state = state(p, 0);
        native_state.jump_hold_broken = 1;
        native_state.last_extra_hit_tick = 17;
        native_state.has_last_extra_hit_tick = 1;
        assert_eq!(rsf_arena_set_car_state(p, 0, &native_state), 0);
        let native_state = state(p, 0);
        assert!(native_state.wheels_with_contact.iter().all(|w| *w != 0));

        let mut mapped = native_state;
        mapped.phys.pos.x += 500.0;
        mapped.boost = 12.0;
        mapped.jump_hold_broken = 0;
        mapped.has_last_extra_hit_tick = 0;
        mapped.last_extra_hit_tick = 0;
        mapped.wheel_contacts = [RsfRaycastHit::default(); 4];
        mapped.wheels_with_contact = [1, 1, 0, 0];

        let mut native = (*p).arena.clone();
        let mut expected = mapped;
        expected.jump_hold_broken = native_state.jump_hold_broken;
        expected.last_extra_hit_tick = native_state.last_extra_hit_tick;
        expected.has_last_extra_hit_tick = native_state.has_last_extra_hit_tick;
        expected.wheel_contacts = [
            native_state.wheel_contacts[0],
            native_state.wheel_contacts[1],
            RsfRaycastHit::default(),
            RsfRaycastHit::default(),
        ];
        native.set_car_state(0, (&expected).into());

        assert_eq!(rsf_arena_set_car_state_preserving(p, 0, &mapped), 0);
        assert_eq!(state(p, 0), RsfCarState::from(native.get_car_state(0)));
        let controls = [RsfCarControls {
            throttle: 1.0,
            ..Default::default()
        }];
        for _ in 0..20 {
            let buffers = step_snapshot(p, &controls, 3, 0);
            reference_step(&mut native, &controls, 3);
            assert_snapshot_matches(&buffers, &native);
        }
        assert_eq!(rsf_arena_set_car_state_preserving(p, 9, &mapped), -1);
        assert_eq!(rsf_arena_set_car_state_preserving(p, 0, std::ptr::null()), -1);
        rsf_arena_free(p);
    }
}

#[test]
fn set_ball_state_preserving_keeps_kickoff_counter() {
    let p = make_arena(1);
    unsafe {
        rsf_arena_add_car(p, 0, std::ptr::null());
        assert_eq!(rsf_arena_reset_kickoff(p, 5), 0);
        step_snapshot(p, &[RsfCarControls::default()], 10, 0);
        let mut ball = RsfBallState::default();
        assert_eq!(rsf_arena_get_ball_state(p, &mut ball), 0);
        let counter = ball.tick_count_since_kickoff;
        assert_eq!(counter, 10);

        let mut native = (*p).arena.clone();
        let mut restarted = (*p).arena.clone();
        let mut mapped = ball;
        mapped.tick_count_since_kickoff = 0;
        assert_eq!(rsf_arena_set_ball_state_preserving(p, &mapped), 0);
        let kept_counter = RsfBallState {
            tick_count_since_kickoff: counter,
            ..mapped
        };
        native.set_ball_state((&kept_counter).into());
        restarted.set_ball_state((&mapped).into());

        let mut out = RsfBallState::default();
        assert_eq!(rsf_arena_get_ball_state(p, &mut out), 0);
        assert_eq!(out.tick_count_since_kickoff, counter);
        let idle = [RsfCarControls::default()];
        let buffers = step_snapshot(p, &idle, 40, 0);
        reference_step(&mut native, &idle, 40);
        reference_step(&mut restarted, &idle, 40);
        assert_snapshot_matches(&buffers, &native);
        // The kickoff clock decides when the ball launches, so a restarted counter launches later.
        assert_ne!(
            buffers.ball.phys.vel,
            RsfBallState::from(restarted.get_ball_state()).phys.vel
        );

        let future_damage = RsfBallState {
            ds_has_damage_tick: 1,
            ds_last_damage_tick: rsf_arena_tick_count(p) + 1,
            ..mapped
        };
        assert_eq!(rsf_arena_set_ball_state_preserving(p, &future_damage), -1);
        rsf_arena_free(p);
    }
}

#[test]
fn tiles_round_trip_and_damage_from_charged_ball() {
    let soccar = make_arena(0);
    unsafe {
        assert_eq!(rsf_arena_num_tiles(soccar), 0);
        assert_eq!(rsf_arena_get_tiles(soccar, std::ptr::null_mut(), 0), 0);
        assert_eq!(rsf_arena_set_tiles(soccar, [0u8; 140].as_ptr(), 140), -1);
        rsf_arena_free(soccar);
    }
    let soccar_predictor = rsf_ball_predictor_new(0, 1);
    assert!(!soccar_predictor.is_null());
    unsafe {
        assert_eq!(
            rsf_ball_predictor_set_tiles(soccar_predictor, [0u8; 140].as_ptr(), 140),
            -1
        );
        rsf_ball_predictor_free(soccar_predictor);
    }
    if !has_dropshot_meshes() {
        return;
    }

    let p = make_arena(4);
    unsafe {
        assert_eq!(rsf_arena_num_tiles(p), 140);
        let mut tiles: Vec<u8> = (0..140).map(|i| (i % 2) as u8).collect();
        tiles[3] = RSF_TILE_BROKEN;
        assert_eq!(rsf_arena_set_tiles(p, tiles.as_ptr(), 140), 0);
        let mut read = vec![9u8; 140];
        assert_eq!(rsf_arena_get_tiles(p, read.as_mut_ptr(), 140), 0);
        assert_eq!(read, tiles);
        assert_eq!(rsf_arena_get_tiles(p, read.as_mut_ptr(), 139), -1);
        assert_eq!(rsf_arena_set_tiles(p, [3u8; 140].as_ptr(), 140), -1);
        assert_eq!(rsf_arena_set_tiles(p, tiles.as_ptr(), 139), -1);

        assert_eq!(rsf_arena_set_tiles(p, [RSF_TILE_FULL; 140].as_ptr(), 140), 0);
        let mut ball = RsfBallState::default();
        assert_eq!(rsf_arena_get_ball_state(p, &mut ball), 0);
        ball.phys.pos = RsfVec3 { x: 0.0, y: -1500.0, z: 400.0 };
        ball.phys.vel = RsfVec3 { x: 0.0, y: 0.0, z: -2000.0 };
        ball.ds_charge = 2;
        ball.ds_target = -1;
        ball.ds_has_damage_tick = 0;
        assert_eq!(rsf_arena_set_ball_state_preserving(p, &ball), 0);
        let mut native = (*p).arena.clone();
        let buffers = step_snapshot(p, &[], 60, 0);
        reference_step(&mut native, &[], 60);
        assert_snapshot_matches(&buffers, &native);
        let damaged = buffers.tiles.iter().filter(|t| **t == RSF_TILE_DAMAGED).count();
        assert!(damaged > 1, "a charged ball damages a ring of tiles, got {damaged}");

        // A ball resting over broken tiles falls through and scores in the predictor.
        let predictor = rsf_ball_predictor_new(4, 0);
        assert!(!predictor.is_null());
        let drop = RsfBallState {
            phys: RsfPhysState {
                vel: RsfVec3::default(),
                ..ball.phys
            },
            ds_target: 0,
            ..ball
        };
        let mut predicted = vec![RsfBallState::default(); 240];
        assert_eq!(rsf_ball_predict(predictor, &drop, 240, predicted.as_mut_ptr()), -1);
        let broken = [RSF_TILE_BROKEN; 140];
        assert_eq!(rsf_ball_predictor_set_tiles(predictor, broken.as_ptr(), 140), 0);
        assert!(rsf_ball_predict(predictor, &drop, 240, predicted.as_mut_ptr()) >= 0);
        rsf_ball_predictor_free(predictor);
        rsf_arena_free(p);
    }
}

#[test]
fn ball_predict_matches_stepping_and_reports_score_index() {
    initialize();
    let predictor = rsf_ball_predictor_new(0, 1);
    assert!(!predictor.is_null());
    let shot = RsfBallState {
        phys: RsfPhysState {
            pos: RsfVec3 { x: 300.0, y: 3500.0, z: 300.0 },
            rot_mat: Mat3A::IDENTITY.into(),
            vel: RsfVec3 { x: -200.0, y: 2000.0, z: 100.0 },
            ang_vel: RsfVec3::default(),
        },
        ..RsfBallState::from(&BallState::DEFAULT)
    };
    let lob = RsfBallState {
        phys: RsfPhysState {
            pos: RsfVec3 { x: -1000.0, y: 0.0, z: 900.0 },
            vel: RsfVec3 { x: 1500.0, y: -300.0, z: 0.0 },
            ..shot.phys
        },
        ..shot
    };
    let num_ticks = 240u32;
    let mut scores = Vec::new();
    for start in [shot, lob, shot] {
        let mut reference = Arena::new_with_config(ArenaConfig {
            no_ball_rot: true,
            ..ArenaConfig::new(GameMode::Soccar)
        });
        reference.set_ball_state((&start).into());
        let mut expected = Vec::new();
        let mut expected_score = -1;
        for i in 0..num_ticks {
            reference.step_tick();
            expected.push(RsfBallState::from(reference.get_ball_state()));
            if expected_score < 0 && reference.is_ball_scored() {
                expected_score = i as i32;
            }
        }
        let mut predicted = vec![RsfBallState::default(); num_ticks as usize];
        let score =
            unsafe { rsf_ball_predict(predictor, &start, num_ticks, predicted.as_mut_ptr()) };
        assert_eq!(score, expected_score);
        assert_eq!(predicted, expected);
        scores.push(score);
    }
    assert!(scores[0] >= 0, "the shot never reached the goal");
    assert_eq!(scores[1], -1);

    let mut scratch = [RsfBallState::default(); 1];
    unsafe {
        assert_eq!(
            rsf_ball_predict(predictor, std::ptr::null(), 1, scratch.as_mut_ptr()),
            i32::MIN
        );
        assert_eq!(
            rsf_ball_predict(std::ptr::null_mut(), &shot, 1, scratch.as_mut_ptr()),
            i32::MIN
        );
        assert_eq!(rsf_ball_predict(predictor, &shot, 1, std::ptr::null_mut()), i32::MIN);
        rsf_ball_predictor_free(predictor);
    }
    assert!(rsf_ball_predictor_new(99, 0).is_null());
}

#[test]
fn ball_predict_reports_damage_tick_as_age() {
    initialize();
    let predictor = rsf_ball_predictor_new(0, 1);
    assert!(!predictor.is_null());
    let start = RsfBallState {
        ds_has_damage_tick: 1,
        ..RsfBallState::from(&BallState::DEFAULT)
    };
    let num_ticks = 30u32;
    // One age inside the rebase limit and one far beyond it, which the predictor caps internally.
    for start_age in [5u64, 1000] {
        let mut predicted = vec![RsfBallState::default(); num_ticks as usize];
        let aged = RsfBallState {
            ds_last_damage_tick: start_age,
            ..start
        };
        assert_eq!(
            unsafe { rsf_ball_predict(predictor, &aged, num_ticks, predicted.as_mut_ptr()) },
            -1
        );
        for (i, state) in predicted.iter().enumerate() {
            assert_eq!(state.ds_has_damage_tick, 1);
            assert_eq!(state.ds_last_damage_tick, start_age + i as u64 + 1);
        }
    }
    unsafe { rsf_ball_predictor_free(predictor) };
}
