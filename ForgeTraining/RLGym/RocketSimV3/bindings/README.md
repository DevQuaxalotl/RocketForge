# RocketSim v3 C and C++ bindings

This standalone crate wraps the public Rust API. `include/rocketsim_ffi.h` is the generated C ABI; `include/rocketsim.hpp` is an optional C++17 ownership wrapper. Neither requires RocketForge, v2 headers, Qt, LibTorch or private simulator APIs.

## Build and upstream adoption

Use Rust 1.95+ and a matching native toolchain. From `RocketSimV3`, run `cargo build --locked --release --manifest-path bindings/Cargo.toml`. The pinned toolchain is in `rust-toolchain.toml`; the minimum is Rust 1.95. The dependency points at `../upstream/rocketsim`. Upstream adoption should relocate this crate and adjust its dependency path/workspace membership. RF's backend, `RLGym/Sim/src/Backend/V3Backend.cpp`, stays outside this package.

The static library is `rocketsim_ffi.lib` on Windows or `librocketsim_ffi.a` on Unix. Link Rust's platform libraries (Windows: ntdll, userenv, ws2_32, dbghelp, bcrypt; Linux: pthread, dl, m).

The build generates a C header in OUT_DIR and verifies it against the checked-in header. After an intentional ABI change, the build fails and prints both versions; copy the generated `OUT_DIR/rocketsim_ffi.h` (under `<target-dir>/<profile>/build/rocketsim_ffi-*/out/`) to `include/rocketsim_ffi.h`, review it, and rebuild consumers. Increment `rsf_abi_version` for incompatible changes; consumers compare it against the version they were built for (`rsf_abi_version()` is currently 2, the version that adds the batched stepping, preserving setters, tile and ball-predictor exports below).

## Usage and contracts

Initialize meshes once before creating arenas. Mode IDs: Soccar=0, Hoops=1, Heatseeker=2, Snowday=3, Dropshot=4, TheVoid=5. Team IDs: Blue=0, Orange=1. Memory modes: Heavy=0, Balanced=1, Light=2. Negative seeds request random seeds. Rotation axes are forward/right/up; distances use Rocket League world units.

Create/free handles through this library. The C++ wrapper owns its handle and is move-only. Inputs must be live and aligned; outputs writable and disjoint from inputs/arena storage. All calls on one arena require exclusive access, including reads. Different arenas can run on different threads. Never free a borrowed handle or use one after its owner is destroyed.

Status functions return 0 on success, -1 on failure. Handles return null and count/index calls use UINT32_MAX on failure. Read `rsf_last_error()` immediately on the same thread. Fallible ABI entry points catch panics; discard an arena after an unexpected engine panic. Invalid caller pointers cannot be made safe by error handling.

Read state before modifying it: a zero-filled C struct is not a native default. State transfer preserves exposed native fields, including jump ticks, contacts and mode state. A setter is a teleport, not an exact rollback of collision caches. Use `rsf_arena_reset_contacts` for episode resets or `rsf_arena_clone` for independent in-memory physics/RNG copies.

Stepping retains events from all requested ticks with absolute tick indices. Zero ticks preserves state and events. Event positions are contact points; ball contacts also record the ball center after their tick. Query count and copy into a caller-owned buffer.

### Batched stepping

`rsf_arena_step_snapshot` sets every car's controls, steps any number of ticks and fills a caller-allocated `RsfStepOutput` in one call:
every car's state, the ball, every pad, the Dropshot tiles and `isBallScored`.
Per car it also reports the last ball touch (`RsfStepCarResult`) and whether the car was on the ground at any point, including the step-start state.
Car-car bumps and demos go to a caller-sized `RsfBump` buffer; `bumpsWritten` counts every bump, so a value above `bumpsCap` means some were not stored.
Touch and bump ticks are relative to the step (0 is its first tick), and a touch's `hitBallPos` is the ball center at the start of its tick.
Buffer counts must equal the arena's car, pad and tile counts. Snapshot steps do not record into the `rsf_arena_get_events` buffer.

`rsf_arena_set_car_state_preserving` and `rsf_arena_set_ball_state_preserving` teleport like the plain setters but keep native state a caller usually cannot map:
the car's jump-hold-broken flag, last extra-hit tick and existing wheel-contact details,
and the ball's `tickCountSinceKickoff`, so a teleport does not restart the Hoops/Dropshot launch delay.

`rsf_arena_num_tiles`, `rsf_arena_get_tiles` and `rsf_arena_set_tiles` read and write the 140 Dropshot tiles as `RSF_TILE_FULL`/`DAMAGED`/`BROKEN` bytes, Blue's 70 then Orange's.

`rsf_ball_predictor_new(mode, no_ball_rot)` owns a ball-only arena. `rsf_ball_predict` fills one ball state per tick and returns the first scoring index, -1 for none or `INT32_MIN` on failure.
Every prediction starts fresh from the given ball and the tiles last passed to `rsf_ball_predictor_set_tiles` (all full by default).
The predictor has no game clock, so `dsLastDamageTick` in its input and output is an age: ticks since the ball last damaged tiles.

## Validation

Set `RSF_TEST_MESHES` to a collision-mesh directory and run `cargo test --locked --manifest-path bindings/Cargo.toml`. Tests compare native public stepping/cloning with the ABI, transfer native car fields, exercise ball mode state/boost cooldowns, compare batch/scalar controls and reject invalid inputs. They also compare snapshot steps, preserving setters, tiles and ball prediction with native tick-by-tick stepping. The Dropshot tile checks run only when the mesh directory has a `dropshot/` folder, and print a notice when it does not. RF's C++ integration tests cover the generic wrapper and the separate facade. Keep the public submodule clean throughout validation.
