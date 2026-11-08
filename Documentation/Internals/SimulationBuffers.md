# Simulation-loop buffer reuse

`EnvSet` keeps separate storage for each arena for the following data:
observations, action masks, actions, applied controls, macro lists, macro indices, and macro durations.
Each arena's current job owns its storage.
The buffers keep their capacity across steps and episode resets,
and they're released with the environment set.
No history accumulates in them;
their capacity follows the largest output or macro list that the arena has produced.

The buffers keep the following call contracts:

- Macro lists are cleared before each parser call,
  so custom parsers still receive an empty list
  and can leave it empty to request the ordinary-action fallback.
- Macro indices restart at zero for each action window.
- Applied controls are overwritten before use, including on the delayed and per-player control paths.
- Observation and mask buffers start empty on each arena observation rebuild.
- Unmasked rows are filled with allowed actions every time, including after masked steps.

For reward buffers, see [Reward evaluation and buffer reuse](RewardEvaluation.md).

## Observation builders

The stock `DefaultObs`, `AdvancedObs` and `AdvancedObsPadded` classes implement `BuildObsInto` with caller-owned storage.
They write the self, teammate, and opponent rows in the same order, with the same arithmetic, as `BuildObs`.
Their stateless player encoders append teammates and opponents directly, without temporary team vectors.

Further subclasses keep virtual `BuildObs` dispatch.
When a subclass calls a stock base's `BuildObs`,
the original `AddPlayerToObs` hook order and temporary-vector contents are preserved,
including for stateful custom hooks.
The `DefaultObsPadded` fallback keeps its shuffle order and random-generator use.
Custom builders that only return a vector stay compatible and can still allocate;
to reuse the supplied capacity, implement `BuildObsInto`.

An in-place observation or mask builder must follow these rules:

- Replace its output with a complete row.
- Don't keep references to caller-owned output storage.

Before you reset or otherwise operate on an arena, synchronize that arena's pending jobs.

## Tests

The `SimulationBuffers` regression tests cover the following behavior:

- Exact stock observation values and ordering.
- Retained capacity.
- Live simulation output compared with the returning observation API.
- Custom overrides and stateful hooks.
- The padded observation's shuffle and random-generator state.
- Parallel 1v1, 2v2, and 3v3 arenas.
- Mask changes, macro changes, and empty fallbacks.
- Resets and delayed controls.
- Recovery after an observation builder throws an exception.

The arena and reward regression tests also exercise the shared stepping paths.

Validation record, October 17, 2026:
Release builds of the GUI, engine host, console trainer, and test target succeeded for both simulators.
The focused regression selection passed 94 tests on v2 and 98 on v3, without skips,
including the separately run live-observation test.

## Measured performance

These measurements were taken on October 17, 2026,
against public revision `afca37df318f9c893989d167bfe17907a90b9af2`,
on Windows with MSVC 19.44 Release builds.
The standalone harness used one arena, one collection thread, an eight-tick action window, stock observations,
and seven samples after warm-up.
Ordinary stepping used 2,000 steps per sample;
observation-only measurements rebuilt all six players' rows 20,000 times.

| Simulator | Workload | Before (ms) | After (ms) | Time reduction |
| --- | --- | ---: | ---: | ---: |
| v2 | 1v1 stepping, AdvancedObs | 55.42 | 53.44 | 3.6% |
| v2 | 2v2 stepping, AdvancedObs | 96.34 | 94.25 | 2.2% |
| v2 | 3v3 stepping, AdvancedObs | 141.91 | 130.81 | 7.8% |
| v3 | 1v1 stepping, AdvancedObs | 24.31 | 20.17 | 17.0% |
| v3 | 2v2 stepping, AdvancedObs | 45.60 | 34.54 | 24.2% |
| v3 | 3v3 stepping, AdvancedObs | 65.17 | 47.47 | 27.2% |
| v2 | 3v3 AdvancedObs only | 186.66 | 59.27 | 68.2% |
| v3 | 3v3 AdvancedObs only | 177.26 | 57.69 | 67.5% |
| v2 | 3v3 DefaultObs only | 153.92 | 39.99 | 74.0% |
| v3 | 3v3 DefaultObs only | 165.48 | 40.22 | 75.7% |

Warm observation rebuilds dropped from 223 C++ allocations (AdvancedObs) or 211 (DefaultObs) to zero.
On v2, ordinary stepping dropped from 52, 148, and 232 allocations for 1v1, 2v2, and 3v3 to one.
On v3, it dropped from an average of 52.90, 148.95, and 232.95 to 1.90, 1.95, and 1.95.
These counts cover C++ `operator new` calls on the calling thread,
not Rust allocations or total process memory.
Custom builders that only return a vector can still allocate,
and more complicated control paths still allocate state copies.

The 3v3 macro-action case measured 2.0% longer on v2 and 22.9% shorter on v3.
The delayed per-player-control case measured 3.6% shorter on v2 and 20.4% shorter on v3.
Treat small timing changes as noisy local measurements.
These runs include physics, action parsing, masks, and observations,
but no policy inference, reward evaluation, or optimizer work,
so they aren't estimates of training speedup.

### Trace comparison

The harness also compared observation, mask, control, and physics traces through 256 steps with repeated resets.
All seven v3 comparisons, and their final timing checksums, matched exactly.

v2 requires normalizing fixture car identities and teams, because its car collection is unordered.
The final bounded v2 trace checks also matched.
However, repeated runs of the unchanged v2 baseline showed small differences in the bounded macro trace,
and they diverged over long continuous physics trajectories.
v2's long-run checksums therefore aren't a bitwise compatibility guarantee,
and its stepping timings can reflect differing contact histories.
The stock-builder unit tests also compare observation values bit for bit for the same input, including live states,
without relying on cross-process physics determinism.
