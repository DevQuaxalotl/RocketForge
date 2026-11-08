# RocketForge bot package

This package holds one RocketForge bot, ready to build with Bob the Bot Builder for the RLBot bot pack.
It contains only the bot's inference sources, its checkpoint, and the build scripts,
and it doesn't need a RocketForge installation.

Team batching (hive mind) is on by default:
teammates running this bot share one process and one batched policy call per decision.

## Build the package

The device in `rlbot/device.txt` decides what you build: `cpu` or `gpu`.

### CPU package

Run a plain `bob build`, the same way the RLBot bot pack builds bots.
Without a staged LibTorch, the Docker build uses the bot pack's Windows CPU LibTorch archive.

### GPU package, or a LibTorch of your own

Use the build helper. It needs Bob and Docker:

```bat
python scripts/build_bob.py
```

The helper stages the LibTorch headers and import libraries for the selected device from the shared RLBot bot pack,
and fingerprints them so that Bob's cache rebuilds when they change.
GPU packages must be built this way.

To build against a different LibTorch, pass its folder:

```bat
python scripts/build_bob.py --libtorch "C:/path/to/libtorch"
```

You can also select the SDK with the `LIBTORCH_CUDA_ROOT` (GPU) or `LIBTORCH_CPU_ROOT` (CPU) environment variable,
which also applies when the bot runs.
Use the same LibTorch version as the machine that runs the bot.
Only headers and import libraries enter the build container;
the LibTorch DLLs stay in the shared bot pack, and the package never bundles them.

### Native Windows build

To build without Bob, configure a Release or RelWithDebInfo build,
and point CMake at the LibTorch for the selected device with `-DLIBTORCH_ROOT=<path>`.
Put the launcher next to `bot.toml`,
and put the matching runtime executable, the models, `agent_setup.json`, the prediction libraries, and the collision meshes in `000-runtime`.
Bob arranges this layout automatically.

## Choose the device

Set `rlbot/device.txt` to `cpu` or `gpu` before you build, and rebuild after you change it.

- **CPU** packages run on Windows and Linux.
  The RLBot bot pack builds only CPU packages.
- **GPU** packages run on Windows with a compatible NVIDIA GPU and driver.
  At startup, the bot checks CUDA and warms up the policy before RLBot sends live packets.
  If CUDA isn't available, the bot exits with an error; it never falls back to the CPU.

### Where the bot looks for LibTorch

On Windows, a GPU package looks for `libtorch/lib` in nearby bot-pack folders,
and then in `%LOCALAPPDATA%/RLBot5/bots/libtorch/lib`.
A CPU package uses `torch-archive/torch/lib`.
A `LIBTORCH_CUDA_ROOT` or `LIBTORCH_CPU_ROOT` override takes precedence, and must point at a valid LibTorch.

The launcher starts only the runtime that matches the device:
`000-runtime/RLBotCoreCUDA.exe` or `000-runtime/RLBotCoreCPU.exe`.
It doesn't download any runtime DLLs.

The build and launch layout follows the public [GGLBot bob-gpu branch](https://github.com/SubparN0va/GGLBot/tree/bob-gpu).

## Turn off team batching

Set `hivemind = false` in `rlbot/bot.toml`, and change `--hivemind on` to `--hivemind off` in its run commands.
Keep the two settings consistent.
Team batching doesn't depend on the device.

## Record a match for diagnosis

You can record what the bot sees and decides during a match, and replay it offline.
Recording is off by default, and it adds overhead, so use it only for diagnostic matches.

1. Append `--capture-dir "C:/rlbot-captures/match"` to the bot's `run_command` in `bot.toml`.
1. Play the match, and stop the bots normally from RLBot, so that each recording gets its footer.
1. Remove `--capture-dir` again.
1. Replay the recording offline, with a launcher for the matching device:

   ```bat
   RLBot.exe --replay "C:/rlbot-captures/match/capture-ID/callbacks.rfcapture" --model-dir "000-runtime" --capture-dir "C:/rlbot-captures/replay" --device gpu
   ```

1. Compare the recording with the replay:

   ```bat
   python scripts/rlbot_capture.py compare "C:/rlbot-captures/match/capture-ID/callbacks.rfcapture" "C:/rlbot-captures/replay" --html "C:/rlbot-captures/report.html"
   ```

Keep recordings outside this package; they contain game packets and player names.

`--hivemind off` replays team-recorded cars one at a time.
`--allow-incomplete` accepts a recording whose footer is missing,
but never a corrupt recording or one with missing callbacks.
A replay checks the bot's decisions against the recorded inputs; it doesn't simulate the effect of different controls.
