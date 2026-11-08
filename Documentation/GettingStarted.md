# Get started with RocketForge

This guide takes you from a fresh clone to a trained bot that you can watch.
You install the prerequisites, build RocketForge, train the ExampleBot's ball-chase stage,
watch the result in the RocketSimVis viewer, and then play it in Rocket League through RLBot.

You can do every step in ForgeGUI or with the console trainer, `ForgeConsoleTrainer`.
Both read and write the same settings files, so you can switch between them at any point.
Each step below gives both ways.

## Before you begin

You need the following:

- Windows 10 or 11, 64-bit.
  The console trainer also runs on Linux, in the training image; for more information, see [Remote training](RemoteTraining/RemoteTraining.md).
- Free disk space for the toolchain, libtorch, and the build:
  the CUDA build of libtorch alone takes 7.7 GB, and the CPU build about 1 GB.
- An NVIDIA GPU is optional.
  Without one, RocketForge trains on the CPU, more slowly.
- [Git](https://git-scm.com/download/win).
- To play your bot in Rocket League: Rocket League and [RLBot v5](https://rlbot.org/).

## Get the code

Clone the repository with its submodules:

```powershell
git clone --recursive https://github.com/DevQuaxalotl/RocketForge.git
cd RocketForge
```

If you already cloned without `--recursive`, initialize the submodules:

```powershell
git submodule update --init --recursive
```

## Install the prerequisites

Each script checks for an existing install first and asks before it installs anything.
Run them from the repository root.

1. Install the C++ compiler, CMake, and Ninja (Build Tools for Visual Studio 2022).
   Skip this step if Visual Studio 2022 already has the **Desktop development with C++** workload.

   ```powershell
   powershell -ExecutionPolicy Bypass -File .\tools\setup\install_build_tools.ps1
   ```

1. Install Python 3.12.
   The build embeds Python, and the viewer runs on it.
   Any Python from 3.9 to 3.12 on `PATH` works instead.

   ```powershell
   powershell -ExecutionPolicy Bypass -File .\tools\setup\install_python.ps1
   ```

1. If you have an NVIDIA GPU, install the [CUDA 12.6 toolkit](https://developer.nvidia.com/cuda-12-6-0-download-archive).
   Without one, skip this step.
1. Install libtorch 2.13.0.
   The script picks the CUDA build when it finds an NVIDIA GPU, and the CPU build otherwise.
   To choose, add `-Variant Cuda` or `-Variant Cpu`.

   ```powershell
   powershell -ExecutionPolicy Bypass -File .\tools\setup\install_libtorch.ps1
   ```

1. To use ForgeGUI, install Qt 6.10.2.
   Without Qt, the build produces only the console trainer.

   ```powershell
   powershell -ExecutionPolicy Bypass -File .\tools\setup\install_qt.ps1
   ```

1. Open a new terminal, so that `PATH` includes the new tools.

### Use your own installs

If you already have these tools, or want them in other folders, point the build at them:

- **libtorch:** the build looks in `%LOCALAPPDATA%\libtorch`, the folder that holds `share\cmake\Torch`.
  For another folder, pass `-DLIBTORCH_ROOT=<path>` when you configure.
- **Qt:** the build looks in `C:\Qt\6.10.2\msvc2022_64`.
  For another folder, set the `QTDIR` environment variable to it.
- **Python:** any Python from 3.9 to 3.12 on `PATH`.

The prebuilt libtorch and Qt binaries are MSVC builds, so build RocketForge with the Visual Studio 2022 compiler.

## Build RocketForge

Build in one of the following ways:

- **Build script:** from the repository root, run:

  ```powershell
  powershell -ExecutionPolicy Bypass -File .\tools\build\build_and_test.ps1 -BuildGui
  ```

  The script sets up the Visual Studio x64 developer environment, configures the `forge` preset,
  builds and runs the tests, and then builds `ForgeConsoleTrainer` and, when Qt is installed, `ForgeGUI`.
- **Visual Studio or Rider:** open the repository folder, select the **forge** configure preset,
  and build with the **GUI + Console** build preset.

The first build takes a while, because it compiles RocketSim and the training framework.
Both executables land in `out\build\forge`.

To check the build, list the ExampleBot's agent and stages:

```powershell
out\build\forge\ForgeConsoleTrainer.exe list agents
out\build\forge\ForgeConsoleTrainer.exe list stages
```

The output includes the agent `example-agent` and the stages `example-ball-chase`, `example-basic-score`, and `example-advanced`.

To start ForgeGUI, run `out\build\forge\ForgeGUI.exe`.
The console examples in the rest of this guide write `ForgeConsoleTrainer` for `out\build\forge\ForgeConsoleTrainer.exe`;
run them from the repository root.

## Train the ExampleBot

The ExampleBot lives in `Bots/ExampleBot`.
Its training queue starts with the `example-ball-chase` stage,
which teaches a 1v1 car to drive to the ball and hit it, and which trains to 100 million total timesteps.
The first time you open the agent, RocketForge creates `Bots/ExampleBot/ExampleAgent/forge_settings.json`
from `forge_settings.example.json`.

The example settings leave training metrics off.
To send them to [Weights & Biases](https://wandb.ai/), run `wandb login` once,
and then turn metrics on as described below.

### With ForgeGUI

1. Open the **Training** tab.
1. Under **Agent & Build**, select `example-agent` as the **Agent**.
   Keep the default **Build**, or click **New Build** to start a separately named one.
1. Check the **Training Queue**.
   It shows the `example-ball-chase` stage with its timestep target.
1. Optional: under **Global Settings**, select **Metrics** to send metrics to Weights & Biases.
1. Click **Start Training**.
   The **Training Stats** panel shows each iteration's progress, and **Console Output** shows the log.
1. When you're ready, click **Stop and Save**.
   The trainer saves a checkpoint and stops.
   The next **Start Training** continues from that checkpoint.

Closing ForgeGUI also ends training, after it asks whether to save first.

### With the console trainer

1. Preview what the queue does, without changing anything:

   ```powershell
   ForgeConsoleTrainer train --agent example-agent --preview
   ```

1. Start training:

   ```powershell
   ForgeConsoleTrainer train --agent example-agent
   ```

   The trainer prints its progress after each iteration.
1. When you're ready, press `Q`, or press Ctrl+C once.
   The trainer saves a checkpoint and stops.
   The next `train` run continues from that checkpoint.

To turn on metrics, set `metricsEnabled` to `true` in `forge_settings.json`.

Either way, checkpoints go under `Bots/ExampleBot/ExampleAgent/Models/`.
For what the queue fields mean, see [Settings files](CLI/SettingsFiles.md).
For how builds, segments, and checkpoints are stored, see [Model versioning](Concepts/ModelVersioning.md).

## Watch your bot in RocketSimVis

RocketSimVis is the bundled 3D viewer.
It plays your model in the same simulator that trained it, so you don't need Rocket League for this step.
The first time it starts, it installs its Python dependencies, which takes a minute.

### With ForgeGUI

1. Open the **Debug** tab.
1. Select the agent, the build, and the segment to watch.
1. Optional: under **Render Options**, change the game speed or force a state setter or team size.
1. Click **Start Debug Render**.
   The viewer opens in its own window.
   ForgeGUI also shows each reward and the critic's value while it renders.
1. Click **Stop** to end the render.

### With the console trainer

1. Render the newest checkpoint of the build's active segment:

   ```powershell
   ForgeConsoleTrainer debug --agent example-agent
   ```

   The viewer opens in its own window.
1. Optional: to speed up the game, add `--speed 2`.
   To watch an older checkpoint, find its timesteps with `list checkpoints`, and pass them with `--checkpoint`.
1. Press `q` in the viewer, or Ctrl+C in the terminal, to stop.

The viewer's keyboard controls are listed in the [RocketSimVis README](../ForgeTraining/ForgeDebugger/RocketSimVis/README.md#controls).

## Play your bot in Rocket League

RLBot runs your trained model as a bot in a real Rocket League match.
An *export* turns one segment into a bot folder that RLBot can launch.

1. Build the RLBot runner with the `build-rlbot-all-agents` preset.
   In Visual Studio or Rider, select the **RLBot** configure preset and the **RLBot, copied into every agent** build preset.
   On the command line, in an **x64 Native Tools Command Prompt for VS 2022**, run:

   ```powershell
   cmake --preset rlbot
   cmake --build --preset build-rlbot-all-agents
   ```

   The preset updates every agent's `rlbot` folder with the new runner, so you don't copy anything yourself.
1. Export the segment.
   - **ForgeGUI:** open the **File Management** tab, select the segment, and click **Export to RLBot**.
   - **Console:** export the build's active segment:

     ```powershell
     ForgeConsoleTrainer export-rlbot --agent example-agent
     ```

   The export writes a bot config to `Bots/ExampleBot/ExampleAgent/rlbot_exports/`.
1. In RLBot, open the **Add/Remove** dialog, and add the agent's `rlbot_exports` folder:
   `Bots\ExampleBot\ExampleAgent\rlbot_exports`.
   You do this once per agent; later exports of the agent appear automatically.
1. Find the bot in RLBot's bot list.
   Its name combines the agent, the build, and the segment's timesteps.
1. Add the bot to a match and start it.

The bot runs on the GPU by default.
To run it on the CPU, set `Bots/ExampleBot/ExampleAgent/rlbot/device.txt` to `cpu` before you export.

For the inference device, team batching, troubleshooting, and packaging a bot for the RLBot bot pack,
see [Play your bot in Rocket League with RLBot](Features/RLBot.md).

## What's next

- Continue the ExampleBot's queue with the `example-basic-score` stage.
  In ForgeGUI, add it to the **Training Queue**.
  In `forge_settings.json`, add it to `stages`, as shown in [Settings files](CLI/SettingsFiles.md#forge_settingsjson).
- [Create your own bot](AddingABot.md) from the ExampleBot template.
- Learn how stages, rewards, and state setters fit together in [Training stages](Features/TrainingStages.md) and [Rewards](Features/Rewards.md).
- Compare models against each other with [Model benchmarking](Features/ModelBenchmarking.md).
- Train on a rented cloud GPU with [Remote training](RemoteTraining/RemoteTraining.md).
- Browse the rest of the guides in the [documentation index](README.md).
