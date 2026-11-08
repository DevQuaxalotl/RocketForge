# RocketForge documentation

This folder holds RocketForge's user and developer documentation.
For an overview of the project, see the [project README](../README.md).

## Getting started

- [Get started with RocketForge](GettingStarted.md): install the prerequisites, build, train the ExampleBot, and watch it in RocketSimVis and Rocket League.

## Concepts

- [Model versioning](Concepts/ModelVersioning.md): builds, training segments, branches, and the `Models/` storage layout.
- [Phasic Policy Gradient](Concepts/PPGAlgorithm.md): how PPG differs from PPO, and how RocketForge runs its two phases.

## Features

- [Adding a bot](AddingABot.md): create a bot with the GUI or create-bot, or by hand, and keep it in its own repository.

- [Training without an NVIDIA GPU](Features/CpuOnly.md): set up a CPU-only build, what it can't do, and how much slower it trains.

Training setup:

- [Training stages](Features/TrainingStages.md): stage structure, team sizes, state setters, terminal conditions, and learner settings.
- [Rewards](Features/Rewards.md): writing rewards, zero-sum and multi-part rewards, and the reward cache.
- [Adaptive reward weights](Features/AdaptiveRewardWeights.md): weights that tune themselves toward a target production rate.
- [Transfer learning](Features/TransferLearning.md): train a new policy to imitate a trained one after you change its observations, actions, or network.

Running and inspecting training:

- [Training sessions](Features/TrainingSessions.md): stopping, skipping, the per-build writer lock, and what checkpoints keep.
- [Training dashboard](Features/TrainingDashboard.md): what each dashboard card shows, and how to read the skill rating.
- [Model catalog](Features/ModelCatalog.md): how the GUI's model lists stay current, refreshing after manual changes, and the cache files.
- [Importing models](Features/ImportingModels.md): bringing an old checkpoint or another bot's policy into an agent.

Simulation and evaluation:

- [Simulation engines](Features/SimulationEngines.md): build RocketSim v3, choose an engine and ball predictor, and the known engine differences.
- [Model benchmarking](Features/ModelBenchmarking.md): run a benchmark, read and reuse its results, and run it from the console.

Playing in Rocket League:

- [RLBot](Features/RLBot.md): build the RLBot runner, export a segment, add it to RLBot, team batching, and bot-pack submission packages.

## Console trainer

- [Console trainer](CLI/ConsoleTrainer.md): every `ForgeConsoleTrainer` command, the shared options, stop and skip, and common workflows.
- [Settings files](CLI/SettingsFiles.md): the JSON files the console reads, field by field.

## Remote training

- [Remote training](RemoteTraining/RemoteTraining.md): train on a rented cloud GPU with Docker, Cloudflare R2, and W&B.
- [Remote training on Vast.ai](RemoteTraining/VastAI.md): deployment and self-termination on Vast.ai (recommended).
- [Remote training on RunPod](RemoteTraining/RunPod.md): deployment and self-termination on RunPod.

## Internals

- [Reward evaluation and buffer reuse](Internals/RewardEvaluation.md): the batch reward API and its buffer contract.
- [Simulation engine internals](Internals/SimulationEngines.md): the two backends behind one API, shared ball prediction, and how saved engine settings load.
- [Simulation-loop buffer reuse](Internals/SimulationBuffers.md): per-arena buffers and the in-place observation builder contract.

## Development

- [Code standards](CodeStandards.md): naming, comments, layout, tests, and CMake conventions for all code.
- [Performance testing](PerformanceTesting.md): measuring training steps per second with `training-benchmark`.

## Tools and components

These guides live next to the code they describe:

- [Public inference agents](../Bots/PublicBots/README.md): the packaged Necto, Nexto, and Element policies, and their licenses.
- [RocketSimVis](../ForgeTraining/ForgeDebugger/RocketSimVis/README.md): the bundled viewer, its keyboard controls, and its network ports.
- [RLBot match recording and offline replay](../tools/rlbot/capture/README.md): record RLBot matches and compare inference across devices.
- [Parity probe](../tools/rlbot/parity_probe/README.md): check that two policy runtimes make the same decision from the same game state.

## Other files

- `Diagrams/`: draw.io diagrams of the module dependency graph and the file architecture.
- `ValidationRecords/`: JSON records of earlier model-benchmark validation runs, kept for reference.
