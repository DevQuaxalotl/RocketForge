# Adding a bot

A bot is a folder under `Bots/` that holds one agent, its training plan (the stage queue's code) and an RLBot folder.
New bots start as a renamed copy of `Bots/ExampleBot`.

## Create a bot

- GUI: open the Training tab, press **New Bot** next to the agent selector, and enter a name.
  A failed check shows its reason in a "Cannot create bot" box and writes nothing.
- Console: `ForgeConsoleTrainer create-bot <Name>` (see [`create-bot`](CLI/ConsoleTrainer.md#create-bot)). It prints `Created <path>` and the next step.

Then re-configure and rebuild (`cmake --preset forge`, or `tools/build/build_and_test.ps1`).
`Bots/CMakeLists.txt` finds bot folders only when CMake configures, and each bot's agent is compiled into the executables, so the new agent does not exist until you do.
After the rebuild, select the agent `<Name>` in the GUI, or run `ForgeConsoleTrainer train --agent <Name>`.

## What you get

For a bot named `MyBot`:

```text
Bots/MyBot/
  CMakeLists.txt
  MyBotAgent/
    CMakeLists.txt
    forge_settings.example.json
    rlbot/
      bot.toml
      device.txt
      loadout.toml
      run.bat
    src/
      MyBotAgent.cpp
      MyBotAgent.h
  MyBotTrainingPlan/
    CMakeLists.txt
    src/
      MyBotTrainingPlan.cpp
      MyBotTrainingPlan.h
```

The agent is `MyBot`, with the stages `mybot-ball-chase`, `mybot-basic-score` and `mybot-advanced`.

Deliberately not copied: `Models/`, `forge_settings.json`, `forge_debug_settings.json`, and the `.exe` and `.dll` files in `rlbot/`.
These are local training state and build outputs. On first run the trainer creates `forge_settings.json` from `forge_settings.example.json`.

## Naming rules

A name is 1 to 32 ASCII letters and digits and starts with a letter.
It cannot be `Example` in any capitalisation, because that is the template's name.
It is also refused when:

- a folder in `Bots/` already has that name, ignoring case;
- any bot already has an `<Name>Agent` or `<Name>TrainingPlan` folder, ignoring case, because each folder registers a CMake target named after itself;
- an agent named `<Name>` is already registered, ignoring case.

## Doing it by hand

1. Copy only these 12 files from `Bots/ExampleBot` into `Bots/<Name>`, keeping the folder structure:
   `CMakeLists.txt`; `ExampleAgent/` `CMakeLists.txt`, `forge_settings.example.json`, `rlbot/bot.toml`, `rlbot/device.txt`, `rlbot/loadout.toml`, `rlbot/run.bat`, `src/ExampleAgent.cpp`, `src/ExampleAgent.h`;
   `ExampleTrainingPlan/` `CMakeLists.txt`, `src/ExampleTrainingPlan.cpp`, `src/ExampleTrainingPlan.h`.
2. Rename the paths and the file contents in a single pass, so replaced text is never replaced again:
   `example-agent` and `ExampleBot` become `<Name>`, `Example` becomes `<Name>`, `example` becomes `<name>` (lower case) and `EXAMPLE` becomes `<NAME>` (upper case).
   Keep the file name `forge_settings.example.json` as it is.
3. In `<Name>Agent/rlbot/bot.toml`, set `agent_id` to `RocketForge/<Name>` and `name` to `<Name>`.
   They hold the placeholder `RFBot`, which the renaming does not touch.
4. Re-configure and rebuild.

## Team sizes

A stage appears in the GUI and in `list stages` only for agents that can play it: its `StageCompatInfo::minAgentTeamSize` (the third `REGISTER_STAGE` argument) must not exceed the agent's `maxTeamSize`.
ExampleBot's agent is 1v1 (`maxTeamSize = 1`), so a stage that trains 3v3 also needs `maxTeamSize = 3` in `<Name>Agent/src/<Name>Agent.cpp`,
along with the stage's `weightedTeamSizes` and `minAgentTeamSize` in `<Name>TrainingPlan/src/<Name>TrainingPlan.cpp`.
ExampleBot's `AdvancedObsPadded` pads missing players to `maxTeamSize`, so every arena in a run produces the same obs size without changing the obs builder.
Changing the obs builder or `maxTeamSize` changes the network's input size, so start a new build for it.
See [Training stages](Features/TrainingStages.md).

## Organising your bot's code

- Put custom rewards in `<Name>TrainingPlan/src/Rewards/` and custom state setters in `<Name>TrainingPlan/src/StateSetters/`.
  The training plan's CMake file picks up every `.cpp` and `.h` under `src/`, but a new file needs a re-configure first.
- Put your classes in a bot namespace, such as `namespace MyBot`.
  Every bot links into the same executables, and two global classes with the same name collide without any error.
- Rewards shared by several bots, such as `RLGC::BoostReward`, live in `ForgeTraining/RLGym/src/RLGymCPP/Rewards/CommonRewards.h`.

## Keeping a bot in its own git repository

RocketForge's `.gitignore` ignores `Bots/*/` except `ExampleBot` and `PublicBots`, so a new bot is never part of the RocketForge repository.
Run `git init` inside `Bots/<Name>` and push it to any remote.
On another machine, clone it into `Bots/` and re-configure.

Keep local training state and binaries out of the bot's repository with a `.gitignore` in `Bots/<Name>`:

```gitignore
# Training state and model checkpoints
*/Models/
*/forge_settings.json
*/forge_debug_settings.json

# RLBot executables and libraries
*/rlbot/*.exe
*/rlbot/*.dll
```

`forge_settings.example.json` stays tracked, so a fresh clone can create its own `forge_settings.json`.
