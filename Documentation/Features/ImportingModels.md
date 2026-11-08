# Importing models

An *imported model* is a trained policy copied into an agent from outside its own builds:
an old checkpoint from another run or machine, or the policy of a bot released somewhere else.
Once imported, it can be used in three places:

- as a frozen **static opponent** in training (Training tab, **Static Opponents...**);
- in **Model Benchmark**, alongside the agent's own segments;
- in **Debug render**, by choosing the agent with no build selected.

Imported models are frozen: they can't be trained further yet.

## What an import needs

Import doesn't convert anything.
The agent you import into builds its own networks and loads the model's weights into them,
so the agent must match the model:

- the same observation builder, so the shared head gets the same number of inputs;
- the same action parser, so the policy has the same number of outputs;
- the same shared head and policy layer sizes, and the same activation and layer norm settings;
- the same tick skip and action delay, or the model plays at the wrong rate.

If the agent doesn't match, the import still works,
but the model fails to load until you change the agent to match:
training with it selected as a static opponent stops at startup with a load error,
and Model Benchmark and Debug render can't load it.
The import window checks this for you before you import.

A static opponent always plays through the agent it was imported into,
so a model imported into one agent can be an opponent while you train a different agent.

## Import a model

In the GUI, open **File Management** and press **Import Model...** under **Storage**.

1. **Agent:** the agent that will use the model.
2. **Model folder:** press **Browse...**, or paste a path and press Enter, for one of these:
   - a folder that holds the model files, `POLICY.lt` and `SHARED_HEAD.lt`,
     or `INFERENCE_POLICY.json` for a scripted policy;
   - a folder of numbered checkpoints, such as an old GigaLearn run folder;
     the newest checkpoint is used;
   - a RocketForge segment folder, `Models/<build>/segments/<id>_<stage>/`, or one checkpoint inside it;
     the newest checkpoint is used.
3. **Name:** shown in model lists. The folder name is made from it.
   It starts as the build and segment for a RocketForge checkpoint, or the folder's name otherwise,
   skipping generic folder names such as `rlbot` (so a bot's `FuryBotpack/rlbot` becomes `FuryBotpack`).
   **Saves to** shows exactly where the model is copied:
   `<agent's Models folder>/Imported Models/<name>/`.
4. Read the summary box:
   - **Files to copy** and their total size.
   - **Usable as:** whether the folder has a policy that can play.
   - Whether the critic, optimizers and running stats are included.
   - Whether the model loads into the agent's current networks.
     When it doesn't, the reason names the file and layer,
     for example `POLICY.lt layer 0.weight is 1024 x 1024 in the file, but 512 x 295 in example-agent's network.`
5. Optionally, open **Where it came from** to record the model's **Build**, **Made from** and **Stage**, and notes.
   These are labels for your reference: they show in the matching columns of File Management and in Model Benchmark,
   and they don't change how the model is used.
   For a RocketForge segment they're filled in from the segment's `segment.json`.
   For any other folder they start empty, with **Made from** set to `1.0`.
6. Press **Import**.
   If a model with that name already exists, tick **Replace the existing model with this name** or choose another name.

Only model files are copied, flat into the new folder:
`POLICY.lt`, `SHARED_HEAD.lt`, `CRITIC.lt`, `AUX_CRITIC.lt`, their `*_OPTIM.lt` optimizer files,
`RUNNING_STATS.json`, `INFERENCE_POLICY.json`, `agent_setup.json`,
and the weight files that `INFERENCE_POLICY.json` names, when they sit beside it.
Anything else in the folder, such as `bot.toml`, a logo, `policy_versions/` or older checkpoints, is left behind.
The import also writes `imported_model.json`, which records the source and what the model can be used for.

You can also copy a model folder into `Imported Models/` by hand.
It's picked up without `imported_model.json`, but then has no name, source or notes.

## Import an old checkpoint

When the checkpoint was trained with an agent that still exists, or one you've recreated:

1. Make sure the agent's observation builder, action parser, network sizes and tick settings match the checkpoint.
2. Import the checkpoint's folder, or the run folder that holds it, into that agent.

The import copies the critic and optimizer files too when the checkpoint has them.

## Import a bot from somewhere else

A bot released elsewhere, such as an RLBot bot built on GigaLearn,
usually ships its policy as `POLICY.lt` and `SHARED_HEAD.lt` next to its RLBot files.
RocketForge needs an agent that sees and acts exactly like that bot:

1. Create a bot for it (see [Adding a bot](../AddingABot.md)), for example `ForgeConsoleTrainer create-bot Fury`.
2. Port the bot's observation builder and action parser into the new bot,
   and register them with `REGISTER_OBS_BUILDER` and `REGISTER_ACTION_PARSER`.
   Port any kickoff or other per-tick controller the bot uses as well.
3. In the agent's constructor (`Bots/<Name>/<Name>Agent/src/<Name>Agent.cpp`),
   copy the bot's settings: `obsBuilderName`, `actionParserName`, `tickSkip`, `actionDelay`, `maxTeamSize`,
   `sharedHead.layerSizes`, `policy.layerSizes`, and the activation and layer norm settings.
   The bot's own source usually sets these where it creates its inference model.
4. Re-configure and rebuild, so the new agent exists.
5. Import the bot's model folder into the new agent.
   The import window should report that the model loads into the agent's networks.
   If it doesn't, the reason names the mismatch, such as the number of inputs or the layer sizes.

Check the bot's license before you use or share its model.

### Policies that aren't GigaLearn checkpoints

The packaged RLBotPack bots, **Necto**, **Nexto** and **Element**, show how to bring in a policy from another framework.
Each is its own agent with ported observations and actions,
and its model is converted to TorchScript and described by an `INFERENCE_POLICY.json`.
See `Bots/PublicBots/README.md` and the conversion scripts in `Bots/PublicBots/tools/`.

## From the console

The console can list and use imported models, but not import them yet:

- `ForgeConsoleTrainer list imported-models --agent <agent>` prints each model's folder name.
- `ForgeConsoleTrainer debug --agent <agent> --imported <folder> --stage <stage>` renders one.

For more information, see [Console trainer](../CLI/ConsoleTrainer.md).
