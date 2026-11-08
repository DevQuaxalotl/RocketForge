# Public inference agents

This folder packages three public RLBot bots, **Necto**, **Nexto**, and **Element**, as ordinary RocketForge agents.
Use them as opponents for your own bot:

- As participants in [model benchmarking](../../Documentation/Features/ModelBenchmarking.md),
  to measure your bot against known public bots.
- As frozen static opponents during training.
  To choose them, click **Static Opponents…** on ForgeGUI's **Training** tab,
  and turn on `staticOpponentsEnabled` for the stage in [`forge_settings.json`](../../Documentation/CLI/SettingsFiles.md).

These agents are inference-only.
Their packaged policies don't include a critic, optimizer, or training state,
so you can't resume or fine-tune them in RocketForge.

## Where the models come from

The models come from the RLBot bot pack at
[`RLBot/RLBotPack@0033a98d`](https://github.com/RLBot/RLBotPack/tree/0033a98d4e060670f334f60f7f8dc5ee21ba93bb).
Each model folder records the artifact's hash and its source.

| Bot | Observations and actions | Conversion |
|---|---|---|
| Necto | The published attention observations and discrete action table. | Converted to a GPU-portable attention policy with `tools/convert_attention_models.py`. The unmodified published TorchScript file is kept next to it for parity testing. |
| Nexto | The published attention observations and discrete action table. | Same as Necto. |
| Element | The published 1v1 pairing observation and factorized action policy. | Its Python pickle is converted to TorchScript with `tools/convert_element_model.py`, without changing its deterministic per-head action choice. |

To reproduce a conversion, run the script against the pinned bot-pack snapshot.
All three policies run on the CPU or with CUDA, including RocketForge's CUDA-graph inference path.

## Behavior

All three bots act on their published 8-tick policy cadence.

- Necto and Nexto include their published 168-tick recorded kickoff sequence and their kickoff-taker selection.
- Element includes its published speed-flip kickoff state machine.

These kickoff controllers run on every physics tick in RLBot, in model benchmarks, and as frozen static opponents in training.

## Licenses

These bots keep their own licenses; RocketForge's [MIT License](../../LICENSE) doesn't replace them.

- **Necto and Nexto** are licensed under [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/).
  Their policies, and RocketForge's ports of their observations and actions, are for non-commercial use only,
  and any adaptation must keep the same license.
  The license text is in `LICENSES/Necto-Nexto-CC-BY-NC-SA-4.0.txt`.
- **Element** and the surrounding RLBotPack source are licensed under the RLBotPack MIT license,
  reproduced in `LICENSES/RLBotPack-MIT.txt`.

For every third-party component in RocketForge, see [THIRD_PARTY_NOTICES.md](../../THIRD_PARTY_NOTICES.md).
