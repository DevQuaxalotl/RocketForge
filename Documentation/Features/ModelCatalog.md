# Model catalog

The *model catalog* is ForgeGUI's index of an agent's builds, segments, and checkpoints.
The **File Management**, **Param Testing**, and **Model Benchmarking** tabs all list models from it,
so they always agree with each other.

This page explains how the catalog stays current, what to do after you change model folders by hand,
and which files you can safely delete.

## How the catalog stays current

You don't need to refresh the catalog while you train.
Each checkpoint save updates the segment's and the build's metadata directly,
and the catalog picks up the change in the background.
It also watches the agent's `Models` folder for file changes, and periodically checks it,
so the model lists follow changes made by another ForgeGUI window, the console trainer, or a remote-training pull.

The lists keep your selected rows, column widths, and baseline choices across refreshes where they can.
While a confirmation dialog is open in **File Management**, its list waits to refresh until you close the dialog.

Imported models are discovered separately, and aren't part of the catalog.

## Refresh after you change model folders by hand

If you copy a segment folder into a build, or otherwise change the `Models` folder while ForgeGUI isn't running,
click **Refresh Files** in **File Management**.
It reconciles the build metadata with the folders on disk in the background,
so the copied segments appear with their checkpoints and totals.

Browsing the lists never writes to the model metadata;
only training, **Refresh Files**, and the model-management actions do.

## Checkpoints and autosaves

A training segment has two kinds of saves:

- *Checkpoints*, in the segment's `checkpoints/` folder, are permanent.
  They appear in the checkpoint lists, and they count toward the build's totals.
- *Autosaves*, in the segment's `temp_checkpoints/` folder, are temporary.
  You can resume training from an autosave and render it, but it doesn't appear in the permanent checkpoint list.

To keep every autosave as a permanent checkpoint, turn on `persistSegmentSnapshotsDuringTraining` in [`forge_settings.json`](../CLI/SettingsFiles.md#globals).
For the full folder layout, see [Model versioning](../Concepts/ModelVersioning.md#storage-layout).

## Keep model metadata safe

RocketForge protects the metadata files in the following ways:

- It replaces `builds.json`, `build.json`, and each `segment.json` atomically,
  so a crash never leaves a half-written file.
- One process at a time updates a build's metadata.
  The others wait for it.
- If a process stops between updating a segment and updating its build,
  the next training run rebuilds the build's totals before it continues.
- Saves keep each segment's starting skill ratings, notes, and eligibility flags.

These protections cover RocketForge's own writes.
If you edit the metadata files by hand, do it while no trainer or ForgeGUI window is running,
and then click **Refresh Files**.

## Cache files

The catalog keeps its index in a `.rocketforge` folder inside the agent's `Models` folder.
Those files are a disposable local cache:
git ignores them, and you can delete them at any time.
ForgeGUI rebuilds them on its next scan, without changing any model data or benchmark history.
