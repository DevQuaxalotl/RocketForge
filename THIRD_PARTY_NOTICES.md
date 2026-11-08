# Third-party notices

RocketForge includes, fetches, links against, or is built with the third-party components listed below.
Each component remains under its own license.
RocketForge's own license, the MIT License in [LICENSE](LICENSE), covers only RocketForge's own code
and does not replace or override any license on this page.

The page groups components by how they reach you:

1. [Non-commercial components](#non-commercial-components): read this first.
2. [Source vendored in this repository](#source-vendored-in-this-repository)
3. [Git submodules](#git-submodules)
4. [Fetched at build time](#fetched-at-build-time)
5. [Installed separately and linked or bundled into builds](#installed-separately-and-linked-or-bundled-into-builds)
6. [Components whose license needs confirmation](#components-whose-license-needs-confirmation)

Where a component's license requires its full text to accompany copies,
the text is either bundled beside the component (the path is given)
or, for a component that this repository doesn't itself contain, linked at its upstream source.
If you distribute **binaries** built from RocketForge,
you also need to ship the license texts for everything those binaries contain;
the sections below note where that applies.

## Non-commercial components

> [!WARNING]
> **Necto** and **Nexto** are licensed under
> [Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0)](https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
> They may **not** be used for commercial purposes,
> and anything adapted from them must be shared under the same license.

| | |
|---|---|
| Location | `Bots/PublicBots/Necto/`, `Bots/PublicBots/Nexto/`, and the observation/action ports in `Bots/PublicBots/src/` |
| Upstream | [RLBot/RLBotPack @ `0033a98d`, `RLBotPack/Necto/`](https://github.com/RLBot/RLBotPack/tree/0033a98d4e060670f334f60f7f8dc5ee21ba93bb/RLBotPack/Necto) |
| Authors | Rolv, Soren, and several contributors (as credited in the upstream `bot.cfg` files) |
| License | CC BY-NC-SA 4.0 |
| License text | Notice in [`Bots/PublicBots/LICENSES/Necto-Nexto-CC-BY-NC-SA-4.0.txt`](Bots/PublicBots/LICENSES/Necto-Nexto-CC-BY-NC-SA-4.0.txt); full legal code at the link above and in the upstream `RLBotPack/Necto/LICENSE` |
| Changes | The TorchScript policies were converted to GPU-portable form (the unmodified originals are kept beside them), and the published Python observation/action code was ported to C++. |

What this means in practice:

- **Use and redistribution are allowed for non-commercial purposes only,**
  with attribution to the original authors and a link to the license.
- **Adaptations stay CC BY-NC-SA 4.0.**
  The converted models and the C++ ports in `Bots/PublicBots/src/` are adaptations,
  so they carry the same license, even though the rest of RocketForge is MIT-licensed.
- **This restriction applies on its own terms,**
  independently of RocketForge's own license in [LICENSE](LICENSE),
  which sets the terms for the rest of RocketForge.

## Source vendored in this repository

These components are copied into the repository tree.
Their license files travel with them at the paths given.

### RocketSim (v2, C++)

| | |
|---|---|
| Location | `ForgeTraining/RLGym/RocketSim/` |
| Upstream | <https://github.com/ZealanL/RocketSim> |
| License | MIT |
| Copyright | Copyright (c) 2022 ZealanL |
| License text | [`ForgeTraining/RLGym/RocketSim/LICENSE`](ForgeTraining/RLGym/RocketSim/LICENSE) |

### Bullet Physics 3.24 (subset, inside RocketSim v2)

| | |
|---|---|
| Location | `ForgeTraining/RLGym/RocketSim/libsrc/bullet3-3.24/` (`BulletCollision`, `BulletDynamics`, `LinearMath` only) |
| Upstream | <https://github.com/bulletphysics/bullet3> |
| License | zlib |
| Copyright | Copyright (c) 2003-2009 Erwin Coumans and contributors (per the source file headers) |
| License text | [`ForgeTraining/RLGym/RocketSim/libsrc/bullet3-3.24/LICENSE.txt`](ForgeTraining/RLGym/RocketSim/libsrc/bullet3-3.24/LICENSE.txt) |

The zlib license requires altered source versions to be plainly marked as altered.
Bullet's `Extras/` and `examples/ThirdPartyLibs/` folders, which carry other licenses, are not included.

### nlohmann/json 3.11.3

| | |
|---|---|
| Location | `ForgeTraining/GigaLearn/libsrc/json/` |
| Upstream | <https://github.com/nlohmann/json> |
| License | MIT |
| Copyright | Copyright (c) 2013-2022 Niels Lohmann |
| License text | [`ForgeTraining/GigaLearn/libsrc/json/LICENSE.MIT`](ForgeTraining/GigaLearn/libsrc/json/LICENSE.MIT) |

### pybind11 2.12.0.dev1

| | |
|---|---|
| Location | `ForgeTraining/GigaLearn/pybind11/` |
| Upstream | <https://github.com/pybind/pybind11> |
| License | BSD 3-Clause |
| Copyright | Copyright (c) 2016 Wenzel Jakob |
| License text | [`ForgeTraining/GigaLearn/pybind11/LICENSE`](ForgeTraining/GigaLearn/pybind11/LICENSE) |

pybind11's own test suite can fetch Eigen, Catch, and Boost,
but RocketForge's build doesn't build those tests.

### thread-pool

| | |
|---|---|
| Location | `ForgeTraining/RLGym/thread_pool/` |
| Upstream | <https://github.com/DeveloperPaul123/thread-pool> |
| License | MIT |
| Copyright | Copyright (c) 2021-2023 Paul Tsouchlos |
| License text | [`ForgeTraining/RLGym/thread_pool/LICENSE`](ForgeTraining/RLGym/thread_pool/LICENSE) |

### Arena collision meshes (from RLGym)

| | |
|---|---|
| Location | `ForgeTraining/RLGym/Sim/collision_meshes/` |
| Upstream | [RLGym/rlgym @ `2fc8d43`, `rlgym/rocket_league/sim/collision_meshes`](https://github.com/RLGym/rlgym/tree/2fc8d435663454b7fe45d2556777efd65cba8c2d/rlgym/rocket_league/sim/collision_meshes) |
| License | Apache License 2.0 |
| Copyright | Copyright 2021 Lucas Emery |
| License text | [`ForgeTraining/RLGym/Sim/collision_meshes/LICENSE`](ForgeTraining/RLGym/Sim/collision_meshes/LICENSE) |

The files are copied unchanged.

### Element (from RLBotPack)

| | |
|---|---|
| Location | `Bots/PublicBots/Element/` and the Element observation/action ports in `Bots/PublicBots/src/` |
| Upstream | [RLBot/RLBotPack @ `0033a98d`, `RLBotPack/Element/`](https://github.com/RLBot/RLBotPack/tree/0033a98d4e060670f334f60f7f8dc5ee21ba93bb/RLBotPack/Element) |
| Author | Rangler (as credited in the upstream `bot.cfg`) |
| License | MIT (the RLBotPack repository license; the `Element/` folder has no license of its own) |
| Copyright | Copyright (c) 2019 RLBot |
| License text | [`Bots/PublicBots/LICENSES/RLBotPack-MIT.txt`](Bots/PublicBots/LICENSES/RLBotPack-MIT.txt) |

The Python pickle was converted to TorchScript, and the observation/action code was ported to C++.

### GigaLearnCPP and RLGymCPP

| | |
|---|---|
| Location | `ForgeTraining/GigaLearn/` (excluding the vendored libraries above) and `ForgeTraining/RLGym/src/` |
| Upstream | <https://github.com/ZealanL/GigaLearnCPP-Leak> |
| Author | ZealanL |
| License | No license file upstream; used with the author's permission (see below) |

RocketForge's training framework is built on and heavily modified from this code.
The upstream repository has no `LICENSE` file and names no license.
The author, ZealanL, gave RocketForge permission to use it, by Discord direct message on 18 October 2026.
That permission was given to RocketForge;
it isn't a license, and RocketForge's MIT License doesn't extend to ZealanL's original code.

### RocketSimVis

| | |
|---|---|
| Location | `ForgeTraining/ForgeDebugger/RocketSimVis/` |
| Upstream | <https://github.com/ZealanL/RocketSimVis> |
| Author | ZealanL |
| License | **Not stated**; see [Components whose license needs confirmation](#components-whose-license-needs-confirmation) |

The viewer is modified for RocketForge integration.
It also contains:

- `src/qt_style_sheet.css`: Copyright 2013 Emanuel Claesson, Apache License 2.0
  (the license header is retained in the file).
- `data/*.obj`, `data/*.mtl`, and `data/*.png`: car, ball, boost pad, and arena models and textures;
  their provenance is listed under [needs confirmation](#components-whose-license-needs-confirmation).

## Git submodules

Submodules aren't stored in this repository,
but `git clone --recursive` fetches them, and their sources are compiled into RocketForge.
Each submodule carries its own `LICENSE` file at its root.

### RocketSim v3 (Rust)

| | |
|---|---|
| Location | `ForgeTraining/RLGym/RocketSimV3/upstream` (branch `v3-rust`, pinned commit `ea2efa0`, crate version 0.2.12) |
| Upstream | <https://github.com/ZealanL/RocketSim> |
| License | MIT |
| Copyright | Copyright (c) 2025 VirxEC |
| License text | `LICENSE` at the submodule root |

RocketForge's own C ABI wrapper, `ForgeTraining/RLGym/RocketSimV3/bindings/`,
statically links the `rocketsim` crate and the crates below into `rocketsim_ffi`,
so binaries built with the v3 backend contain all of them.
Versions are pinned in `ForgeTraining/RLGym/RocketSimV3/bindings/Cargo.lock`.
Where a crate offers a choice of licenses, RocketForge uses it under MIT.

| Crate | Version | License | Copyright |
|---|---|---|---|
| [glam](https://github.com/bitshifter/glam-rs) | 0.33.5 | MIT OR Apache-2.0 | Copyright 2020 Cameron Hart |
| [rustc-hash](https://github.com/rust-lang/rustc-hash) | 2.1.3 | Apache-2.0 OR MIT | The Rust Project Developers |
| [arrayvec](https://github.com/bluss/arrayvec) | 0.7.8 | MIT OR Apache-2.0 | Copyright (c) Ulrik Sverdrup "bluss" 2015-2023 |
| [byteorder](https://github.com/BurntSushi/byteorder) | 1.5.0 | Unlicense OR MIT | Copyright (c) 2015 Andrew Gallant |
| [fastrand](https://github.com/smol-rs/fastrand) | 2.5.0 | Apache-2.0 OR MIT | Stjepan Glavina |
| [fern](https://github.com/daboross/fern) | 0.7.1 | MIT | Copyright (c) 2014-2017 David Ross |
| [log](https://github.com/rust-lang/log) | 0.4.34 | MIT OR Apache-2.0 | Copyright (c) 2014 The Rust Project Developers |

[cbindgen](https://github.com/mozilla/cbindgen) 0.29 (MPL-2.0) runs at build time to generate the C header.
It isn't linked into any binary, and the checked-in header `bindings/include/rocketsim_ffi.h` is generated output.
The remaining entries in `Cargo.lock` are cbindgen's own build-time dependencies.

### RLBot cpp-interface

| | |
|---|---|
| Location | `RLBot/inc/cpp-interface` (pinned commit `8a7faf9`; RocketForge applies `RLBot/patches/cpp-interface.patch` to a build-tree copy) |
| Upstream | <https://github.com/RLBot/cpp-interface> |
| License | MIT |
| Copyright | Copyright (c) 2024 Michael Theall |
| License text | `LICENSE` at the submodule root |

## Fetched at build time

CMake downloads these components during configure.
They aren't stored in this repository.

| Component | Used by | Upstream | License | Copyright |
|---|---|---|---|---|
| GoogleTest 1.17.0 | Test executables only | <https://github.com/google/googletest> | BSD 3-Clause | Copyright 2008, Google Inc. |
| FlatBuffers v25.2.10 | RLBot cpp-interface (runtime library and `flatc`) | <https://github.com/google/flatbuffers> | Apache License 2.0 | Copyright 2014 Google Inc. |
| RLBot flatbuffers-schema (commit `70bcb31`) | RLBot cpp-interface (schema and prebuilt `flatc` binaries) | <https://github.com/RLBot/flatbuffers-schema> | **Not stated**; see [needs confirmation](#components-whose-license-needs-confirmation) | — |
| Tracy v0.12.0 | RLBot cpp-interface, only with `RLBOT_CPP_ENABLE_TRACY` (off by default) | <https://github.com/wolfpld/tracy> | BSD 3-Clause | See upstream `LICENSE` |
| liburing 2.9 | RLBot cpp-interface, Linux builds only | <https://github.com/axboe/liburing> | Dual MIT / LGPL-2.1 (see upstream `COPYING*`) | See upstream |

## Installed separately and linked or bundled into builds

These components are not in the repository.
You install them yourself (the scripts in `tools/setup/` download them from their official sources),
and RocketForge's build links against them.
Some build steps copy their runtime libraries next to RocketForge's executables;
if you distribute those build outputs, you are redistributing these components and must follow their terms.

### LibTorch (PyTorch C++ distribution) 2.13.0

| | |
|---|---|
| Upstream | <https://pytorch.org>, <https://github.com/pytorch/pytorch> |
| License | BSD 3-Clause-style (PyTorch license) |
| Copyright | Copyright (c) 2016- Facebook, Inc. and the PyTorch contributors; see upstream `LICENSE` and `NOTICE` |
| License text | [PyTorch `LICENSE`](https://github.com/pytorch/pytorch/blob/main/LICENSE), [PyTorch `NOTICE`](https://github.com/pytorch/pytorch/blob/main/NOTICE) |
| Redistributed by builds | Yes: the build copies every DLL from LibTorch's `lib/` folder next to the executables. |

The CUDA builds of LibTorch ("shared-with-deps") bundle NVIDIA libraries,
including the CUDA runtime (`cudart`), cuBLAS, and cuDNN.
Those libraries are licensed under NVIDIA's own terms
([CUDA Toolkit EULA](https://docs.nvidia.com/cuda/eula/), [cuDNN Software License Agreement](https://docs.nvidia.com/deeplearning/cudnn/latest/reference/eula.html)),
which allow redistribution of the listed runtime files with an application, under conditions.
The CPU-only LibTorch build contains no NVIDIA libraries.
The Docker image builds on `nvidia/cuda:12.9.1-cudnn-devel-ubuntu22.04`,
which is governed by the same NVIDIA terms.

### Qt 6 (Widgets, SQL)

| | |
|---|---|
| Used by | `ForgeGUI` and some tests |
| Upstream | <https://www.qt.io>, <https://code.qt.io> |
| License | GNU LGPL v3 (open-source edition), used through dynamic linking |
| Copyright | Copyright (C) The Qt Company Ltd. and other contributors |
| License text | <https://www.gnu.org/licenses/lgpl-3.0.html> |
| Redistributed by builds | Yes: the build runs `windeployqt`, which copies the Qt DLLs, plugins, and translations next to `ForgeGUI.exe`. |

RocketForge links Qt dynamically and doesn't modify it.
If you distribute a build that includes the Qt DLLs, LGPLv3 requires you to:

- state that the program uses Qt under the LGPLv3, and include the LGPLv3 and GPLv3 texts;
- provide, or offer, the Qt source code for the version you ship;
- let users replace the Qt libraries with their own compatible build (shipping Qt as separate DLLs satisfies this);
- not prohibit reverse engineering that is needed to debug such modifications.

The Qt SQLite driver plugin (`qsqlite`) embeds [SQLite](https://www.sqlite.org/copyright.html), which is in the public domain.

### Python 3.12 (embedded interpreter)

| | |
|---|---|
| Used by | GigaLearn embeds the interpreter through pybind11 to send training metrics to Weights & Biases. |
| Upstream | <https://www.python.org> |
| License | Python Software Foundation License Version 2 |
| Copyright | Copyright (c) 2001 Python Software Foundation; All Rights Reserved |
| License text | <https://docs.python.org/3/license.html> |
| Redistributed by builds | Yes: on Windows, the build copies the Python runtime DLLs next to the executables. |

Python packages are installed with `pip` and are not redistributed by RocketForge:
`wandb` (MIT) for metrics,
`vastai` in the Docker image for remote training,
and the RocketSimVis viewer's `requirements.txt` (`moderngl`, `moderngl-window`, `PyOpenGL`, `pyrr`, `numpy`, `pywavefront`, `pyqt5`).
Each is under its own license.
**PyQt5** in particular is licensed under the GPL v3 (or a commercial license from Riverbank Computing).

### Build tools

CMake, Ninja, MSVC or another C++ compiler, the CUDA Toolkit, and the Rust toolchain (`rustup`, `cargo`)
are needed only to build RocketForge.
They are not redistributed, and their licenses don't apply to RocketForge's source or output beyond their normal runtime-library terms.

## Components whose license needs confirmation

The components below are in the repository or used by the build,
but no license file was found for them, upstream or bundled.
Without a license, the default is that the author reserves all rights,
so their inclusion in a public release should be confirmed with the author or replaced.

| Component | Where | What was found |
|---|---|---|
| RocketSimVis | `ForgeTraining/ForgeDebugger/RocketSimVis/` | The upstream repository ([ZealanL/RocketSimVis](https://github.com/ZealanL/RocketSimVis)) has no `LICENSE` file and its README states no terms. |
| RocketSimVis models and textures | `ForgeTraining/ForgeDebugger/RocketSimVis/data/` | `Octane.obj`, `Ball.obj`, the boost pad and arena meshes, and their textures. Their origin isn't recorded; the car, ball, and arena shapes resemble *Rocket League* game assets, whose rights belong to Psyonix / Epic Games. |
| RLBot flatbuffers-schema | Fetched by the RLBot cpp-interface build | The upstream repository ([RLBot/flatbuffers-schema](https://github.com/RLBot/flatbuffers-schema)) has no `LICENSE` file. Code generated from its schema is compiled into RLBot builds, and its bundled `flatc` binaries are FlatBuffers (Apache 2.0). |

The arena collision meshes in `ForgeTraining/RLGym/Sim/collision_meshes/` are distributed by RLGym under the Apache License 2.0,
but they are geometry extracted from *Rocket League*;
RLGym's license covers RLGym's distribution of them, not any rights Psyonix / Epic Games may hold in the underlying game data.
