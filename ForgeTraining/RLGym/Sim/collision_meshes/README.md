# Collision meshes

The arena collision meshes that both RocketSim engines load at startup, one folder per game mode.
`ForgeTraining/RLGym/Sim/CMakeLists.txt` defines this location,
and C++ code reaches it through `RocketSim::CollisionMeshesFolder()` / `CollisionMeshesRelativePath()`
(`ForgeTraining/RLGym/Sim/include/RocketSim/Arena.h`).

The files are copied unchanged from the RLGym project ([`rlgym/rocket_league/sim/collision_meshes`][rlgym-meshes], commit `2fc8d43`),
which distributes them under the Apache License 2.0; [LICENSE](LICENSE) is its copy of that license.

[rlgym-meshes]: https://github.com/RLGym/rlgym/tree/2fc8d435663454b7fe45d2556777efd65cba8c2d/rlgym/rocket_league/sim/collision_meshes
