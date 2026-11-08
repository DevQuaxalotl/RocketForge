# RocketSimVis

RocketSimVis is the 3D viewer that shows your bot playing in the simulator.
RocketForge bundles it, and starts it for you when you render a model.

It's based on [RocketSimVis](https://github.com/ZealanL/RocketSimVis) by ZealanL,
with changes for RocketForge.

## Open the viewer

You don't start RocketSimVis yourself.
It opens when you do either of the following:

- In ForgeGUI, click **Start Debug Render** on the **Debug** tab, or train with rendering on.
- On the console, run `ForgeConsoleTrainer debug`.
  For its options, see [`debug`](../../../Documentation/CLI/ConsoleTrainer.md#debug).

RocketSimVis runs on Python.
The first time it starts, it installs the packages in `requirements.txt` into the Python that RocketForge uses,
which takes a minute.

## Controls

While RocketForge renders, the focus, speed, and pause keys control the render loop itself;
bird cam only changes the viewer's camera.

| Key | Action |
|---|---|
| Left click, `→`, or `Tab` | Focus the next player. Wraps around. |
| `←` or `Shift+Tab` | Focus the previous player. Wraps around. |
| `0` to `7` | Focus that player index, if it exists. |
| `P` | Focus the player nearest the ball. |
| `B` | Toggle bird cam. |
| `=` or `-` | Step the game speed up or down. |
| `Backspace` | Reset the game speed to 1x. |
| `Space` | Toggle pause. |

## Ports

RocketSimVis uses two UDP ports on the local machine:

| Port | Direction | Carries |
|---|---|---|
| `9273` | RocketForge to the viewer | Game states, as JSON. For the format, see [Networking format](networking-format.md). |
| `9274` | The viewer to RocketForge | Focus, speed, and pause commands. |

If another process, such as a second render, already holds port 9274,
the keys that control the render loop stop working until that process exits.
The focus keys still move the viewer's own camera.

## Run the viewer on its own

To use RocketSimVis with your own program, run `RUN.bat`, or `python src/main.py` from this folder,
and send game states to port 9273 in the [networking format](networking-format.md).
Without a RocketForge render loop, the focus keys move the local camera, and speed and pause do nothing.

## For maintainers

### Rendering

Debug lines share one draw call, and keep their own colors and camera-facing width.
Trails upload and draw only their live vertices.
The receiver parses each packet before it briefly locks to publish it,
and rendering keeps its own synthetic ball rotation, so it never changes a published scene.
A malformed packet leaves the last complete scene on screen.

### Build output

CMake copies the viewer's files into the build output, and tracks each file separately (CMake 3.12 or later).
An unchanged build doesn't copy the viewer again.
Added and changed files are copied, and removed files are deleted only if the previous build's manifest owned them,
so unrelated files in the output folder stay.
Python bytecode caches aren't copied.

### Tests

With the viewer's dependencies installed, run these from the repository root:

```text
python -B -m unittest discover -s ForgeTraining/ForgeDebugger/RocketSimVis/tests -v
python -B -m unittest discover -s ForgeTraining/ForgeDebugger/tests -v
```

- The first suite checks geometry, pixel output from a real offscreen OpenGL context, draw counts,
  packet publication, and animation history.
  Without an OpenGL driver, the pixel tests skip; set `RF_REQUIRE_GL_TESTS=1` to make that a failure.
- The second suite needs CMake and Ninja (or `NINJA_EXECUTABLE`),
  and creates temporary build fixtures under the system temporary folder.

The native debugger tests are the `Collector`, `DebugRenderFocus`, `CriticProbe`, `CriticValueMath`, and `CriticHeadRows` suites.
