# RocketForge code standards

All code in RocketForge follows these standards.
Read this page before you write code for the project.

## Naming conventions

### Classes and structs

Use PascalCase for all class and struct names.

```cpp
class Reward { ... };
struct GameState { ... };
class DefaultObsPadded : public DefaultObs { ... };
```

### Methods and functions

Use PascalCase for all methods and free functions.

```cpp
virtual float GetReward(const Player& player, const GameState& state, bool isFinal);
virtual void Reset(const GameState& initialState);
int GetActionAmount();
```

### Variables

Use camelCase for instance variables and local variables.

```cpp
bool goalScored;
std::string agentName;
int maxTeamSize;
```

Prefer descriptive variable names over abbreviations.
Single-letter or heavily abbreviated names are acceptable only for advanced math,
where the notation is conventional and is commented at first use.

```cpp
// Good
float normalizedWeight = teamResult.weight / totalWeight;
int playerIndex = 0;
std::vector<float> stepRewards = reward->GetAllRewards(state, isFinal);

// Avoid
float w = teamResult.weight / totalWeight;
int pi = 0;
auto vals = reward->GetAllRewards(state, isFinal);
```

For a mathematical variable, add a comment that defines the notation where the variable is first used:

```cpp
// G₀ = discounted return from episode start: G₀ = Σ(γ^t · r_t) for t = 0..T-1.
std::vector<float> sumG0(numRewards, 0.f);
```

Give private and cached members a leading underscore.

```cpp
std::string _cachedName;
Arena* _lastArena;
```

**Exception for Qt classes:** classes that derive from `QObject`, and most other Qt types,
can use the `m_` prefix instead.
The prefix matches Qt's own conventions, the existing `ForgeGUI/` code,
and the `m_thread`, `m_worker`, and `m_widget` patterns throughout the Qt documentation.
Within a single class, pick one prefix and use it consistently.

```cpp
class TrainingWorker : public QObject {
    Q_OBJECT
    QThread* m_thread = nullptr;
    AnalysisWorker* m_worker = nullptr;
};
```

### Constants

Use UPPER_SNAKE_CASE for constants inside namespaces.

```cpp
namespace CommonValues {
    constexpr float SIDE_WALL_X = 4096;
    constexpr float BACK_WALL_Y = 5120;
    constexpr int BOOST_LOCATIONS_AMOUNT = 34;
}
```

### Agent and stage registry names

Use kebab-case string literals for the agent and stage names that you register with the macro system.

```cpp
REGISTER_AGENT("example-agent", ExampleTrainingAgent)
REGISTER_STAGE("example-ball-chase", ExampleTrainingStage_BallChase, exampleCompatInfo)
```

## Namespaces

| Namespace | Purpose |
|-----------|---------|
| `RLGC` | Game environment abstractions, such as `GameState`, `Reward`, and `ObsBuilder`. |
| `GGL` / `GigaLearn` | PPO training framework, such as `Learner`, `InferUnit`, and `ModelSet`. |
| `RocketForge` | Orchestration layer, such as `TrainingStage`, the CLI, and the registry. |

Don't use `using namespace` in header files.

In `.cpp` files, prefer `using namespace` directives at the top of the file for commonly used namespaces,
rather than repeating the prefix on every symbol.
A namespace prefix exists to disambiguate, or to show where a symbol comes from;
when it does neither, the prefix is clutter.

```cpp
// MyTrainer.cpp

using namespace std;
using namespace RLGC;
using namespace RocketForge;

// Good — the prefixes don't add information at this point.
vector<WeightedReward> rewards = stage->GetRewards();
unique_ptr<ObsBuilder> obsBuilder(stage->agent->MakeObsBuilder());

// Avoid — std:: and RLGC:: are noise here.
std::vector<RLGC::WeightedReward> rewards = stage->GetRewards();
std::unique_ptr<RLGC::ObsBuilder> obsBuilder(stage->agent->MakeObsBuilder());
```

Keep an explicit prefix when it helps a reader.
Most often, that's when the symbol comes from a less common namespace,
or when two libraries share a name and the reader benefits from knowing which one is in use.

```cpp
// Good — the prefix tells the reader at a glance which library this comes from.
nlohmann::json j = result;
RocketSim::Arena* arena = RocketSim::Arena::Create(stage->agent->gameMode);
GGL::InferUnit inferUnit(...);
```

Also keep the prefix on symbols whose unqualified name is too generic for a reader to tell what they do at a glance,
such as `move`, `min`, `max`, `abs`, `sqrt`, and `swap`.
`std::move(x)` is unambiguous,
but a bare `move(x)` could mean several things,
and it can even resolve silently to a member function from a Qt base class, such as `QWidget::move(QPoint)`.

```cpp
// Good — std:: makes intent obvious and avoids hidden member-function shadowing.
m_result = std::move(result);
const float clamped = std::min(value, ZERO_GUARD);

// Avoid — what `move` is this? Reader has to scan upward to find the using directive
// and check for collisions with member functions.
m_result = move(result);
const float clamped = min(value, ZERO_GUARD);
```

A function definition for a symbol declared in another namespace must still be qualified at the definition site,
regardless of any `using namespace` directives:

```cpp
// Good
void RocketForge::to_json(nlohmann::json& j, const RewardImpactEntry& entry) { ... }
```

## File and folder organization

Name each header file exactly after its primary class.

```
GameState.h      →  class GameState
Reward.h         →  class Reward
ActionParser.h   →  class ActionParser
```

Group files by responsibility, not by type:

```
RLGymCPP/
├── ActionParsers/
├── Gamestates/
├── ObsBuilders/
├── Rewards/
├── StateSetters/
└── TerminalConditions/
```

When a feature spans three or more files, put them together in a named subfolder under `src/`.
Before you create a subfolder, check whether an existing subfolder is already a natural home for the files.
Name folders after the feature, not the file type.
Two-file pairs, one `.h` and one `.cpp`, stay at the `src/` root.

```
RocketForge/ForgeTraining/src/
├── Settings/           ← ForgeSettings + validation + JSON serialization
├── Storage/            ← BuildStorage + ImportedModelStorage
├── StaticOpponents/    ← all StaticOpponentSelection* files
├── StageQueue/         ← StageQueueRunner + StageQueueSignals
├── RewardAnalysis/     ← RewardAnalysis + RewardAnalysisJson
└── TrainingSetup/      ← snapshot, comparison, version management
```

For a library that exposes headers to other libraries,
put the public headers in a `public/` subdirectory and the implementation files in `private/`.

Use `#pragma once` in all headers, not `#ifndef` guards.

## Memory management

Use `std::unique_ptr` for ownership, and transfer ownership with `std::move()`.

```cpp
std::unique_ptr<Agent> Clone() const override {
    return std::make_unique<ExampleTrainingAgent>(*this);
}

std::unique_ptr<Agent> agent = RocketForgeTraining::CreateAgent(agentName);
if (!agent) return 1;
SomeFunction(std::move(agent));
```

Use raw pointers only for non-owning references.

```cpp
Arena* lastArena = nullptr;  // not owned; set externally
```

Use `std::vector` for collections of dynamic or unknown size.
Raw arrays are acceptable for collections whose size is a compile-time constant.

```cpp
std::vector<Player> players;         // dynamic
std::vector<float> rewards;          // dynamic
float boostLocations[BOOST_AMOUNT];  // known constant size — raw array is fine
```

## Function parameters

Split a function's parameters across multiple lines only when it has more than four parameters,
or when the line exceeds 120 characters.
Keep fewer than four parameters on a single line.

```cpp
// Good — 2 params, single line
void Reset(Arena* arena, const GameState& state);

// Good - max for single line
RocketForge::TrainingStage* CreateTrainingStage(const std::string& agentName, const std::string& stageName, const GlobalSettings& globals, const StageSettings& stage);

// Good — 4+ params, split across lines
TeamSizeImpactResult RunTeamSizeAnalysis(
    TrainingStage* stage,
    GGL::InferUnit& inferUnit,
    int teamSizeA,
    int teamSizeB,
    int numTimesteps,
    float gaeGamma,
    const std::function<void(int, int)>& progressCallback);
```

Avoid literal values in function call arguments when the function name alone doesn't reveal their meaning,
especially booleans.
Use a named variable or constant instead.

```cpp
// Avoid — what does 'true' mean here?
GGL::InferUnit inferUnit(obsBuilder, obsSize, parser, head, policy, path, true);

// Good — intent is clear
const bool useGPU = true;
GGL::InferUnit inferUnit(obsBuilder, obsSize, parser, head, policy, path, useGPU);
```

For a function with more than four parameters and several default values,
consider a config struct to simplify the call sites.

## Type declarations with `auto`

Don't use `auto` for a variable declaration unless the type is long or hard to read.
When the type is short and obvious, spell it out:
it aids readability without adding meaningful length.

```cpp
// Good — type is short and clear
std::vector<float> stepRewards = reward->GetAllRewards(state, isFinal);
RLGC::WeightedReward& weightedReward = weightedRewards[rewardIndex];
const RLGC::GameState& currentState  = envSet.state.gameStates[0];

// Good — auto is appropriate when the type is verbose
auto obsBuilder = std::unique_ptr<RLGC::ObsBuilder>(stage->agent->MakeObsBuilder());
const std::vector<RocketForge::WeightedTeamSize>& teamSizes = stage->GetWeightedTeamSizes();

// Avoid — type is short, auto hides it
auto vals = reward->GetAllRewards(state, isFinal);
auto& wr  = weightedRewards[i];
```

## Error handling and logging

Use the provided macros instead of writing custom error paths.

```cpp
// General logging
RG_LOG("Training started for agent: " << agentName);

// Fatal error — logs and throws
RG_ERR_CLOSE("Arena boost pad count mismatch: " << actual << "/" << expected);

// Internal invariant — disabled in RG_UNSAFE builds
RG_ASSERT(size[1] == newRow.size());

// Conditional debug logging tied to a reward's debug flag
RG_REWARD_LOG(rewardPtr, "Ball touched this step");
```

Use `RG_ERR_CLOSE` only for states that are truly unrecoverable.
For internal checks, prefer `RG_ASSERT`.

## Comments and documentation

Comments matter, but be **selective**.
A comment earns its place when it tells the reader something that the code can't quickly tell them itself:
a non-obvious *why*, a hidden invariant, or some external context.
Comments also cost attention.
Each one pulls the eye and adds to the work of scanning a block,
and too many bury the few that carry weight.
Aim for fewer, sharper comments: not none, and not one per line.

Follow these rules:

- **Explain the non-obvious, and only that.**
  When logic needs context, supply the missing context,
  and don't also restate what the code already makes plain.
  Pairing the obvious with the non-obvious hides the part worth reading.
- **Let names do the work first.**
  A descriptive variable or function name often removes the need for a comment.
  Add a comment when the name can't carry the meaning,
  for example when an important variable is buried among many others and a short label helps the reader find it.
- **Keep it brief.**
  State the non-obvious thing in as few words as it takes.
  If one statement needs several lines of explanation, check whether all of it is needed.
- **Break lines where the sentence breaks.**
  In a multi-line comment, end each line at a sentence or clause boundary rather than at a fixed width.
  For more information, see [Line breaks in multi-line text](#line-breaks-in-multi-line-text).
- **Spend big comment blocks carefully.**
  A multi-line block in the middle of a function draws strong attention.
  Add one only when the insight justifies the interruption,
  not for every small detail that the code doesn't spell out.
- **Write for the code's context, not for the change that you're making.**
  A comment describes the code as it's read for the rest of its life,
  not the task, bug, feature, or ticket that prompted your edit.
  In otherwise generic code, a comment that names the specific scenario, caller, or stage that you were working on
  injects narrow specifics into general-purpose code.
  It reads as a note about the diff, not about the code, and it quickly becomes stale and misleading.
  Match the comment's altitude to the surrounding code,
  and refer to a specific scenario only when the code itself is specific to it.

```cpp
// Avoid — narrates the obvious; the loop already reads this way
// Iterate the pads and copy each active flag into the observation
for (int i = 0; i < padCount; i++)
    obs[i] = pads[i].isActive;

// Good — no comment; nothing here is unclear
for (int i = 0; i < padCount; i++)
    obs[i] = pads[i].isActive;

// Good — one line of the non-obvious *why*, nothing more
// RLBot orders pads differently from ours, so remap before indexing.
```

```cpp
// A generic clamp in a shared reward base class.

// Avoid — the comment is about the author's task, not this generic code.
// Clamp so the late-stage aerial reward in my bot doesn't spike when the ball is on the wall.
value = std::clamp(value, -maxReward, maxReward);

// Good — describes what this code does at its own altitude; no comment needed if the name says it.
value = std::clamp(value, -maxReward, maxReward);
```

### Line breaks in multi-line text

When a comment spans several lines, break it at **meaning boundaries**, not at a fixed column.
Filling every line to the same width is what a word-wrap algorithm does:
it routinely cuts a sentence one word before a comma,
or splits a phrase that the eye expects to read as a unit.
That costs the reader more than an even right edge is worth.

Break at the end of a sentence,
or at a clause boundary that the punctuation already marks:
after a comma, colon, or dash, or before a conjunction that starts a new clause.
Keep short parenthetical or bracketed groups whole on one line.
A line that ends up noticeably shorter than the one above it is fine.
That raggedness is the point, because each line is then a self-contained thought.

```cpp
// Avoid — wrapped to a target width; the second line ends one word before its comma
// Poll for pending work (or a stop request) for up to idle_spin_. Returns
// true when either arrived while spinning, false on timeout or when
// spinning is disabled (idle_spin_ == 0).

// Good — one line per clause
// Poll for pending work (or a stop request) for up to idle_spin_.
// Returns true when either arrived while spinning,
// false on timeout or when spinning is disabled (idle_spin_ == 0).

// Also good — one line per sentence
// Poll for pending work (or a stop request) for up to idle_spin_.
// Returns true when either arrived while spinning, false on timeout or when spinning is disabled (idle_spin_ == 0).
```

This rule applies to all multi-line prose in the project, not only to code comments:
`/** */` blocks, commit messages, and Markdown documentation follow it too.
Lines still mustn't run extremely long.
When a single clause is so wide that the reader has to scan back to find where the next one starts,
split it at the next natural boundary inside the clause.

### Header files

Comments that explain a class or field are encouraged, but keep them concise.
Use `/** */` block comments in header files.
Always put a block comment **above** what it describes,
never inline after a declaration on the same line.

If the comment fits on one line, keep it on one line.
If it's too long, use the multi-line `/** * ... */` format;
don't wrap text inside single-line `/** */` delimiters.

```cpp
// Avoid — text wraps inside single-line delimiters
/** Constructs a temporary stage instance to read its weightedTeamSizes without starting training.
    Returns an empty vector if the stage is not registered. */

// Good — short enough for one line
/** Constructs a temporary stage to read weightedTeamSizes. Returns empty if the stage is unregistered. */

// Good — too long for one line, use multi-line format instead
/**
 * Constructs a temporary stage instance to read its weightedTeamSizes without starting training.
 * Returns an empty vector if the stage is not registered.
 */
```

Use single-line block comments for brief method and field annotations:

```cpp
/** Returns all rewards for a given game step. */
virtual std::vector<float> GetAllRewards(const GameState& state, bool isFinal);
```

Use multi-line block comments for longer descriptions, such as classes, complex methods, and architecture notes:

```cpp
/**
 * Base class for all observation builders.
 * Subclasses implement BuildObs() to define neural network inputs.
 */
class ObsBuilder { ... };
```

For a declaration comment, put the comment on the line above the field,
and always leave an empty line above the comment:

```cpp
/** mean(|G0_per_player|): gross magnitude — non-zero for ZeroSum rewards. */
float meanAbsDiscountedReturn = 0.f;

// Avoid — inline placement
float meanAbsDiscountedReturn = 0.f; /** mean(|G0_per_player|) */
```

To group related members in a large header, use section separators:

```cpp
/**************** Reward Weights ****************/
float rw_touchBall;
float rw_velocityToBall;

/**************** State Setter Weights ****************/
float sw_standard;
float sw_random;
```

Use `// NOTE:` for non-obvious invariants and warnings:

```cpp
// NOTE: Could be null if no arena has been stepped yet
Arena* lastArena = nullptr;
```

### Source files

Use `//` line comments in `.cpp` files.
Don't comment self-evident code.

In a long or complex function, `//` section headers are acceptable to orient the reader:

```cpp
float ComputeReward(...) {
    // Align velocity toward ball
    float velAlignment = ...;

    // Penalize aerial inactivity
    float airPenalty = ...;

    return velAlignment + airPenalty;
}
```

When you port a class or function from a Python original, add a URL reference to the original:

```cpp
// https://github.com/AechPro/rocket-league-gym-sim/blob/main/rlgym_sim/utils/gamestates/game_state.py
struct GameState { ... };
```

## Logic

### Readability

Give code room to breathe.
Dense, wall-of-text functions are hard to scan,
so use blank lines liberally to separate distinct steps,
and a reader can pick out the structure at a glance.
This applies everywhere, including GUI setup code such as widget construction, signal wiring, and layout assembly,
not only to algorithm-heavy functions.

Insert a blank line in the following places:

- **Between separate pieces of logic** within a function.
  Each step, whether a computation, a decision, or an action, gets visual separation from the next.
- **Above every comment** that introduces a new block.
  Never let a comment touch the previous statement.
- **Between a setup group and the logic that consumes it.**
  For example, after a run of variable declarations, before the loop or branch that uses them.
- **Around multi-line control-flow blocks** (`if`, `for`, `while`, and `try`) that sit among other statements:
  one blank line before the block and one after.
- **Before a concluding `return`** that caps off a computation.

Group closely related one-liners together, with no blank line between them,
so that the spacing shows which statements belong to the same step.
Don't add extra double blank lines; one blank line is the unit of separation.

When you construct GUI widgets, separate the logical groups:
create the widgets, configure them, add them to the layout, and connect the signals.
A blank line between the groups makes the intent obvious.

```cpp
// Bad — everything squished; reader can't tell the groups apart
auto* form = new QFormLayout();
m_agentCombo = new AgentComboBox(this);
m_buildCombo = new BuildComboBox(this);
form->addRow("Agent:", m_agentCombo);
form->addRow("Build:", m_buildCombo);
m_startBtn = new QPushButton("Start", this);
connect(m_agentCombo, &AgentComboBox::AgentChanged, this, &Foo::OnAgentChanged);
connect(m_startBtn, &QPushButton::clicked, this, &Foo::OnStartClicked);

// Good — blank lines separate construction, layout, and wiring
auto* form = new QFormLayout();
m_agentCombo = new AgentComboBox(this);
m_buildCombo = new BuildComboBox(this);
form->addRow("Agent:", m_agentCombo);
form->addRow("Build:", m_buildCombo);

m_startBtn = new QPushButton("Start", this);

connect(m_agentCombo, &AgentComboBox::AgentChanged, this, &Foo::OnAgentChanged);
connect(m_startBtn,   &QPushButton::clicked,        this, &Foo::OnStartClicked);
```

The same applies to ordinary logic:

```cpp
// StageQueueRunner.cpp

// Bad - Everything is squished together and difficult to read
std::optional<GGL::SegmentVersionNumber> explicitVersion = entry.versionOverride;
// A user-supplied versionOverride wins over the allocated segment id. Only fall
// back to reusing the segment id when no override is requested.
if (!action.startNewReuseSegmentId.empty() && !explicitVersion.has_value()) {
	// Reuse the allocated segment id rather than allocating a new one.
	GGL::SegmentVersionNumber vn{};
	std::sscanf(action.startNewReuseSegmentId.c_str(), "v%d.%d", &vn.major, &vn.minor);
	explicitVersion = vn;
}
GGL::SegmentVersionNumber assigned = AllocateTrainingSegment(
	stage.get(), entry.versionIncrement, explicitVersion, entry.versionNotes);
entry.createdSegmentId = GGL::MakeSegmentVersionId(assigned.major, assigned.minor);

// Good - There is spacing above comments and between logic making it easier to read
std::optional<GGL::SegmentVersionNumber> explicitVersion = entry.versionOverride;

// A user-supplied versionOverride wins over the allocated segment id. Only fall
// back to reusing the segment id when no override is requested.
if (!action.startNewReuseSegmentId.empty() && !explicitVersion.has_value()) {
	// Reuse the allocated segment id rather than allocating a new one.
	GGL::SegmentVersionNumber vn{};
	std::sscanf(action.startNewReuseSegmentId.c_str(), "v%d.%d", &vn.major, &vn.minor);
	explicitVersion = vn;
}

GGL::SegmentVersionNumber assigned = AllocateTrainingSegment(stage.get(), entry.versionIncrement, explicitVersion, entry.versionNotes);

entry.createdSegmentId = GGL::MakeSegmentVersionId(assigned.major, assigned.minor);
```

### Line length

Avoid splitting a line of code across multiple lines when you can.
The maximum line length is 140 characters;
a line that fits within that limit stays on one line.

```cpp
// Avoid this - This can just be one line
const bool userExplicitlyChoseNew =
	(i == 0) && entry.bootstrap.mode == BootstrapConfig::Mode::New;

// This is fine - It is clearly too long to be on one line
const bool userPickedDifferentExistingVersion = (i == 0) &&	
	entry.bootstrap.mode == BootstrapConfig::Mode::ContinueExisting &&
	!entry.bootstrap.selectedExistingVersionId.empty() &&
	entry.bootstrap.selectedExistingVersionId != entry.createdSegmentId;
```

## Tests

Tests use GoogleTest.
Name each test `TEST(<Unit>, <Behavior>)`,
where `<Unit>` is the class or function under test, in PascalCase without underscores:

```cpp
TEST(GAE, PropagatesWithinEpisodesAndStopsAtNormalTerminals) { ... }
```

Include `TestSupport.h` for the shared helpers, and follow these rules:

- **Skip a test only for a missing environment**, such as a gitignored asset or absent hardware,
  or for an opt-in run gated by an environment variable, such as a benchmark.
  Never skip to avoid a failure.
  Use `ROCKETFORGE_REQUIRE_ROCKETSIM()`, `ROCKETFORGE_REQUIRE_COLLISION_MESHES()`, or `ROCKETFORGE_REQUIRE_CUDA()`
  as the first statement of the `TEST` body.
  They're macros because `GTEST_SKIP()` returns from the function that it's written in.
  A test needs `ROCKETFORGE_REQUIRE_ROCKETSIM()` whenever it builds an arena,
  including indirectly, for example through an `EnvSet`, `Agent::GetObsSize()`, or a policy match.
- **Never write to a fixed path under the system temp directory.**
  Use `RocketForgeTests::TestTempDir(label)`, which is unique per test and per process,
  so parallel CTest runs can't collide.
- For new floating-point comparisons, prefer `ASSERT_NEAR(actual, expected, tolerance)`.
- An `ASSERT_*` inside a `void` helper stops the whole test; for an example, see `InstallTestEnvironment`.
  A helper that returns a value can't contain `ASSERT_*`;
  return the value and assert in the test.
- Several cases of one function read best as one table-driven test:
  a local `Case` struct, a `cases[]` array, and a loop.
  Give each case a `label`, or reuse a field that already names it,
  and call `SCOPED_TRACE(testCase.label)` so that a failure names its row.

Add each new test file to `ForgeTrainingTests/CMakeLists.txt` explicitly; the file list doesn't use GLOB.

## Formatting with `.clang-format`

The repository-root `.clang-format` file encodes the layout rules on this page:
tabs, 140 columns, attached braces, and one parameter per line when parameters are split.
Editors use it to format new code to match these standards.
It requires clang-format 16 or later; the copy bundled with Visual Studio 2022 qualifies.

**Format only code that you're writing or changing.**
Use "format selection" or `git clang-format`, never "format document" on an existing file.
Much of the existing code predates the config,
so formatting a whole file produces a large unrelated diff
and strips hand-written alignment that the standards allow.

Keep editor formatting manual:

- Leave format-on-save off.
- In Visual Studio, consider turning off format-on-`}` and format-on-paste,
  in **Tools > Options > Text Editor > C/C++ > Code Style > Formatting**,
  because closing an existing block can reformat all of it.

The formatter never aligns consecutive assignments or declarations.
Where alignment helps readability, align by hand,
and wrap the block in `// clang-format off` and `// clang-format on`.

`ForgeGUI/` is mostly indented with spaces, while the config uses tabs,
so check the indentation when you format GUI code.

## Macros and registry

Register agents and stages above their constructor definitions, in the `.cpp` file.

```cpp
// ExampleAgent.cpp

REGISTER_AGENT("example-agent", ExampleTrainingAgent)

ExampleTrainingAgent::ExampleTrainingAgent() {
    agentName = "example-agent";
    maxTeamSize = 1;
    sharedHead.layerSizes = { 256, 256 };
    policy.layerSizes    = { 256, 256, 256 };
    critic.layerSizes    = { 256, 256, 256 };
}
```

```cpp
// ExampleTrainingPlan.cpp

static const RocketForge::StageCompatInfo exampleCompatInfo{};
REGISTER_STAGE("example-ball-chase", ExampleTrainingStage_BallChase, exampleCompatInfo)

ExampleTrainingStage_BallChase::ExampleTrainingStage_BallChase() {
    ...
}
```

## CMake conventions

Every module's `CMakeLists.txt` file follows this structure:

```cmake
cmake_minimum_required(VERSION 3.10)
project("ModuleName")

file(GLOB_RECURSE FILES_SRC "src/*.cpp" "src/*.h")
add_library(ModuleName OBJECT ${FILES_SRC})

set_target_properties(ModuleName PROPERTIES LINKER_LANGUAGE CXX)
set_target_properties(ModuleName PROPERTIES CXX_STANDARD 20)

target_link_libraries(ModuleName PUBLIC SomeDependency)
```

All targets must use C++20.
Always set `LINKER_LANGUAGE CXX` explicitly.
Use `GLOB_RECURSE` to pick up all sources under `src/`.

Use the provided CMake presets instead of configuring manually:

| Preset | Purpose |
|--------|---------|
| `forge` | Console trainer, tests, and the GUI when Qt is found (Windows). |
| `rlbot` | RLBot only. |
| `forge-console-linux` | Console trainer only (Linux and Docker). |

Pick targets through the build presets (`build-console`, `build-gui`, `build-tests`, and so on) instead of adding a configure preset:
each configure preset is a separate build tree that compiles ForgeTraining from scratch.
