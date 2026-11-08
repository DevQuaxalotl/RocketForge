#!/usr/bin/env bash
# Configure + build ForgeConsoleTrainer and the unit tests on Linux, run the tests, and optionally run a short GPU smoke-train.
# Linux sibling to build_and_test.ps1. Run from any directory — locates the repo root automatically.
#
# Requirements: gcc-11+, cmake, ninja-build, CUDA toolkit 12.9, libtorch 2.13.0+cu129 at /opt/libtorch,
#               python3-dev, and (for a metrics smoke-train) `pip install wandb` + `wandb login`.
#               RocketSim v3 also needs Rust (tools/setup/install_rust.sh) and its submodule; without them only v2 is built.
#
# Usage:
#   ./tools/build/build_and_test.sh                 # configure + build, then run the tests
#   ./tools/build/build_and_test.sh --skip-config   # build only (faster when no new .cpp added)
#   ./tools/build/build_and_test.sh --no-tests      # build the trainer only
#   ./tools/build/build_and_test.sh --smoke         # also run a ~5-min GPU smoke-train (no metrics) at the end
set -euo pipefail

SKIP_CONFIG=0
RUN_TESTS=1
SMOKE=0
for arg in "$@"; do
  case "$arg" in
    --skip-config) SKIP_CONFIG=1 ;;
    --no-tests)    RUN_TESTS=0 ;;
    --smoke)       SMOKE=1 ;;
    *) echo "Unknown argument: $arg" >&2; exit 2 ;;
  esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"
cd "$REPO_ROOT"

BUILD_DIR="$REPO_ROOT/out/build/forge-console-linux"

if [ "$SKIP_CONFIG" -eq 0 ]; then
  echo "==> Configuring forge-console-linux preset..."
  if [ "$RUN_TESTS" -eq 1 ]; then
    cmake --preset forge-console-linux -DROCKETFORGE_BUILD_TESTS=ON
  else
    cmake --preset forge-console-linux
  fi
fi

ENGINES_FILE="$BUILD_DIR/engines.txt"
if [ -f "$ENGINES_FILE" ]; then
  head -n 1 "$ENGINES_FILE"
fi

TARGETS=(ForgeConsoleTrainer)
if [ "$RUN_TESTS" -eq 1 ]; then
  TARGETS+=(ForgeTrainingTests)
fi
echo "==> Building ${TARGETS[*]}..."
cmake --build --preset build-console-linux --target "${TARGETS[@]}"

TRAINER="$BUILD_DIR/bin/ForgeConsoleTrainer"
if [ ! -x "$TRAINER" ]; then
  echo "Trainer executable not found: $TRAINER" >&2
  exit 1
fi
echo "==> Build succeeded: $TRAINER"

if [ "$RUN_TESTS" -eq 1 ]; then
  TESTS="$BUILD_DIR/bin/ForgeTrainingTests"
  if [ ! -x "$TESTS" ]; then
    echo "Test executable not found: $TESTS (configure without --skip-config to turn the tests on)" >&2
    exit 1
  fi
  echo "==> Running ForgeTrainingTests..."
  "$TESTS"
fi

if [ "$SMOKE" -eq 1 ]; then
  echo "==> Running smoke-train (train --agent example-agent --stage example-ball-chase --new --no-metrics)..."
  echo "    Let it run a few minutes, then stop with Ctrl+C and confirm a checkpoint was written."
  "$TRAINER" train --agent example-agent --stage example-ball-chase --new --no-metrics
fi
