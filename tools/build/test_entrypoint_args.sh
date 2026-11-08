#!/usr/bin/env bash
# Checks how entrypoint.sh derives the seed/export agent from the pod start command.
# Sources entrypoint.sh for its functions only (its main is guarded), so nothing is launched or synced.
#
# Usage: bash tools/build/test_entrypoint_args.sh
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

# shellcheck source=../../entrypoint.sh
source "$REPO_ROOT/entrypoint.sh"

failures=0

# $1 = label, $2 = expected agent, remaining = the start command's arguments.
expect_agent() {
    local label="$1" expected="$2" actual
    shift 2
    actual="$(resolve_agent_arg "$@")"
    if [[ "$actual" == "$expected" ]]; then
        echo "PASS  $label"
    else
        echo "FAIL  $label: expected '$expected', got '$actual'"
        failures=$((failures + 1))
    fi
}

expect_agent "no arguments" ""
expect_agent "--agent example-agent" "example-agent" --agent example-agent
expect_agent "train --agent example-agent --no-metrics" "example-agent" train --agent example-agent --no-metrics
expect_agent "--no-metrics" "" --no-metrics

if [[ $failures -ne 0 ]]; then
    echo "$failures case(s) failed."
    exit 1
fi

echo "All cases passed."
