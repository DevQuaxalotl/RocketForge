#!/usr/bin/env bash
# Installs the Rust toolchain RocketForge needs to build RocketSim v3. Nothing is installed without a "yes".
#
#   ./tools/setup/install_rust.sh          # asks before installing
#   ./tools/setup/install_rust.sh --yes    # no prompt, for scripted use
#
# When Rust is already installed this only adds the pinned toolchain.
set -euo pipefail

TOOLCHAIN="1.99.0"

YES=0
for arg in "$@"; do
  case "$arg" in
    --yes|-y) YES=1 ;;
    *) echo "Unknown argument: $arg" >&2; exit 2 ;;
  esac
done

if command -v cargo >/dev/null 2>&1; then
  echo "Rust is already installed; adding toolchain $TOOLCHAIN if it is missing."
  rustup toolchain install "$TOOLCHAIN" --profile minimal
  exit 0
fi

echo "This will install:"
echo "  - rustup, from https://sh.rustup.rs"
echo "  - Rust $TOOLCHAIN (minimal profile)"
echo "  into $HOME/.rustup and $HOME/.cargo (about 300 MB)."

if [ "$YES" -eq 0 ]; then
  if [ ! -t 0 ]; then
    echo "Nothing was installed: there is no terminal to confirm the install. Re-run with --yes to install without asking." >&2
    exit 1
  fi
  read -r -p "Install Rust now? [y/N] " answer
  case "$answer" in
    y|Y|yes|YES) ;;
    *) echo "Cancelled. Nothing was installed."; exit 1 ;;
  esac
fi

curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal --default-toolchain "$TOOLCHAIN"

# rustup updates shell profiles, not this process's PATH.
"$HOME/.cargo/bin/cargo" --version

echo "Rust installed. Rebuild RocketForge (tools/build/build_and_test.sh) to enable RocketSim v3."
