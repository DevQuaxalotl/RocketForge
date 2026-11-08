# Training without an NVIDIA GPU

An NVIDIA GPU is not required.
RocketForge builds, trains and plays on the CPU, more slowly.
This page shows you how to set up a CPU-only build, what it can't do, and how much slower it trains.

## Set up a CPU-only build

On a machine without an NVIDIA GPU, skip the CUDA step in [Getting started](../GettingStarted.md);
`install_libtorch.ps1` then installs the CPU build of libtorch,
and the build and tests need nothing else.
To build for the CPU on a machine that does have an NVIDIA GPU, pass `-Variant Cpu` to `install_libtorch.ps1`.

## What works on the CPU

Everything works on the CPU: training (console and GUI), the tests, debug rendering, reward analysis,
model benchmarks, RLBot and CPU submission exports.
Only the CUDA features are missing: CUDA graph inference, GPU submission packages,
and the tests for those, which report as skipped.

To run an RLBot export on the CPU, see [Choose the inference device](RLBot.md#choose-the-inference-device).

## How much slower it trains

Training on the CPU runs about 7 times slower, mostly because the PPO update runs on the CPU too.
On one machine (24-thread i9-12900K, RTX 3080), 150 seconds of `example-ball-chase` from a new policy measured:

| Build | Average steps per second | Steps in 150 s |
|---|---|---|
| CUDA (RTX 3080) | about 216,000 | 30.7 million |
| CPU only | about 31,000 | 4.5 million |

Expect a larger policy or batch to widen the gap, since the PPO update grows with both.
