# RocketForge — ForgeConsoleTrainer training image
#
# Base: CUDA 12.9 + cuDNN 9 on Ubuntu 22.04. CUDA >= 12.8 is required to target
#       Blackwell (sm_120); 12.9 matches the libtorch cu129 build below so RTX
#       50-series cloud instances are usable.
#
# Build:
#   git submodule update --init ForgeTraining/RLGym/RocketSimV3/upstream
#   docker build -t rocketforge .
#
# .dockerignore drops every .git, so the RocketSim v3 submodule can't be fetched inside the build;
# it has to be checked out on the host first, or the configure step fails.
#
# Run (training):
#   docker run --gpus all \
#     -e WANDB_API_KEY=<your_key> \
#     -v /host/path/to/Bots:/workspace/RocketForge/Bots \
#     rocketforge \
#     train --agent example-agent --stage example-ball-chase
#
# The Bots/*/Models volume mount persists checkpoints across container restarts.
# Bots/ source code is baked into the image; only the Models/ subdirs are mounted.
# WANDB_API_KEY is read by wandb at runtime — never bake credentials into the image.
#
# On clean exit (run completion or `docker stop`), the entrypoint exports all
# Bots/**/Models trees to Cloudflare R2 when R2_BUCKET + RCLONE_CONFIG_R2_* env
# vars are set (see Documentation/RemoteTraining/RemoteTraining.md). Without them, export is skipped.
#
# Run (no metrics / quick test):
#   docker run --gpus all rocketforge train --agent example-agent --stage example-ball-chase --no-metrics

FROM nvidia/cuda:12.9.1-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

# python_scripts/ (metric_receiver, render_receiver) lives in source at this path.
# Adding its parent to PYTHONPATH makes "import python_scripts.*" work at runtime
# without needing the cmake build directory (which lives only in the BuildKit cache).
ENV PYTHONPATH=/workspace/RocketForge/ForgeTraining/GigaLearn

# ── System packages ───────────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        ninja-build \
        python3-dev \
        python3-pip \
        wget \
        ca-certificates \
        gpg \
        unzip \
        git \
        openssh-server \
    && rm -rf /var/lib/apt/lists/*

# Entrypoint mode (see Documentation/RemoteTraining/VastAI.md) replaces the host's
# own SSH-setup script, so sshd must be configured here instead. Key-only login —
# entrypoint.sh writes $PUBLIC_KEY (auto-injected by Vast.ai/RunPod) to
# authorized_keys and starts sshd; without that env var sshd is never started.
RUN mkdir -p /var/run/sshd \
    && sed -i 's/^#\?PermitRootLogin.*/PermitRootLogin yes/' /etc/ssh/sshd_config \
    && sed -i 's/^#\?PasswordAuthentication.*/PasswordAuthentication no/' /etc/ssh/sshd_config

# Vast.ai reads EXPOSE metadata at instance-creation time to decide which ports to
# direct-map on the host; without this, sshd runs fine inside the container but
# Vast.ai never allocates a host port for it (confirmed: ssh_host/ssh_port stayed
# null even with sshd verifiably running, until this was added).
EXPOSE 22

# ── CMake ≥ 3.23 from Kitware ─────────────────────────────────────────────────
# Ubuntu 22.04 ships cmake 3.22 which is too old for CMakePresets.json version 6.
RUN wget -qO - https://apt.kitware.com/keys/kitware-archive-latest.asc \
        | gpg --dearmor -o /usr/share/keyrings/kitware-archive-keyring.gpg \
    && echo 'deb [signed-by=/usr/share/keyrings/kitware-archive-keyring.gpg] https://apt.kitware.com/ubuntu/ jammy main' \
        > /etc/apt/sources.list.d/kitware.list \
    && apt-get update \
    && apt-get install -y --no-install-recommends cmake \
    && rm -rf /var/lib/apt/lists/*

# ── libtorch 2.13.0+cu129 ─────────────────────────────────────────────────────
# Version-matched to the dev machine so the c10 API used by GigaLearn resolves
# identically. libtorch-shared-with-deps is now the unified (cxx11) ABI build,
# required for linking against system GCC on Linux. The cu129 build carries the
# Blackwell (sm_120) kernels while retaining Ampere/Ada/Hopper, so one image runs
# on both current and RTX 50-series cloud GPUs.
RUN wget -qO /tmp/libtorch.zip \
        "https://download.pytorch.org/libtorch/cu129/libtorch-shared-with-deps-2.13.0%2Bcu129.zip" \
    && unzip -q /tmp/libtorch.zip -d /opt \
    && rm /tmp/libtorch.zip

# Register libtorch with the dynamic linker so its .so files are discoverable
# at runtime. Separate step so the 2 GB download layer above stays cached.
RUN echo "/opt/libtorch/lib" > /etc/ld.so.conf.d/libtorch.conf \
    && ldconfig

# ── libtorch's external CUDA dependencies ─────────────────────────────────────
# Recent libtorch declares DT_NEEDED on cuSPARSELt, NCCL, and NVSHMEM but no
# longer bundles them, and the CUDA base image does not ship them. Supplied via
# the cu12 NVIDIA wheels (the exact ABI libtorch is built against); their lib
# dirs go on the loader path so the linker resolves the NEEDED refs at build and
# ld.so finds them at runtime. Missing them fails the final link with undefined
# cusparseLt*/nccl*/nvshmem* references.
RUN pip3 install --no-cache-dir nvidia-cusparselt-cu12 nvidia-nccl-cu12 nvidia-nvshmem-cu12 \
    && printf '%s\n' \
        /usr/local/lib/python3.10/dist-packages/nvidia/cusparselt/lib \
        /usr/local/lib/python3.10/dist-packages/nvidia/nccl/lib \
        /usr/local/lib/python3.10/dist-packages/nvidia/nvshmem/lib \
        > /etc/ld.so.conf.d/nvidia-torch-deps.conf \
    && ldconfig

# ── Python packages ───────────────────────────────────────────────────────────
# wandb: training metrics. vastai: Vast.ai CLI used by entrypoint.sh to
# self-terminate the instance after a verified R2 export (inert off Vast.ai).
RUN pip3 install --no-cache-dir wandb vastai

# ── rclone (checkpoint export to S3-compatible storage, e.g. Cloudflare R2) ────
# Static binary from the official downloads; placed before the source COPY so
# this layer stays cached when only source files change.
RUN wget -qO /tmp/rclone.zip "https://downloads.rclone.org/rclone-current-linux-amd64.zip" \
    && unzip -q /tmp/rclone.zip -d /tmp \
    && cp /tmp/rclone-*-linux-amd64/rclone /usr/local/bin/rclone \
    && chmod 755 /usr/local/bin/rclone \
    && rm -rf /tmp/rclone.zip /tmp/rclone-*-linux-amd64

# ── runpodctl (RunPod pod self-termination after a verified R2 export) ─────────
# Static binary from the official release. Only INVOKED by entrypoint.sh when
# RUNPOD_POD_ID is set, so its presence is inert on non-RunPod hosts. The Vast.ai
# equivalent is the `vastai` pip package installed above.
RUN wget -qO /usr/local/bin/runpodctl \
        "https://github.com/runpod/runpodctl/releases/latest/download/runpodctl-linux-amd64" \
    && chmod 755 /usr/local/bin/runpodctl

# ── Rust (RocketSim v3) ───────────────────────────────────────────────────────
# The default toolchain matches the pin in ForgeTraining/RLGym/RocketSimV3/bindings/rust-toolchain.toml;
# if the two drift apart, every build downloads the pinned toolchain again.
RUN wget -qO /tmp/rustup.sh --https-only https://sh.rustup.rs \
    && sh /tmp/rustup.sh -y --profile minimal --default-toolchain 1.99.0 \
    && rm /tmp/rustup.sh
ENV PATH=/root/.cargo/bin:$PATH

WORKDIR /workspace/RocketForge

# Fetch the v3 crates in a layer keyed only on the lockfile and the submodule, so a source-only change skips it.
# The stub lib.rs satisfies cargo's manifest check; COPY . . brings the real sources.
# The registry is a cache mount shared with the build step below, which would otherwise hide what this layer fetched.
# A missing submodule skips the fetch so the configure step reports it with its fix.
COPY ForgeTraining/RLGym/RocketSimV3/bindings/Cargo.toml \
     ForgeTraining/RLGym/RocketSimV3/bindings/Cargo.lock \
     ForgeTraining/RLGym/RocketSimV3/bindings/rust-toolchain.toml \
     ForgeTraining/RLGym/RocketSimV3/bindings/
COPY ForgeTraining/RLGym/RocketSimV3/upstream/ ForgeTraining/RLGym/RocketSimV3/upstream/
RUN --mount=type=cache,target=/root/.cargo/registry \
    cd ForgeTraining/RLGym/RocketSimV3/bindings \
    && if [ -f ../upstream/rocketsim/Cargo.toml ]; then \
           mkdir -p src && touch src/lib.rs && cargo fetch --locked && rm -r src; \
       fi

# ── Source & build ────────────────────────────────────────────────────────────

# Copy the full source tree.
COPY . .

# --mount=type=cache persists the build directory (object files, CMake state)
# across docker build runs via BuildKit. On a source-only change, ninja
# recompiles only what changed instead of the full tree — much faster iteration.
# The cache is local to this machine and does not affect the final image size.
# The cargo target dir for RocketSim v3 lives under out/, so the Rust build is cached the same way.
# ROCKETFORGE_ENABLE_V3=ON makes a missing v3 (submodule or Rust) fail the build instead of shipping v2 only.
RUN --mount=type=cache,target=/workspace/RocketForge/out \
    --mount=type=cache,target=/root/.cargo/registry \
    cmake --preset forge-console-linux -DROCKETFORGE_ENABLE_V3=ON \
    && cmake --build --preset build-console-linux --target ForgeConsoleTrainer -- -j4 \
    && cp out/build/forge-console-linux/bin/ForgeConsoleTrainer /usr/local/bin/ForgeConsoleTrainer \
    && find out/build/forge-console-linux -maxdepth 1 -name "lib*.so*" -exec cp {} /usr/local/lib/ \; \
    && ldconfig

# ── Convenience stop command ──────────────────────────────────────────────────
# `stop-training` in the pod's web terminal sends SIGTERM to PID 1 (the entrypoint),
# triggering the graceful save → R2 export → self-terminate path. Easier to recall
# than `kill -TERM 1`. `kill -USR1 1` (or `docker kill -s USR1`) skips to the next queued stage.
RUN printf '#!/usr/bin/env bash\nexec kill -TERM 1\n' > /usr/local/bin/stop-training \
    && chmod 755 /usr/local/bin/stop-training

# ── Runtime entrypoint ────────────────────────────────────────────────────────
# entrypoint.sh runs the trainer with signal forwarding and exports Models/ to R2
# on exit. The trainer binary remains on PATH at /usr/local/bin/ForgeConsoleTrainer.
COPY entrypoint.sh /usr/local/bin/entrypoint.sh
# Strip any CR so a CRLF build context (e.g. a Windows checkout) can't break the
# shebang at runtime: /usr/bin/env: 'bash\r': No such file or directory.
RUN sed -i 's/\r$//' /usr/local/bin/entrypoint.sh \
    && chmod +x /usr/local/bin/entrypoint.sh
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
