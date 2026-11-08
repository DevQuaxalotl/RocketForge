FROM ubuntu:24.04

SHELL ["/bin/bash", "-c"]
ENV DEBIAN_FRONTEND=noninteractive

# Bob builds both native Linux and Windows packages from the same source tree.
RUN apt-get update -qq && \
    apt-get install -y -qq --no-install-recommends \
        build-essential \
        ca-certificates \
        cmake \
        curl \
        git \
        gnupg \
        lsb-release \
        make \
        ninja-build \
        pkg-config \
        software-properties-common \
        wget \
        xz-utils \
    && rm -rf /var/lib/apt/lists/*

# Clang 19 can consume the current Windows CRT headers downloaded by xwin.
RUN wget -qO- https://apt.llvm.org/llvm-snapshot.gpg.key \
        | tee /etc/apt/trusted.gpg.d/apt.llvm.org.asc >/dev/null && \
    add-apt-repository -y \
        "deb https://apt.llvm.org/$(lsb_release -sc)/ llvm-toolchain-$(lsb_release -sc)-19 main" && \
    apt-get update -qq && \
    apt-get install -y -qq --no-install-recommends \
        clang-19 \
        clang-tidy-19 \
        libc++-19-dev \
        libc++abi-19-dev \
        lld-19 \
        llvm-19-dev \
    && rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/clang clang /usr/bin/clang-19 100 && \
    update-alternatives --install /usr/bin/clang++ clang++ /usr/bin/clang++-19 100 && \
    update-alternatives --install /usr/bin/lld lld /usr/bin/lld-19 100 && \
    update-alternatives --install /usr/bin/ld.lld ld.lld /usr/bin/ld.lld-19 100 && \
    update-alternatives --install /usr/bin/llvm-ar llvm-ar /usr/bin/llvm-ar-19 100 && \
    update-alternatives --install /usr/bin/llvm-ranlib llvm-ranlib /usr/bin/llvm-ranlib-19 100 && \
    update-alternatives --install /usr/bin/llvm-config llvm-config /usr/bin/llvm-config-19 100 && \
    update-alternatives --install /usr/bin/lld-link lld-link /usr/bin/lld-link-19 100 && \
    update-alternatives --install /usr/bin/llvm-dlltool llvm-dlltool /usr/bin/llvm-dlltool-19 100 && \
    update-alternatives --install /usr/bin/llvm-lib llvm-lib /usr/bin/llvm-lib-19 100

# xwin supplies the redistributable Windows CRT and SDK without requiring MSVC.
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \
        | sh -s -- -y --no-modify-path
ENV PATH="/root/.cargo/bin:$PATH"
RUN cargo install xwin --locked
RUN xwin --accept-license splat --output /tmp/xwin

# scripts/build_bob.py stages the selected Windows SDK, including an SDK fingerprint.
# A plain `bob build` (how the RLBot bot pack builds) stages nothing,
# so CPU builds fall back to the shared bot-pack archive; GPU builds still need the staged CUDA SDK.
COPY build-support/libtorch/ /tmp/libtorch-win/
COPY cmake/ReadDevice.cmake /tmp/ReadDevice.cmake
COPY rlbot/device.txt /tmp/rlbot-device.txt
ARG LIBTORCH_WINDOWS_CPU_URL=https://github.com/VirxEC/pytorch-archive/releases/download/r-1/botpack_x86_64-windows.tar.xz
RUN cmake -DDEVICE_FILE=/tmp/rlbot-device.txt -DOUTPUT_FILE=/tmp/rlbot-device \
        -P /tmp/ReadDevice.cmake && \
    device="$(cat /tmp/rlbot-device)" && \
    if [ -f /tmp/libtorch-win/local/lib/torch.lib ]; then \
        cp -a /tmp/libtorch-win/local /tmp/libtorch-sdk; \
    elif [ "$device" = cpu ]; then \
        curl -fsSL -o /tmp/botpack-windows.tar.xz "$LIBTORCH_WINDOWS_CPU_URL" && \
        mkdir -p /tmp/botpack-windows && \
        tar -xJf /tmp/botpack-windows.tar.xz -C /tmp/botpack-windows && \
        rm -f /tmp/botpack-windows.tar.xz && \
        mkdir -p /tmp/libtorch-sdk/lib && \
        cp -a /tmp/botpack-windows/torch-archive/torch/include /tmp/libtorch-sdk/ && \
        for library in torch torch_cpu c10; do \
            cp /tmp/botpack-windows/torch-archive/torch/lib/"$library".lib /tmp/libtorch-sdk/lib/; \
        done; \
    else \
        echo "error: GPU builds need a staged Windows CUDA LibTorch SDK." >&2; \
        echo "Run scripts/build_bob.py (or stage build-support/libtorch/local) and retry." >&2; \
        exit 1; \
    fi && \
    test -f /tmp/libtorch-sdk/include/torch/csrc/api/include/torch/version.h && \
    test -f /tmp/libtorch-sdk/lib/torch.lib

# CUDA builds skip CPU archives entirely. CPU builds also support Linux.
ARG LIBTORCH_LINUX_CPU_URL=https://github.com/VirxEC/pytorch-archive/releases/download/r-1/botpack_x86_64-linux.tar.xz
RUN if [ "$(cat /tmp/rlbot-device)" = cpu ]; then \
        curl -fsSL -o /tmp/pytorch-linux.tar.xz "$LIBTORCH_LINUX_CPU_URL" && \
        mkdir -p /tmp/pytorch-linux && \
        tar -xJf /tmp/pytorch-linux.tar.xz -C /tmp/pytorch-linux && \
        rm -f /tmp/pytorch-linux.tar.xz; \
    fi

WORKDIR /src
COPY . /src

# Only packages for RocketSim v3 bots carry Rust sources.
# rustup installs the toolchain they pin, plus the Windows target that the clang-cl build links against.
RUN bindings=/src/ForgeTraining/RLGym/RocketSimV3/bindings && \
    if [ -f "$bindings/rust-toolchain.toml" ]; then \
        cd "$bindings" && \
        rustup toolchain install && \
        rustup target add x86_64-pc-windows-msvc; \
    fi

# Cargo's settings for the Windows target: the clang-cl wrappers below for crates that compile C,
# and lld-link with xwin's import libraries for anything Rust links itself.
ENV CC_x86_64_pc_windows_msvc=clang-cl-c \
    CXX_x86_64_pc_windows_msvc=clang-cl \
    AR_x86_64_pc_windows_msvc=llvm-lib \
    CFLAGS_x86_64_pc_windows_msvc="/imsvc/tmp/xwin/crt/include /imsvc/tmp/xwin/sdk/include/ucrt /imsvc/tmp/xwin/sdk/include/um /imsvc/tmp/xwin/sdk/include/shared" \
    CXXFLAGS_x86_64_pc_windows_msvc="/imsvc/tmp/xwin/crt/include /imsvc/tmp/xwin/sdk/include/ucrt /imsvc/tmp/xwin/sdk/include/um /imsvc/tmp/xwin/sdk/include/shared" \
    CARGO_TARGET_X86_64_PC_WINDOWS_MSVC_LINKER=lld-link \
    CARGO_TARGET_X86_64_PC_WINDOWS_MSVC_RUSTFLAGS="-Lnative=/tmp/xwin/crt/lib/x86_64 -Lnative=/tmp/xwin/sdk/lib/um/x86_64 -Lnative=/tmp/xwin/sdk/lib/ucrt/x86_64"

# clang-cl wrappers keep CMake and LibTorch on their MSVC-compatible code paths.
RUN printf '#!/bin/bash\nexec /usr/bin/clang++ --driver-mode=cl -target x86_64-pc-windows-msvc "$@"\n' \
        > /usr/local/bin/clang-cl && \
    printf '#!/bin/bash\nexec /usr/bin/clang --driver-mode=cl -target x86_64-pc-windows-msvc "$@"\n' \
        > /usr/local/bin/clang-cl-c && \
    chmod +x /usr/local/bin/clang-cl /usr/local/bin/clang-cl-c

RUN mkdir -p /src/cmake && \
    cat > /src/cmake/toolchain-msvc.cmake <<'TOOLCHAIN_EOF'
set(CMAKE_SYSTEM_NAME Windows)
set(CMAKE_SYSTEM_VERSION 10.0)

set(CMAKE_C_COMPILER /usr/local/bin/clang-cl-c)
set(CMAKE_CXX_COMPILER /usr/local/bin/clang-cl)
set(CMAKE_C_COMPILER_TARGET x86_64-pc-windows-msvc)
set(CMAKE_CXX_COMPILER_TARGET x86_64-pc-windows-msvc)

set(CMAKE_LINKER lld-link)
set(CMAKE_AR llvm-lib)
set(CMAKE_RANLIB llvm-lib)
set(CMAKE_C_CREATE_STATIC_LIBRARY "<CMAKE_AR> /OUT:<TARGET> <OBJECTS>")
set(CMAKE_CXX_CREATE_STATIC_LIBRARY "<CMAKE_AR> /OUT:<TARGET> <OBJECTS>")

set(CMAKE_C_FLAGS_INIT
  "-MD /imsvc/tmp/xwin/crt/include /imsvc/tmp/xwin/sdk/include/ucrt /imsvc/tmp/xwin/sdk/include/um /imsvc/tmp/xwin/sdk/include/shared /clang:-Wno-unused-command-line-argument"
)
set(CMAKE_CXX_FLAGS_INIT
  "-MD -Xclang -stdlib=libc++ /imsvc/tmp/xwin/crt/include /imsvc/tmp/xwin/sdk/include/ucrt /imsvc/tmp/xwin/sdk/include/um /imsvc/tmp/xwin/sdk/include/shared /clang:-Wno-unused-command-line-argument"
)
set(CMAKE_EXE_LINKER_FLAGS_INIT
  "/LIBPATH:/tmp/xwin/crt/lib/x86_64 /LIBPATH:/tmp/xwin/sdk/lib/um/x86_64 /LIBPATH:/tmp/xwin/sdk/lib/ucrt/x86_64"
)

set(CMAKE_TRY_COMPILE_CONFIGURATION Release)
set(CMAKE_FIND_LIBRARY_SUFFIXES .lib .dll .dll.a .a)
set(CMAKE_FIND_ROOT_PATH /tmp/xwin/crt /tmp/xwin/sdk /tmp/libtorch-sdk)
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
set(CMAKE_MSVC_RUNTIME_LIBRARY MultiThreadedDLL)
TOOLCHAIN_EOF

RUN if [ "$(cat /tmp/rlbot-device)" = cpu ]; then \
        cmake -S . -B build-linux -G Ninja \
            -DCMAKE_BUILD_TYPE=Release -DRLBOT_DEVICE=cpu \
            -DLIBTORCH_ROOT=/tmp/pytorch-linux/torch-archive/torch && \
        cmake --build build-linux --target RLBot --parallel "$(nproc)" && \
        mkdir -p /out/x86_64-unknown-linux-gnu/000-runtime && \
        cp /src/build-linux/RLBot /out/x86_64-unknown-linux-gnu/RLBot && \
        cp /src/build-linux/RLBotCoreCPU /out/x86_64-unknown-linux-gnu/000-runtime/RLBotCoreCPU; \
    fi

# RLBotCPP enables one MSVC-only preprocessor flag which clang-cl does not need.
RUN sed -i '/Zc:preprocessor/d' /src/RocketForgeRLBot/inc/cpp-interface/library/CMakeLists.txt
RUN device="$(cat /tmp/rlbot-device)" && \
    if [ "$device" = gpu ]; then core=RLBotCoreCUDA; else core=RLBotCoreCPU; fi && \
    cmake -S . -B build-win -G Ninja \
        -DCMAKE_TOOLCHAIN_FILE=/src/cmake/toolchain-msvc.cmake \
        -DCMAKE_BUILD_TYPE=Release -DRLBOT_DEVICE="$device" \
        -DLIBTORCH_ROOT=/tmp/libtorch-sdk && \
    cmake --build build-win --target RLBot --parallel "$(nproc)" && \
    mkdir -p /out/x86_64-pc-windows-msvc/000-runtime && \
    cp /src/build-win/RLBot.exe /out/x86_64-pc-windows-msvc/RLBot.exe && \
    cp "/src/build-win/$core.exe" "/out/x86_64-pc-windows-msvc/000-runtime/$core.exe"

# Models and portable runtime data live beside each Torch-linked core.
RUN set -e; for platform in /out/*; do \
        runtime="$platform/000-runtime"; \
        cp /src/rlbot/POLICY.lt "$runtime/POLICY.lt"; \
        if [ -f /src/rlbot/SHARED_HEAD.lt ]; then \
            cp /src/rlbot/SHARED_HEAD.lt "$runtime/SHARED_HEAD.lt"; \
        fi; \
        if [ -f /src/rlbot/agent_setup.json ]; then \
            cp /src/rlbot/agent_setup.json "$runtime/agent_setup.json"; \
        fi; \
        cp -a /src/ForgeTraining/RLGym/Sim/collision_meshes "$runtime/collision_meshes"; \
    done

# Sorting places 000-runtime before the launcher, so Bob selects the launcher
# rather than the Torch-linked core as the platform entry point.
ENTRYPOINT ["tar", "--sort=name", "-C", "/out", "-cf", "-", "."]
