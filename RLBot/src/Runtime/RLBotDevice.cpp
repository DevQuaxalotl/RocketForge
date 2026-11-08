#include "RLBotDevice.h"

#include <stdexcept>
#if defined(RG_CUDA_SUPPORT) || (defined(RLBOT_HAS_CUDA) && RLBOT_HAS_CUDA)
#include <torch/torch.h>
#include <torch/cuda.h>
#endif

// Startup check follows https://github.com/SubparN0va/GGLBot/blob/bob-gpu/src/RLBotMain.cpp
void ValidateRLBotDevice(bool useGPU)
{
    if (!useGPU)
        return;

#if defined(RG_CUDA_SUPPORT) || (defined(RLBOT_HAS_CUDA) && RLBOT_HAS_CUDA)
    if (!torch::cuda::is_available())
        throw std::runtime_error("CUDA is unavailable; check the NVIDIA GPU, driver, and CUDA LibTorch runtime. Select cpu to use CPU inference.");

    // Synchronize a real kernel so unsupported GPU architectures fail before match startup.
    torch::NoGradGuard noGrad;
    torch::Tensor sample = torch::ones({2, 2}, torch::TensorOptions().device(torch::kCUDA));
    torch::Tensor result = torch::matmul(sample, sample).cpu();
#else
    throw std::runtime_error("This executable was built with CPU-only LibTorch. Rebuild with CUDA LibTorch for GPU inference, or select cpu.");
#endif
}
