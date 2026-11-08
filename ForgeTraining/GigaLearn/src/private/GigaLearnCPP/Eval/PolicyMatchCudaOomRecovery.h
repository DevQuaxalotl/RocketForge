#pragma once

#include <c10/util/Exception.h>

#ifdef RG_CUDA_SUPPORT
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime_api.h>
#endif

namespace GGL::PolicyMatchRunnerInternal {

	/** State shared by every inference batch in one policy-match runner invocation. **/
	struct CudaOomRecoveryState {
		bool used = false;
		bool graphsDisabled = false;
	};

	/** Returns whether a LibTorch exception specifically reports CUDA allocation failure. **/
	inline bool IsCudaOutOfMemory(const c10::Error& error) {
#ifdef RG_CUDA_SUPPORT
		if (dynamic_cast<const c10::OutOfMemoryError*>(&error) != nullptr)
			return true;
		const auto* acceleratorError = dynamic_cast<const c10::AcceleratorError*>(&error);
		return acceleratorError != nullptr
			&& acceleratorError->get_error_code() == (int32_t)cudaErrorMemoryAllocation;
#else
		(void)error;
		return false;
#endif
	}

	/** Runs one inference attempt and permits one CUDA-OOM retry in eager mode. **/
	template <typename Attempt, typename IsOom, typename Cleanup>
	bool RunWithOneCudaOomRetry(
		bool isCudaDevice,
		bool requestCudaGraph,
		CudaOomRecoveryState& state,
		Attempt&& attempt,
		IsOom&& isOom,
		Cleanup&& cleanup) {
		try {
			attempt(requestCudaGraph && !state.graphsDisabled);
			return false;
		} catch (const c10::Error& error) {
			if (!isCudaDevice || state.used || !isOom(error))
				throw;

			state.used = true;
			state.graphsDisabled = true;
			cleanup();
			attempt(false);
			return true;
		}
	}
}
