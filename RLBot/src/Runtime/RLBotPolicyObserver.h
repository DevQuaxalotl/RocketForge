#pragma once

#include <vector>

namespace GGL { struct InferUnit; }

/** Receives optional policy telemetry without coupling the runtime to development tooling. **/
class RLBotPolicyObserver {
public:
    virtual ~RLBotPolicyObserver() = default;

    virtual bool CapturePolicyDiagnostics() const = 0;
    virtual void OnPolicyRuntimeReady(double startupMicros) = 0;
    virtual void OnPolicyInference(
        int actionId,
        double inferenceMicros,
        const GGL::InferUnit& inferUnit) = 0;

    /** Row-major diagnostics; elapsed time covers the whole batch, not each car. **/
    virtual void OnPolicyBatchInference(
        const std::vector<unsigned>& indices,
        const std::vector<int>& actions,
        double inferenceMicros,
        const GGL::InferUnit& inferUnit) {
        // Single-car batches forward to OnPolicyInference, so per-car observers need not override this.
        if (actions.size() == 1)
            OnPolicyInference(actions.front(), inferenceMicros, inferUnit);
    }
};
