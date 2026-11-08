#pragma once

#include <RLGymCPP/ActionParsers/ActionParser.h>

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

struct RLBotPolicyInferenceResult {
    int actionId = -1;
    double inferenceMicros = -1.0;
    std::vector<float> observation;
    std::vector<std::uint8_t> actionMask;
    std::vector<float> actionProbabilities;
};

/** Values captured from one bot during an RLBot callback. **/
struct RLBotAnalysisSample {
    unsigned index;
    const RLGC::Player& player;
    const RLGC::GameState& gameState;
    RLGC::ActionParser& actionParser;
    const RLGC::Action& selectedAction;
    const RLGC::Action& appliedControls;
    const RLGC::Action& prevAction;
    const std::vector<RLGC::ActionMacro>& activeActionMacros;
    int activeMacroIndex;
    int activeMacroTicksLeft;
    int tickSkip;
    int actionDelay;
    int ticksElapsed;
    int episode;
    int policyTick;
    float secondsElapsed;
    float gameTimeRemaining;
    int matchPhase;
    float deltaTime;
    double callbackWorkMicros;
    double runtimeStartupMicros;
    bool useGPU;
    bool inferAction;
    bool activatedPendingAction;
    std::uint32_t frameNum;
    const RLGC::Action& rlbotLastInput;
    int rawAirState;
    float rawDodgeTimeout;
    float rawDodgeElapsed;
    float rawDodgeDirX;
    float rawDodgeDirY;
    float rawDemoTimeout;
    int trainingEngineVersion;
};

/**
 * Development-only RLBot testing recorder.
 * Normal bot execution can ignore this class unless the testing flag is enabled.
 **/
class RLBotAnalysis {
public:
    RLBotAnalysis(unsigned team, const std::unordered_set<unsigned>& indices);

    bool IsEnabled() const;
    bool CapturePolicyDiagnostics() const;
    void Configure();
    void ResetBot(unsigned index);
    void RecordInference(
        unsigned index,
        const std::vector<RLGC::ActionMacro>& actionMacros,
        RLBotPolicyInferenceResult&& inference);
    void ActivatePendingAction(unsigned index);
    void WriteSample(const RLBotAnalysisSample& sample);
    void FinishCallback(double callbackMicros);
    void AdvancePolicyStep(unsigned index);

private:
    struct BotState {
        std::vector<RLGC::ActionMacro> selectedActionMacros;
        std::vector<uint8_t> selectedActionMask;
        std::vector<float> selectedObs;
        std::vector<float> selectedActionProbs;
        int selectedActionId = -1;
        int pendingActionId = -1;
        int activeActionId = -1;
        int policyStep = 0;
        double lastInferenceMicros = -1.0;
        bool lastOutputJump = false;
    };

    unsigned team_;
    std::unordered_set<unsigned> indices_;
    std::unordered_map<unsigned, BotState> botStates_;
    bool configured_ = false;
    bool enabled_ = false;
    bool capturePolicyDiagnostics_ = true;
    bool captureStateSnapshots_ = true;
    int rows_ = 0;
    int maxRows_ = -1;
    int flushRows_ = 1;
    double previousCallbackMicros_ = -1.0;
    std::filesystem::path tracePath_;
    std::ofstream traceCsv_;
};
