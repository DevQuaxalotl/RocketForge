#pragma once

#include <GigaLearnInfer/AgentSetupSnapshot.h>

#include <filesystem>
#include <memory>
#include <optional>
#include <string>

class Agent; // global namespace; see GigaLearnInfer/Agent.h

namespace GGL {
	struct InferUnit;
}
namespace RLGC {
	class ObsBuilder;
	class ActionParser;
}
namespace RocketForgeTraining {
	struct StaticOpponentSelectionEntry;
}

namespace RocketForge
{
	/**
	 * Identifies one model to load for inference. Prefer the factory functions below so the
	 * snapshot-path convention stays in one place.
	 */
	struct InferenceModelSource
	{
		std::string agentName;

		/** Numbered checkpoint folder holding POLICY.lt / INFERENCE_POLICY.json / RUNNING_STATS.json. */
		std::filesystem::path checkpointFolder;

		/** training_setup.json to apply to the fresh agent, or empty to keep the current setup. */
		std::filesystem::path setupSnapshotPath;
	};

	/**
	 * Segment checkpoint source: derives the snapshot path as <segment>/training_setup.json
	 * (checkpointFolder is <segment>/checkpoints/<timesteps>), matching RunRewardAnalysis.
	 */
	InferenceModelSource FromSegmentCheckpoint(std::string agentName, std::filesystem::path checkpointFolder);

	/**
	 * Imported-model source: resolves the model folder under importedRoot and leaves the
	 * snapshot path empty (imported/scripted policies define their network via
	 * INFERENCE_POLICY.json through the agent's MakeInferenceModels).
	 */
	InferenceModelSource FromImportedModel(std::string agentName, std::filesystem::path importedRoot);

	/**
	 * Static-opponent source: dispatches on the entry's checkpoint layout — a segment
	 * checkpoint (.../checkpoints/<timesteps>) derives its training_setup snapshot, an
	 * imported model folder does not. Uses the entry's own source agent, which may differ
	 * from the agent being debugged.
	 */
	InferenceModelSource FromStaticOpponentEntry(const RocketForgeTraining::StaticOpponentSelectionEntry& entry);

	/** Result of ResolveInferenceAgent: a ready-to-use agent plus what was applied. */
	struct ResolvedInferenceAgent
	{
		/** Created, snapshot applied, modelPath set. */
		std::unique_ptr<Agent> agent;

		std::filesystem::path checkpointFolder;

		/** Set iff a snapshot was loaded and applied. */
		std::optional<AgentSetupSnapshot> appliedSnapshot;
	};

	/**
	 * Creates the named agent, applies saved setup, and sets its modelPath.
	 * An absent historical snapshot retains current network settings but resolves simulator/prediction to v2.
	 * Unknown agents and malformed existing snapshots throw instead of changing setup silently.
	 */
	ResolvedInferenceAgent ResolveInferenceAgent(const InferenceModelSource& source);

	/**
	 * Convenience: builds an InferUnit through the agent's own MakeInferenceModels ctor so
	 * imported/scripted inference policies load. obsBuilder/actionParser must have been made
	 * from resolved.agent and outlive the returned InferUnit.
	 */
	std::unique_ptr<GGL::InferUnit> MakeInferUnit(
		const ResolvedInferenceAgent& resolved,
		RLGC::ObsBuilder* obsBuilder,
		RLGC::ActionParser* actionParser,
		bool useGPU);

	/** Outcome of loading a model folder into an agent's inference networks, without playing it. */
	struct ModelFitCheck
	{
		bool fits = false;

		/** Why the model doesn't load into the agent; empty when it fits. */
		std::string problem;
	};

	/**
	 * Builds the agent's current inference networks on the CPU and loads the folder's weights into them.
	 * Reports a mismatch (input size, action count, layer sizes, missing files) instead of throwing.
	 */
	ModelFitCheck CheckModelFitsAgent(const std::string& agentName, const std::filesystem::path& modelFolder);

	/**
	 * Whether inference should run on the GPU: true only when requested and CUDA is available.
	 * A refused request prints a notice to stderr, so machine-read stdout stays clean.
	 */
	bool ResolveInferenceUseGPU(bool requested = true);
}
