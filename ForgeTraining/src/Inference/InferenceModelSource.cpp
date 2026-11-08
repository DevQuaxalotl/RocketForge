#include "InferenceModelSource.h"

#include <GigaLearnCPP/TrainingStorage.h>
#include <GigaLearnInfer/Agent.h>
#include <GigaLearnInfer/AgentRegistry.h>
#include <GigaLearnInfer/InferUnit.h>
#include <GigaLearnInfer/Models.h>

#include "TrainingSetup/TrainingSetupSnapshot.h"
#include "StaticOpponents/StaticOpponentSelection.h"

#include <torch/cuda.h>
#include <torch/serialize/input-archive.h>

#include <iostream>

namespace
{
	// Strips the nested "RG FATAL ERROR: <context>: " layers, unescapes quoted paths,
	// and truncates long messages such as libtorch stack traces.
	std::string CleanLoadErrorText(std::string text)
	{
		const std::string fatalPrefix = "RG FATAL ERROR: ";
		const size_t innermost = text.rfind(fatalPrefix);
		if (innermost != std::string::npos)
			text = text.substr(innermost + fatalPrefix.size());

		for (size_t pos = text.find("\\\\"); pos != std::string::npos; pos = text.find("\\\\", pos + 1))
			text.erase(pos, 1);

		constexpr size_t MAX_LENGTH = 400;
		if (text.size() > MAX_LENGTH)
			text = text.substr(0, MAX_LENGTH) + "...";

		return text;
	}

	std::string FormatShape(c10::IntArrayRef sizes)
	{
		std::string text;
		for (size_t i = 0; i < sizes.size(); i++)
			text += (i ? " x " : "") + std::to_string(sizes[i]);

		return text.empty() ? "a scalar" : text;
	}

	// Reads a dotted parameter name such as "2.weight" from a torch::save archive,
	// whose child modules are nested archives.
	bool TryReadSavedParameter(torch::serialize::InputArchive& archive, const std::string& name, torch::Tensor& out)
	{
		const size_t dot = name.find('.');
		if (dot == std::string::npos)
			return archive.try_read(name, out);

		torch::serialize::InputArchive child;
		return archive.try_read(name.substr(0, dot), child) && TryReadSavedParameter(child, name.substr(dot + 1), out);
	}

	// One network parameter's name and shape, captured before a load can overwrite the parameters.
	struct ParameterShape
	{
		std::string name;
		std::vector<int64_t> sizes;
	};

	std::vector<ParameterShape> CaptureParameterShapes(GGL::Model& model)
	{
		std::vector<ParameterShape> shapes;
		for (const auto& parameter : model.seq->named_parameters())
			shapes.push_back({ parameter.key(), parameter.value().sizes().vec() });

		return shapes;
	}

	// Names the first layer whose saved shape differs from the network's, or returns empty when every layer matches.
	std::string DescribeShapeMismatch(
		const std::vector<ParameterShape>& expected,
		const std::filesystem::path& file,
		const std::string& agentName)
	{
		torch::serialize::InputArchive archive;
		archive.load_from(file.string(), torch::kCPU);

		const std::string fileName = file.filename().string();
		for (const ParameterShape& parameter : expected) {
			torch::Tensor saved;
			if (!TryReadSavedParameter(archive, parameter.name, saved))
				return fileName + " has no layer " + parameter.name + ", which " + agentName + "'s network has.";

			if (saved.sizes() != c10::IntArrayRef(parameter.sizes)) {
				return fileName + " layer " + parameter.name + " is " + FormatShape(saved.sizes()) + " in the file, but "
					+ FormatShape(parameter.sizes) + " in " + agentName + "'s network.";
			}
		}

		return {};
	}

	// Loads one network's weights and explains a failure in terms of the file and layer; returns empty on success.
	std::string DescribeModelLoadProblem(GGL::Model& model, const std::filesystem::path& modelFolder, const std::string& agentName)
	{
		// torch::load replaces the parameters before the size check fails, so capture the network's own shapes first.
		const std::vector<ParameterShape> expectedShapes = CaptureParameterShapes(model);

		try {
			const bool allowNotExist = false;
			model.Load(modelFolder, allowNotExist);
			return {};
		} catch (const std::exception& e) {
			const std::filesystem::path file = model.GetSavePath(modelFolder);
			const bool isScriptedPolicy = std::filesystem::exists(modelFolder / "INFERENCE_POLICY.json");
			if (!isScriptedPolicy && !std::filesystem::exists(file))
				return file.filename().string() + " is missing, but " + agentName + "'s network needs it.";

			std::string shapeProblem;
			try {
				shapeProblem = DescribeShapeMismatch(expectedShapes, file, agentName);
			} catch (...) {
				// A file that isn't a plain torch::save archive falls back to the load error itself.
			}

			return shapeProblem.empty() ? CleanLoadErrorText(e.what()) : shapeProblem;
		}
	}
}

namespace RocketForge
{
	InferenceModelSource FromSegmentCheckpoint(std::string agentName, std::filesystem::path checkpointFolder)
	{
		InferenceModelSource source;
		source.agentName = std::move(agentName);

		// <segment>/checkpoints/<timesteps> -> <segment>/training_setup.json
		source.setupSnapshotPath = checkpointFolder.parent_path().parent_path() / TrainingSetupSnapshot::FILE_NAME;

		source.checkpointFolder = std::move(checkpointFolder);
		return source;
	}

	InferenceModelSource FromImportedModel(std::string agentName, std::filesystem::path importedRoot)
	{
		InferenceModelSource source;
		source.agentName = std::move(agentName);
		source.checkpointFolder = GGL::TrainingStorage::ResolveModelFolder(importedRoot);

		// Imported/scripted policies define their network via INFERENCE_POLICY.json, not a
		// training_setup snapshot — leave setupSnapshotPath empty.
		return source;
	}

	InferenceModelSource FromStaticOpponentEntry(const RocketForgeTraining::StaticOpponentSelectionEntry& entry)
	{
		// A segment checkpoint lives at <segment>/checkpoints/<timesteps>; anything else is an
		// imported model folder whose network is defined by INFERENCE_POLICY.json.
		const bool isSegment = entry.checkpointPath.parent_path().filename() == "checkpoints";
		return isSegment
			? FromSegmentCheckpoint(entry.sourceAgentName, entry.checkpointPath)
			: FromImportedModel(entry.sourceAgentName, entry.checkpointPath);
	}

	ResolvedInferenceAgent ResolveInferenceAgent(const InferenceModelSource& source)
	{
		// AgentRegistry::Create throws std::runtime_error for an unknown agent name.
		ResolvedInferenceAgent resolved;
		resolved.agent = AgentRegistry::Create(source.agentName);
		resolved.checkpointFolder = source.checkpointFolder;

		// Saved setup owns physics. Historical segments without metadata used v2;
		// malformed metadata must not silently switch the model to current agent settings.
		if (!source.setupSnapshotPath.empty()) {
			const std::optional<TrainingSetupSnapshot> snapshot = LoadExistingTrainingSetupSnapshot(source.setupSnapshotPath);
			const AgentSetupSnapshot setup = ResolveSavedAgentSetup(snapshot, CaptureAgentSetupSnapshot(*resolved.agent));
			ApplyAgentSetupSnapshot(*resolved.agent, setup);
			if (snapshot && snapshot->agentSetup)
				resolved.appliedSnapshot = setup;
		}

		resolved.agent->modelPath = source.checkpointFolder;
		return resolved;
	}

	std::unique_ptr<GGL::InferUnit> MakeInferUnit(
		const ResolvedInferenceAgent& resolved,
		RLGC::ObsBuilder* obsBuilder,
		RLGC::ActionParser* actionParser,
		bool useGPU)
	{
		// Build through the agent's own MakeInferenceModels ctor so imported/scripted policies load.
		return std::make_unique<GGL::InferUnit>(
			obsBuilder, resolved.agent->GetObsSize(), actionParser, *resolved.agent, resolved.checkpointFolder, useGPU);
	}

	ModelFitCheck CheckModelFitsAgent(const std::string& agentName, const std::filesystem::path& modelFolder)
	{
		ModelFitCheck check;
		GGL::ModelSet models;

		try {
			InferenceModelSource source;
			source.agentName = agentName;
			source.checkpointFolder = modelFolder;

			const ResolvedInferenceAgent resolved = ResolveInferenceAgent(source);
			const std::unique_ptr<RLGC::ActionParser> actionParser(resolved.agent->MakeActionParser());

			const int obsSize = resolved.agent->GetObsSize();
			resolved.agent->MakeInferenceModels(obsSize, actionParser->GetActionAmount(), torch::kCPU, models);

			if (!models["shared_head"] && std::filesystem::exists(modelFolder / "SHARED_HEAD.lt")) {
				check.problem = "The model has a shared head (SHARED_HEAD.lt), but " + agentName + " has no shared head.";
			}

			// The shared head feeds the policy, so a mismatch there is the root cause; report it first.
			std::vector<GGL::Model*> loadOrder;
			if (GGL::Model* sharedHead = models["shared_head"])
				loadOrder.push_back(sharedHead);

			for (GGL::Model* model : models) {
				if (model != models["shared_head"])
					loadOrder.push_back(model);
			}

			for (GGL::Model* model : loadOrder) {
				if (!check.problem.empty())
					break;
				check.problem = DescribeModelLoadProblem(*model, modelFolder, agentName);
			}

			check.fits = check.problem.empty();
		} catch (const std::exception& e) {
			check.problem = CleanLoadErrorText(e.what());
		} catch (...) {
			check.problem = "Unknown error while loading the model.";
		}

		models.Free();
		return check;
	}

	bool ResolveInferenceUseGPU(bool requested)
	{
		if (!requested)
			return false;
		if (torch::cuda::is_available())
			return true;
		std::cerr << "CUDA is not available; running inference on the CPU." << std::endl;
		return false;
	}
}
