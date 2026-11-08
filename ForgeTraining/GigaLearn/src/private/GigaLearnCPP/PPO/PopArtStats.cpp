#include "PopArtStats.h"

#include <algorithm>
#include <cmath>
#include <utility>

void GGL::PopArtHeadState::Serialize(JsonArchive& ar) {
	ar.Field("name", name);
	ar.Field("initialized", initialized);
	ar.Field("mean", mean);
	ar.Field("mean_sq", meanSq);
}

GGL::PopArtStats::PopArtStats(const std::vector<std::string>& headNames, float beta, float sigmaFloor)
	: _beta(beta), _sigmaFloor(sigmaFloor) {

	_heads.reserve(headNames.size());
	for (const std::string& name : headNames) {
		PopArtHeadState head = {};
		head.name = name;
		_heads.push_back(head);
	}
}

void GGL::PopArtStats::SetHeads(std::vector<PopArtHeadState> heads) {
	RG_ASSERT(heads.size() == _heads.size());
	for (size_t i = 0; i < heads.size(); i++)
		RG_ASSERT(heads[i].name == _heads[i].name);

	_heads = std::move(heads);
}

void GGL::PopArtStats::RemapFrom(const std::vector<PopArtHeadState>& savedHeads) {
	for (PopArtHeadState& head : _heads) {
		const auto saved = std::find_if(savedHeads.begin(), savedHeads.end(), [&](const PopArtHeadState& savedHead) {
			return savedHead.name == head.name;
		});

		if (saved != savedHeads.end()) {
			head = *saved;
		} else {
			PopArtHeadState fresh = {};
			fresh.name = head.name;
			head = fresh;
		}
	}
}

double GGL::PopArtStats::UnflooredSigma(const PopArtHeadState& head) const {
	return std::sqrt(std::max(head.meanSq - head.mean * head.mean, 0.0));
}

torch::Tensor GGL::PopArtStats::GetMeans() const {
	std::vector<float> means(_heads.size());
	for (size_t i = 0; i < _heads.size(); i++)
		means[i] = _heads[i].initialized ? (float)_heads[i].mean : 0.f;

	return torch::tensor(means);
}

torch::Tensor GGL::PopArtStats::GetSigmas() const {
	std::vector<float> sigmas(_heads.size());
	for (size_t i = 0; i < _heads.size(); i++)
		sigmas[i] = _heads[i].initialized ? (float)std::max(UnflooredSigma(_heads[i]), _sigmaFloor) : 1.f;

	return torch::tensor(sigmas);
}

int GGL::PopArtStats::CountHeadsAtFloor() const {
	int count = 0;
	for (const PopArtHeadState& head : _heads)
		if (head.initialized && UnflooredSigma(head) < _sigmaFloor)
			count++;

	return count;
}

torch::Tensor GGL::PopArtStats::Denormalize(const torch::Tensor& normalized) const {
	return normalized * GetSigmas().to(normalized.device()) + GetMeans().to(normalized.device());
}

torch::Tensor GGL::PopArtStats::Normalize(const torch::Tensor& raw) const {
	return (raw - GetMeans().to(raw.device())) / GetSigmas().to(raw.device());
}

GGL::PopArtUpdateResult GGL::PopArtStats::Update(const torch::Tensor& targets, const torch::Tensor& presence) {
	RG_ASSERT(targets.dim() == 2 && targets.size(1) == GetNumHeads());
	RG_ASSERT(presence.sizes() == targets.sizes());

	PopArtUpdateResult result = {};
	result.meansBefore = GetMeans();
	result.sigmasBefore = GetSigmas();
	result.preserveMask = torch::zeros({ GetNumHeads() }, torch::kBool);

	const torch::Tensor cellMask = presence.to(torch::kCPU).ne(0).to(torch::kFloat64);
	const torch::Tensor targets64 = targets.to(torch::kCPU).to(torch::kFloat64);
	const torch::Tensor counts = cellMask.sum(0);
	const torch::Tensor sums = (targets64 * cellMask).sum(0);
	const torch::Tensor sumSquares = (targets64.square() * cellMask).sum(0);

	auto countValues = counts.accessor<double, 1>();
	auto sumValues = sums.accessor<double, 1>();
	auto sumSquareValues = sumSquares.accessor<double, 1>();
	auto preserveValues = result.preserveMask.accessor<bool, 1>();

	for (int headIdx = 0; headIdx < GetNumHeads(); headIdx++) {
		PopArtHeadState& head = _heads[headIdx];
		preserveValues[headIdx] = head.initialized;

		const double count = countValues[headIdx];
		if (count <= 0.0)
			continue;

		const double batchMean = sumValues[headIdx] / count;
		const double batchMeanSq = sumSquareValues[headIdx] / count;

		if (head.initialized) {
			head.mean = (1.0 - _beta) * head.mean + _beta * batchMean;
			head.meanSq = (1.0 - _beta) * head.meanSq + _beta * batchMeanSq;
		} else {
			head.mean = batchMean;
			head.meanSq = batchMeanSq;
			head.initialized = true;
		}
	}

	return result;
}

void GGL::PopArtStats::PreserveOutputs(torch::Tensor weight, torch::Tensor bias, const PopArtUpdateResult& update) const {
	RG_ASSERT(weight.size(0) == GetNumHeads() && bias.size(0) == GetNumHeads());

	torch::NoGradGuard noGrad;
	const torch::Device device = weight.device();
	const torch::Tensor preserve = update.preserveMask.to(device);
	const torch::Tensor sigmasBefore = update.sigmasBefore.to(device);
	const torch::Tensor meansBefore = update.meansBefore.to(device);
	const torch::Tensor sigmasAfter = GetSigmas().to(device);
	const torch::Tensor meansAfter = GetMeans().to(device);

	// Raw output σ·(w·x + b) + μ is unchanged when w' = (σ/σ')·w and b' = (σ·b + μ − μ')/σ'.
	const torch::Tensor weightScale = torch::where(preserve, sigmasBefore / sigmasAfter, torch::ones_like(sigmasAfter));
	const torch::Tensor preservedBias = (sigmasBefore * bias + meansBefore - meansAfter) / sigmasAfter;

	weight.mul_(weightScale.unsqueeze(1));
	bias.copy_(torch::where(preserve, preservedBias, bias));
}
