#include "ExperienceBuffer.h"

using namespace torch;

GGL::ExperienceBuffer::ExperienceBuffer(int seed, torch::Device device) :
	seed(seed), device(device), rng(seed) {

}

GGL::ExperienceTensors GGL::ExperienceBuffer::_GetSamples(const Tensor& indices) const {
	ExperienceTensors result;

	auto* toItr = result.begin();
	auto* fromItr = data.begin();
	for (; toItr != result.end(); toItr++, fromItr++)
		if (fromItr->defined())
			*toItr = torch::index_select(*fromItr, 0, indices);

	return result;
}

std::vector<GGL::ExperienceTensors> GGL::ExperienceBuffer::GetAllBatchesShuffled(int64_t batchSize, bool overbatching) {

	RG_NO_GRAD;

	int64_t expSize = data.states.size(0);

	std::vector<int64_t> indices((size_t)expSize);
	std::iota(indices.begin(), indices.end(), int64_t{ 0 });
	std::shuffle(indices.begin(), indices.end(), rng);
	const bool nonBlockingTransfer = true;
	Tensor tIndices = torch::from_blob(indices.data(), { expSize }, torch::kInt64).to(data.states.device(), nonBlockingTransfer);

	// Get a sample set from each of the batches
	std::vector<ExperienceTensors> result;
	for (int64_t startIdx = 0; startIdx + batchSize <= expSize; startIdx += batchSize) {

		int curBatchSize = batchSize;
		if (startIdx + batchSize * 2 > expSize) {
			// Last batch of the iteration
			if (overbatching) {
				// Extend batch size to the end of the experience
				curBatchSize = expSize - startIdx;
			}
		}

		result.push_back(_GetSamples(tIndices.slice(0, startIdx, startIdx + curBatchSize)));
	}

	return result;
}