#pragma once
#include "RewardDebugSnapshot.h"

// Forward declarations so ApplyPendingSettings can be declared without
// pulling in the full RewardDebugCollector or GameState headers here.
namespace ForgeDebug { class RewardDebugCollector; }
namespace RLGC { struct GameState; }

namespace ForgeDebug {

	/**
	 * Abstract base class for reward debug visualizers.
	 *
	 * Render() is called from the trainer thread each frame a new snapshot is
	 * produced.
	 *
	 * ApplyPendingSettings() is called on the trainer thread each frame, just
	 * before Render(). Implementations may override this to apply any pending
	 * setting changes (norm mode, display mode, player index, etc.) directly
	 * into the collector.
	 *
	 * The pointer stored in InferenceDebugRenderConfig::debugRenderer is
	 * non-owning; the caller must ensure the renderer outlives the render loop.
	 **/
	class IDebugRenderer {
	public:
		virtual ~IDebugRenderer() = default;

		/**
		 * Called each frame on the trainer thread before Render().
		 * Implementations may apply pending user-initiated setting changes to
		 * the collector. Default implementation does nothing.
		 **/
		virtual void ApplyPendingSettings(RewardDebugCollector& collector, const RLGC::GameState& gs) {}

		/** Called each frame with the latest reward debug snapshot. **/
		virtual void Render(const RewardDebugSnapshot& snapshot) = 0;
	};

}
