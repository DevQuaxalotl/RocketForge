#pragma once
#include "NormalizationMode.h"
#include "RewardDisplayMode.h"
#include <string>
#include <vector>

namespace GGL { class JsonArchive; }

namespace ForgeDebug {

/** Settings data for runtime reward visualization */
struct DebugSettings 
{
    /** Controls how rewards are normalized from -1 to 1 */
    NormalizationMode normMode = NormalizationMode::RAW_WEIGHTED;

    /** 
    * Whether to show player earned reward values, or the reward values the agent would be trained with
    * For example TRAINING will display rewards post zero sum
    */
    RewardDisplayMode displayMode = RewardDisplayMode::EARNED;

    /** Index of the player to show rewards for. When rendering, this also controls what player is being rendered */
    int  playerIndex = 0;

    /** While true, every reward is debugged regardless of IsDebugEnabled return value */
    bool forceDebugAll = false;

    /** Per-reward debug selection (names of rewards enabled for debug viz) */
    std::vector<std::string> debugEnabledRewards;

    /** If false, no saved selection yet; use each reward's code default. */
    bool hasRewardSelection = false;

   /** Controls runtime game speed. Exp: 10.0 will render the game at 10x speed */
    float gameSpeed = 1.0f;

    /** Index of which team size to always use. -1 = no override */
    int   teamSizeOverrideIndex = -1;  

    /** If true, rendered games use one forced state setter instead of the stage's weighted mix. */
    bool  forceSingleStateSetter = false;

    /** Index among the stage's active (non-zero weight) state setters to force. */
    int   forceStateSetterIndex = 0;

    /** If enabled, rendered games are limited to this amount of game time in seconds */
    bool  maxEpisodeTimeEnabled = false;
    float maxEpisodeTime = 30.0f;

    /** If true, debug episodes use the Build's selected static-opponent registry. */
    bool includeStaticOpponent = false;

    /** Empty selects a random static opponent each episode; otherwise pins this selection key. */
    std::string staticOpponentSelectionKey;

    /** If true, the critic display lists each decomposed-critic head under the total value. */
    bool showCriticHeads = true;

    /** Names of the critic heads shown in the critic heads table. */
    std::vector<std::string> debugEnabledCriticHeads;

    /** If false, no saved critic head selection yet; every head is shown. */
    bool hasCriticHeadSelection = false;

    /** Serializes all fields to/from JSON via the shared JsonArchive. */
    void Serialize(GGL::JsonArchive& ar);
};

} // namespace ForgeDebug
