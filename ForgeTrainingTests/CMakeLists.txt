add_executable(ForgeTrainingTests
    GTestMain.cpp
    TestSupport.cpp
    TestSupport_tests.cpp
    AdvancedObsPadded_tests.cpp
    AgentComponentRegistry_tests.cpp
    ForgeSettings_tests.cpp
    AgentSetupSnapshot_tests.cpp
    AgentSimulation_tests.cpp
    InferenceModelSource_tests.cpp
    SubmittalExport_tests.cpp
    BuildStorage_tests.cpp
    BotScaffold_tests.cpp
    SetupHashStability_tests.cpp
    EngineRegistry_tests.cpp
    SimConformance_tests.cpp
    SimEngineDifferences_tests.cpp
    ImportedModelStorage_tests.cpp
    StaticOpponentSelectionStorage_tests.cpp
    SkillBenchmarkRegistry_tests.cpp
    DebugSettings_tests.cpp
    StageQueueRunner_tests.cpp
    SyncManifest_tests.cpp
    StageRegistry_tests.cpp
    BallPredBuffer_tests.cpp
    BallPredictionContext_tests.cpp
    ModelBenchmark_tests.cpp
    PolicyVersionManager_tests.cpp
    SkillEstimation_tests.cpp
    MetricSender_tests.cpp
    RecentOpponentScheduler_tests.cpp
    RecentPolicyEnsemble_tests.cpp
    ModelManagerCudaGraph_tests.cpp
    CarMechanics_tests.cpp
    PPG_tests.cpp
    TransferLearning_tests.cpp
    CriticHeadModel_tests.cpp
    CriticHeadRemap_tests.cpp
    GAE_tests.cpp
    ScheduledValue_tests.cpp
    SimulationBuffer_tests.cpp
    EnvSet_tests.cpp
    TrainingStorage_tests.cpp
    AgentModelPaths_tests.cpp
    PublicBots_tests.cpp
    ForgeDebugSettings_tests.cpp
    ThreadPool_tests.cpp
    RewardDebugCollector_tests.cpp
    RewardBoxLayout_tests.cpp
    ValueTrendTracker_tests.cpp
    RewardName_tests.cpp
    RewardEvaluation_tests.cpp
    MultiPartReward_tests.cpp
    AdaptiveRewardWeight_tests.cpp
    AdaptiveWeightSchedule_tests.cpp
    AdaptiveWeightPersistence_tests.cpp
    PerModeCriticStats_tests.cpp
    AdvantageNormalization_tests.cpp
    PopArtStats_tests.cpp
    DecomposedCritic_tests.cpp
    DecomposedGAE_tests.cpp
    PerTeamSize_tests.cpp
    RewardWeight_tests.cpp
    AdaptiveWeightConfig_tests.cpp
    PersistedAdaptiveWeights_tests.cpp
    TrainingStage_tests.cpp
    RewardAnalysis_tests.cpp
    DebugRenderFocus_tests.cpp
    JsonArchive_tests.cpp
    TrainingSetupSnapshot_tests.cpp
    RewardHeadLayout_tests.cpp
    CriticValueMath_tests.cpp
    CriticProbe_tests.cpp
    RewardDisplayRow_tests.cpp
    CriticHeadRows_tests.cpp
    ModelBenchmarkLibrary_tests.cpp
    CheckpointFingerprint_tests.cpp
    ModelBenchmarkExport_tests.cpp
    ModelBenchmarkRoster_tests.cpp
    Sha256_tests.cpp
    RemoteSyncCommand_tests.cpp
    RemoteSyncRunner_tests.cpp
    TrainingWorker_tests.cpp
    ToolWorkers_tests.cpp
    SimulatorUnavailableDialog_tests.cpp
    DebugBridge_tests.cpp
    ArenaDemo_tests.cpp
    SimulationEquivalence_tests.cpp
    SnapshotReplay_tests.cpp
    SimulationAnalysisOptions_tests.cpp
    SteadyStateSps_tests.cpp
    ConsoleControl_tests.cpp
    CliArgs_tests.cpp
    CliResolve_tests.cpp
    Confirm_tests.cpp
    CommandTable_tests.cpp
    TrainCommand_tests.cpp
    ListCommands_tests.cpp
    ModelCommands_tests.cpp
    BenchmarkCommands_tests.cpp
    DebugRenderTarget_tests.cpp
    DebugCommand_tests.cpp
    PlumbingCommands_tests.cpp
    CreateBotCommand_tests.cpp
    ${CMAKE_SOURCE_DIR}/ForgeConsoleTrainer/src/SimulationAnalysis/SnapshotReplay.cpp
    ${CMAKE_SOURCE_DIR}/ForgeConsoleTrainer/src/Cli/ConsoleControl.cpp
    ${CMAKE_SOURCE_DIR}/ForgeConsoleTrainer/src/Cli/CliArgs.cpp
    ${CMAKE_SOURCE_DIR}/ForgeConsoleTrainer/src/Cli/ClosestMatches.cpp
    ${CMAKE_SOURCE_DIR}/ForgeConsoleTrainer/src/Cli/CliResolve.cpp
    ${CMAKE_SOURCE_DIR}/ForgeConsoleTrainer/src/Cli/Confirm.cpp
    ${CMAKE_SOURCE_DIR}/ForgeConsoleTrainer/src/Cli/TextTable.cpp
    ${CMAKE_SOURCE_DIR}/ForgeConsoleTrainer/src/Cli/CommandTable.cpp
    ${CMAKE_SOURCE_DIR}/ForgeConsoleTrainer/src/Commands/PlumbingCommands.cpp
    ${CMAKE_SOURCE_DIR}/ForgeConsoleTrainer/src/Commands/CreateBotCommand.cpp
    ${CMAKE_SOURCE_DIR}/ForgeConsoleTrainer/src/Commands/TrainCommand.cpp
    ${CMAKE_SOURCE_DIR}/ForgeConsoleTrainer/src/Commands/ListCommands.cpp
    ${CMAKE_SOURCE_DIR}/ForgeConsoleTrainer/src/Commands/ModelCommands.cpp
    ${CMAKE_SOURCE_DIR}/ForgeConsoleTrainer/src/Commands/DebugCommand.cpp
    ${CMAKE_SOURCE_DIR}/ForgeConsoleTrainer/src/Commands/AnalysisCommands.cpp
    ${CMAKE_SOURCE_DIR}/ForgeConsoleTrainer/src/Commands/BenchmarkCommands.cpp
    ${CMAKE_SOURCE_DIR}/ForgeConsoleTrainer/src/SimulationAnalysis/AgentSetup.cpp
    ${CMAKE_SOURCE_DIR}/ForgeConsoleTrainer/src/SimulationAnalysis/Inference.cpp
    ${CMAKE_SOURCE_DIR}/ForgeConsoleTrainer/src/SimulationAnalysis/Options.cpp
    ${CMAKE_SOURCE_DIR}/ForgeConsoleTrainer/src/SimulationAnalysis/SimulationAnalysis.cpp
    ${CMAKE_SOURCE_DIR}/ForgeConsoleTrainer/src/SimulationAnalysis/SimulationState.cpp
    ${CMAKE_SOURCE_DIR}/ForgeConsoleTrainer/src/SimulationAnalysis/SummaryAnalysis.cpp
    ${CMAKE_SOURCE_DIR}/ForgeConsoleTrainer/src/SimulationAnalysis/TraceOutput.cpp
)

# GigaLearn.dll doesn't export these private classes on Windows; on Linux the .so shares them, and a second copy would double-destroy their globals.
if(WIN32)
    target_sources(ForgeTrainingTests PRIVATE
        ${CMAKE_SOURCE_DIR}/ForgeTraining/GigaLearn/src/private/GigaLearnCPP/PolicyVersionManager.cpp
        ${CMAKE_SOURCE_DIR}/ForgeTraining/GigaLearn/src/private/GigaLearnCPP/SkillEstimation.cpp
        ${CMAKE_SOURCE_DIR}/ForgeTraining/GigaLearn/src/private/GigaLearnCPP/SkillFixedSeries.cpp
        ${CMAKE_SOURCE_DIR}/ForgeTraining/GigaLearn/src/private/GigaLearnCPP/PPO/GAE.cpp
        ${CMAKE_SOURCE_DIR}/ForgeTraining/GigaLearn/src/private/GigaLearnCPP/AdaptiveWeightPersistence.cpp
        ${CMAKE_SOURCE_DIR}/ForgeTraining/GigaLearn/src/private/GigaLearnCPP/PPO/PerModeCriticStats.cpp
        ${CMAKE_SOURCE_DIR}/ForgeTraining/GigaLearn/src/private/GigaLearnCPP/PPO/AdvantageNormalization.cpp
        ${CMAKE_SOURCE_DIR}/ForgeTraining/GigaLearn/src/private/GigaLearnCPP/PPO/PopArtStats.cpp
        ${CMAKE_SOURCE_DIR}/ForgeTraining/GigaLearn/src/private/GigaLearnCPP/PPO/DecomposedCritic.cpp
        ${CMAKE_SOURCE_DIR}/ForgeTraining/GigaLearn/src/private/GigaLearnCPP/PPO/DecomposedGAE.cpp
    )
endif()

target_include_directories(ForgeTrainingTests PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/ForgeTraining/src
    ${CMAKE_SOURCE_DIR}/ForgeTraining/GigaLearn/src/private
    ${CMAKE_SOURCE_DIR}/ForgeGUI/src
    ${CMAKE_SOURCE_DIR}/ForgeConsoleTrainer/src
)

# Agents brings the REGISTER_AGENT factories so AgentModelPaths_tests can resolve
# real agents via AgentRegistry::Create.
target_link_libraries(ForgeTrainingTests PRIVATE ForgeTraining Agents TrainingPlans GTest::gtest)
target_compile_features(ForgeTrainingTests PRIVATE cxx_std_20)
target_compile_definitions(ForgeTrainingTests PRIVATE FORGE_SOURCE_DIR="${CMAKE_SOURCE_DIR}")

# Tests of ForgeGUI's Qt-only units build only where Qt is installed.
# Without Qt, each such test file compiles to a single skipped test.
find_package(Qt6 COMPONENTS Core Sql QUIET)
if(Qt6_FOUND)
    target_sources(ForgeTrainingTests PRIVATE ${CMAKE_SOURCE_DIR}/ForgeGUI/src/ModelBenchmark/ModelBenchmarkLibrary.cpp)
    target_sources(ForgeTrainingTests PRIVATE ${CMAKE_SOURCE_DIR}/ForgeGUI/src/RemoteSync/RemoteSyncCommand.cpp)
    target_sources(ForgeTrainingTests PRIVATE ${CMAKE_SOURCE_DIR}/ForgeGUI/src/RemoteSync/RemoteSyncRunner.cpp)
    target_sources(ForgeTrainingTests PRIVATE ${CMAKE_SOURCE_DIR}/ForgeGUI/src/TrainingSetup/TrainingWorker.cpp)
    target_sources(ForgeTrainingTests PRIVATE ${CMAKE_SOURCE_DIR}/ForgeGUI/src/ModelBenchmark/ModelBenchmarkWorker.cpp)
    target_sources(ForgeTrainingTests PRIVATE ${CMAKE_SOURCE_DIR}/ForgeGUI/src/ParamTesting/ParamTestingWorker.cpp)
    target_sources(ForgeTrainingTests PRIVATE ${CMAKE_SOURCE_DIR}/ForgeGUI/src/DebugBridge.cpp)
    set_target_properties(ForgeTrainingTests PROPERTIES AUTOMOC ON)
    target_link_libraries(ForgeTrainingTests PRIVATE Qt6::Core Qt6::Sql)
    target_compile_definitions(ForgeTrainingTests PRIVATE ROCKETFORGE_TESTS_HAVE_QT)

    if(WIN32)
        get_target_property(_tests_qmake_executable Qt6::qmake IMPORTED_LOCATION)
        get_filename_component(_tests_qt_bin_dir "${_tests_qmake_executable}" DIRECTORY)
        add_custom_command(TARGET ForgeTrainingTests POST_BUILD
            COMMAND "${_tests_qt_bin_dir}/windeployqt.exe"
                    --no-translations
                    "$<TARGET_FILE:ForgeTrainingTests>"
        )
    endif()
endif()

# Listing happens when CTest runs rather than as a post-build step, so builds never launch the exe.
gtest_discover_tests(ForgeTrainingTests
    DISCOVERY_MODE PRE_TEST
    DISCOVERY_TIMEOUT 60
    PROPERTIES TIMEOUT 120 LABELS ForgeTrainingTests
)

add_executable(ForgeBuildStorageProcessWorker BuildStorageProcessWorker.cpp)
target_link_libraries(ForgeBuildStorageProcessWorker PRIVATE ForgeTraining)
target_compile_features(ForgeBuildStorageProcessWorker PRIVATE cxx_std_20)
add_dependencies(ForgeTrainingTests ForgeBuildStorageProcessWorker)

set_property(SOURCE BuildStorage_tests.cpp APPEND PROPERTY COMPILE_DEFINITIONS
    BUILD_STORAGE_WORKER_PATH="$<TARGET_FILE:ForgeBuildStorageProcessWorker>")
