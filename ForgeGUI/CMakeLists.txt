cmake_minimum_required(VERSION 3.16)

project("ForgeGUI")

set(CMAKE_AUTOUIC ON)
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)

find_package(QT NAMES Qt6 Qt5 REQUIRED COMPONENTS Widgets Sql)
find_package(Qt${QT_VERSION_MAJOR} REQUIRED COMPONENTS Widgets Sql)

if(NOT Qt6_FOUND)
    message(FATAL_ERROR
        "Qt6 not found. Make sure QT is installed in C:/ or the QTDIR environment variable is set"
    )
endif()

file(GLOB_RECURSE PROJECT_SOURCES CONFIGURE_DEPENDS "src/*.cpp" "src/*.h" "src/*.ui")

if(${QT_VERSION_MAJOR} GREATER_EQUAL 6)
    qt_add_executable(ForgeGUI
        MANUAL_FINALIZATION
        ${PROJECT_SOURCES}
        resources/resources.qrc
    )
else()
    add_executable(ForgeGUI
        ${PROJECT_SOURCES}
        resources/resources.qrc
    )
endif()

# Set C++ version to 20
set_target_properties(ForgeGUI PROPERTIES LINKER_LANGUAGE CXX)
set_target_properties(ForgeGUI PROPERTIES CXX_STANDARD 20)

# Add linked libraries
target_link_libraries(ForgeGUI PRIVATE Qt${QT_VERSION_MAJOR}::Widgets Qt${QT_VERSION_MAJOR}::Sql ForgeTraining Agents TrainingPlans)

set_target_properties(ForgeGUI PROPERTIES
    WIN32_EXECUTABLE TRUE
)

target_include_directories(ForgeGUI PRIVATE "src/")

# The Remote Sync panel launches the repo's tools/*.ps1 scripts.
target_compile_definitions(ForgeGUI PRIVATE PROJECT_ROOT="${CMAKE_SOURCE_DIR}")

include(GNUInstallDirs)
install(TARGETS ForgeGUI
    BUNDLE DESTINATION .
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)

if(QT_VERSION_MAJOR EQUAL 6)
    qt_finalize_executable(ForgeGUI)
endif()

get_target_property(_qmake_executable Qt6::qmake IMPORTED_LOCATION)
get_filename_component(_qt_bin_dir "${_qmake_executable}" DIRECTORY)

# The GUI test exes share ForgeGUI's output folder and load the Qt deployed here,
# and they run on the offscreen platform, which windeployqt leaves out by default.
add_custom_command(TARGET ForgeGUI POST_BUILD
    COMMAND "${_qt_bin_dir}/windeployqt.exe"
            --include-plugins qoffscreen
            "$<TARGET_FILE:ForgeGUI>"
)

if(ROCKETFORGE_BUILD_TESTS)
    foreach(kind Picker SavedResults)
        set(benchmark_gui_sources
            tests/ModelBenchmark${kind}_tests.cpp
            src/ModelBenchmark/ModelBenchmarkSavedResultsWidget.cpp
            src/ModelBenchmark/ModelBenchmarkResultTables.cpp
            src/ModelBenchmark/ModelBenchmarkLibrary.cpp
            resources/resources.qrc)
        if(kind STREQUAL "Picker")
            list(APPEND benchmark_gui_sources
                src/Common/ModelCatalog.cpp src/Common/TableSortController.cpp
                src/Widgets/AgentComboBox.cpp src/Widgets/RunProgressPanel.cpp
                src/ModelBenchmark/ModelBenchmarkWidget.cpp src/ModelBenchmark/ModelBenchmarkWorkspace.cpp
                src/ModelBenchmark/ModelBenchmarkWorker.cpp src/SimulatorUnavailableDialog.cpp)
        endif()

        add_executable(ForgeModelBenchmark${kind}Tests ${benchmark_gui_sources})
        target_include_directories(ForgeModelBenchmark${kind}Tests PRIVATE src tests)
        target_link_libraries(ForgeModelBenchmark${kind}Tests PRIVATE Qt6::Widgets Qt6::Sql ForgeTraining)
        target_compile_features(ForgeModelBenchmark${kind}Tests PRIVATE cxx_std_20)
        target_compile_definitions(ForgeModelBenchmark${kind}Tests PRIVATE PROJECT_ROOT="${CMAKE_SOURCE_DIR}")
        add_test(NAME BenchmarkGUI.${kind} COMMAND ForgeModelBenchmark${kind}Tests)
        set_tests_properties(BenchmarkGUI.${kind} PROPERTIES ENVIRONMENT "QT_QPA_PLATFORM=offscreen" TIMEOUT 180)
    endforeach()
endif()

if(ROCKETFORGE_BUILD_TESTS)
    set(dashboard_sources ${PROJECT_SOURCES})
    list(FILTER dashboard_sources EXCLUDE REGEX "/main\\.cpp$")
    add_executable(ForgeTrainingDashboardTests tests/TrainingDashboard_tests.cpp
        ../ForgeTraining/GigaLearn/src/public/GigaLearnCPP/Util/Report.cpp
        ${dashboard_sources} resources/resources.qrc)
    set_target_properties(ForgeTrainingDashboardTests PROPERTIES AUTOUIC_SEARCH_PATHS "${CMAKE_CURRENT_SOURCE_DIR}/src/MainWindow")
    target_include_directories(ForgeTrainingDashboardTests PRIVATE src tests)
    target_link_libraries(ForgeTrainingDashboardTests PRIVATE Qt6::Widgets Qt6::Sql ForgeTraining Agents TrainingPlans)
    target_compile_features(ForgeTrainingDashboardTests PRIVATE cxx_std_20)
    target_compile_definitions(ForgeTrainingDashboardTests PRIVATE PROJECT_ROOT="${CMAKE_SOURCE_DIR}")
    add_test(NAME TrainingGUI.Dashboard COMMAND ForgeTrainingDashboardTests)
    set_tests_properties(TrainingGUI.Dashboard PROPERTIES ENVIRONMENT "QT_QPA_PLATFORM=offscreen" TIMEOUT 120)
endif()