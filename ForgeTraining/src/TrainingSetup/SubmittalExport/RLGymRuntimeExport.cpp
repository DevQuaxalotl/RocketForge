#include "SubmittalExportInternal.h"

#include <sstream>

namespace fs = std::filesystem;

namespace RocketForgeTraining::SubmittalExportInternal
{
	namespace
	{
		std::string MakeMinimalRLGymFramework()
		{
			return
				"#pragma once\n\n"
				"#include <RocketSim/Includes.h>\n\n"
				"using namespace RocketSim;\n\n"
				"#define RG_LOG(s) { std::cout << s << std::endl; }\n\n"
				"#define RG_NO_COPY(className) \\\n"
				"className(const className&) = delete;  \\\n"
				"className& operator= (const className&) = delete\n\n"
				"#define RG_ERR_CLOSE(s) { \\\n"
				"std::string _errorStr = RS_STR(\"RG FATAL ERROR: \" << s); \\\n"
				"RG_LOG(_errorStr); \\\n"
				"throw std::runtime_error(_errorStr); \\\n"
				"exit(EXIT_FAILURE); \\\n"
				"}\n\n"
				"#ifndef RG_UNSAFE\n"
				"#define RG_ASSERT(cond) { if (!(cond)) { RG_ERR_CLOSE(\"Assertion failed: \" << #cond); } }\n"
				"#else\n"
				"#define RG_ASSERT(cond) {}\n"
				"#endif\n\n"
				"#define RG_DIVIDER std::string(40, '=')\n";
		}

		std::string MakeMinimalRLGymCMake(const std::set<fs::path>& selectedComponentFiles)
		{
			std::ostringstream cmake;
			cmake <<
				"cmake_minimum_required(VERSION 3.8)\n\n"
				"project(RLGym)\n\n"
				"add_library(RLGym STATIC\n"
				"    src/RLGymCPP/Gamestates/GameState.cpp\n"
				"    src/RLGymCPP/Gamestates/Player.cpp\n"
				"    src/RLGymCPP/Gamestates/StateUtil.cpp\n"
				"    src/RLGymCPP/BallPrediction/PredictionContext.cpp\n";

			for (const fs::path& relative : selectedComponentFiles) {
				if (IsCppSourceFile(relative))
					cmake << "    " << relative.generic_string() << "\n";
			}

			cmake <<
				")\n"
				"target_include_directories(RLGym PUBLIC src/)\n"
				"target_compile_features(RLGym PUBLIC cxx_std_20)\n\n"
				"if(ROCKETFORGE_ENABLE_V3)\n"
				"    add_subdirectory(RocketSimV3)\n"
				"else()\n"
				"    set(ROCKETSIM_V3_REASON \"RocketSim v3 isn't built: this package's bot uses only RocketSim v2.\")\n"
				"endif()\n"
				"add_subdirectory(Sim)\n"
				"add_subdirectory(RocketSim)\n"
				"target_link_libraries(RLGym PUBLIC RocketForgeSim)\n";

			return cmake.str();
		}
	}

	void CopyMinimalRLGym(const fs::path& projectRoot, const fs::path& destinationRoot, const std::set<fs::path>& selectedComponentFiles)
	{
		const fs::path sourceRoot = projectRoot / "ForgeTraining" / "RLGym";
		std::set<fs::path> rlgymFiles = {
			"src/RLGymCPP/ActionParsers/ActionParser.h",
			"src/RLGymCPP/BasicTypes/Action.h",
			"src/RLGymCPP/BasicTypes/Lists.h",
			"src/RLGymCPP/CommonValues.h",
			"src/RLGymCPP/Framework.h",
			"src/RLGymCPP/Gamestates/GameState.cpp",
			"src/RLGymCPP/Gamestates/GameState.h",
			"src/RLGymCPP/Gamestates/Player.cpp",
			"src/RLGymCPP/Gamestates/Player.h",
			"src/RLGymCPP/Gamestates/StateUtil.cpp",
			"src/RLGymCPP/Gamestates/StateUtil.h",
			"src/RLGymCPP/Math.h",
			"src/RLGymCPP/ObsBuilders/ObsBuilder.h",
			"src/RLGymCPP/BallPrediction/PredictionContext.cpp",
			"src/RLGymCPP/BallPrediction/PredictionContext.h",
			"src/RLGymCPP/BallPrediction/ProfiledObsBuilder.h",
		};

		rlgymFiles.insert(selectedComponentFiles.begin(), selectedComponentFiles.end());
		CopyExplicitFiles(sourceRoot, destinationRoot, rlgymFiles);
		WriteTextFile(destinationRoot / "CMakeLists.txt", MakeMinimalRLGymCMake(selectedComponentFiles));
		WriteTextFile(destinationRoot / "src" / "RLGymCPP" / "Framework.h", MakeMinimalRLGymFramework());

		const fs::path simSource = sourceRoot / "Sim";
		const fs::path simDestination = destinationRoot / "Sim";
		CopyFile(simSource / "CMakeLists.txt", simDestination / "CMakeLists.txt");
		CopyFile(simSource / "EngineConfig.h.in", simDestination / "EngineConfig.h.in");
		CopyCodeTree(simSource, simDestination, { "src/PhysState.cpp" });

		const fs::path rocketSimSource = sourceRoot / "RocketSim";
		const fs::path rocketSimDestination = destinationRoot / "RocketSim";
		CopyFile(rocketSimSource / "CMakeLists.txt", rocketSimDestination / "CMakeLists.txt");
		CopyFile(rocketSimSource / "LICENSE", rocketSimDestination / "LICENSE");
		CopyCodeTree(rocketSimSource / "src", rocketSimDestination / "src", { "BulletLink.cpp" });

		const fs::path bulletSource = rocketSimSource / "libsrc" / "bullet3-3.24";
		const fs::path bulletDestination = rocketSimDestination / "libsrc" / "bullet3-3.24";
		CopyCodeTree(
			bulletSource,
			bulletDestination,
			{
				"btBulletCollisionCommon.h",
				"btBulletDynamicsCommon.h",
				"BulletCollision/BroadphaseCollision/btAxisSweep3.cpp",
				"BulletCollision/BroadphaseCollision/btAxisSweep3.h",
				"BulletCollision/BroadphaseCollision/btAxisSweep3Internal.h",
				"BulletCollision/BroadphaseCollision/btBroadphaseProxy.cpp",
				"BulletCollision/BroadphaseCollision/btSimpleBroadphase.cpp",
				"BulletCollision/BroadphaseCollision/btSimpleBroadphase.h",
				"BulletCollision/CollisionDispatch/btGhostObject.cpp",
				"BulletCollision/CollisionDispatch/btGhostObject.h",
				"BulletCollision/CollisionDispatch/btSphereBoxCollisionAlgorithm.cpp",
				"BulletCollision/CollisionDispatch/btSphereBoxCollisionAlgorithm.h",
				"BulletCollision/CollisionShapes/btEmptyShape.h",
				"BulletCollision/CollisionShapes/btShapeHull.h",
				"BulletCollision/CollisionShapes/btTetrahedronShape.h",
				"BulletCollision/CollisionShapes/btTriangleBuffer.h",
				"BulletCollision/NarrowPhaseCollision/btComputeGjkEpaPenetration.h",
				"BulletCollision/NarrowPhaseCollision/btGjkCollisionDescription.h",
				"BulletCollision/NarrowPhaseCollision/btGjkEpa3.h",
				"BulletCollision/NarrowPhaseCollision/btMprPenetration.h",
				"BulletCollision/CollisionShapes/btEmptyShape.cpp",
				"BulletCollision/CollisionShapes/btShapeHull.cpp",
				"BulletCollision/CollisionShapes/btTetrahedronShape.cpp",
				"BulletCollision/CollisionShapes/btTriangleBuffer.cpp",
				"LinearMath/btConvexHull.cpp",
				"LinearMath/btConvexHull.h",
				"LinearMath/btDefaultMotionState.h",
				"LinearMath/btImplicitQRSVD.h",
				"LinearMath/btList.h",
				"LinearMath/btMatrixX.h",
				"LinearMath/btModifiedGramSchmidt.h",
				"LinearMath/btPolarDecomposition.cpp",
				"LinearMath/btPolarDecomposition.h",
				"LinearMath/btQuickprof.cpp",
				"LinearMath/btRandom.h",
				"LinearMath/btReducedVector.cpp",
				"LinearMath/btReducedVector.h",
				"LinearMath/btSerializer.cpp",
				"LinearMath/btSerializer64.cpp",
				"LinearMath/btSpatialAlgebra.h",
				"LinearMath/btThreads.cpp",
				"LinearMath/TaskScheduler/btTaskScheduler.cpp",
				"LinearMath/TaskScheduler/btThreadSupportInterface.h",
				"LinearMath/TaskScheduler/btThreadSupportPosix.cpp",
				"LinearMath/TaskScheduler/btThreadSupportWin32.cpp",
			});
		CopyFile(bulletSource / "LICENSE.txt", bulletDestination / "LICENSE.txt");
	}
}
