#pragma once
#include "Framework.h"

namespace RLGC {
	// https://github.com/AechPro/rocket-league-gym-sim/blob/main/rlgym_sim/utils/common_values.py
	namespace CommonValues {
		// Could just use RocketSim's RLConst but I'll just copy to stay faithful

		constexpr float TICK_TIME = 1 / 120.f;

		constexpr float
			SIDE_WALL_X = 4096,
			BACK_WALL_Y = 5120,
			CEILING_Z = 2044,
			BACK_NET_Y = 6000,

			SIDE_WALL_X_HOOPS = (8900 / 3.f),
			BACK_WALL_Y_HOOPS = 3581,
			CEILING_Z_HOOPS = 1820,

			RAMP_HEIGHT = 256,

			GOAL_HEIGHT = 642.775,
			GOAL_WIDTH_FROM_CENTER = 892.755f,

			GOAL_HEIGHT_HOOPS = 364.f,
			GOAL_Y_OFFSET_HOOPS = 2770.f,

			GRAVITY_Z = -650.0,

			BOOST_CONSUMED_PER_SECOND = 100.0 / 3.0;

		constexpr Vec
			ORANGE_GOAL_CENTER = Vec(0, BACK_WALL_Y, GOAL_HEIGHT / 2),
			BLUE_GOAL_CENTER = Vec(0, -BACK_WALL_Y, GOAL_HEIGHT / 2),

			ORANGE_GOAL_CENTER_HOOPS = Vec(0, GOAL_Y_OFFSET_HOOPS, GOAL_HEIGHT_HOOPS),
			BLUE_GOAL_CENTER_HOOPS = Vec(0, -GOAL_Y_OFFSET_HOOPS, GOAL_HEIGHT_HOOPS),

			// Often more useful than center
			ORANGE_GOAL_BACK = Vec(0, BACK_NET_Y, GOAL_HEIGHT / 2),
			BLUE_GOAL_BACK = Vec(0, -BACK_NET_Y, GOAL_HEIGHT / 2),

			// Goal posts (at goal line, ground-level center of posts)
			ORANGE_GOAL_LEFT = Vec(-GOAL_WIDTH_FROM_CENTER, BACK_WALL_Y, 0.f),
			ORANGE_GOAL_RIGHT = Vec(GOAL_WIDTH_FROM_CENTER, BACK_WALL_Y, 0.f),

			BLUE_GOAL_LEFT = Vec(-GOAL_WIDTH_FROM_CENTER, -BACK_WALL_Y, 0.f),
			BLUE_GOAL_RIGHT = Vec(GOAL_WIDTH_FROM_CENTER, -BACK_WALL_Y, 0.f);

		constexpr float
			BALL_RADIUS = 92.75, // :nerd::point_up: "erm it is actually 91.25"

			BALL_MAX_SPEED = 6000,
			CAR_MAX_SPEED = 2300,
			SUPERSONIC_THRESHOLD = 2200,
			CAR_MAX_ANG_VEL = 5.5,

			BLUE_TEAM = 0,
			ORANGE_TEAM = 1,
			NUM_ACTIONS = 8;

		constexpr int BOOST_LOCATIONS_AMOUNT = 34;
		constexpr Vec BOOST_LOCATIONS[BOOST_LOCATIONS_AMOUNT] = {
				{0.f, -4240.0, 70.0},
				{-1792.0, -4184.0, 70.0},
				{1792.0, -4184.0, 70.0},
				{-3072.0, -4096.0, 73.0},
				{3072.0, -4096.0, 73.0},
				{-940.0, -3308.0, 70.0},
				{940.0, -3308.0, 70.0},
				{0.0, -2816.0, 70.0},
				{-3584.0, -2484.0, 70.0},
				{3584.0, -2484.0, 70.0},
				{-1788.0, -2300.0, 70.0},
				{1788.0, -2300.0, 70.0},
				{-2048.0, -1036.0, 70.0},
				{0.0, -1024.0, 70.0},
				{2048.0, -1036.0, 70.0},
				{-3584.0, 0.0, 73.0},
				{-1024.0, 0.0, 70.0},
				{1024.0, 0.0, 70.0},
				{3584.0, 0.0, 73.0},
				{-2048.0, 1036.0, 70.0},
				{0.0, 1024.0, 70.0},
				{2048.0, 1036.0, 70.0},
				{-1788.0, 2300.0, 70.0},
				{1788.0, 2300.0, 70.0},
				{-3584.0, 2484.0, 70.0},
				{3584.0, 2484.0, 70.0},
				{0.0, 2816.0, 70.0},
				{-940.0, 3310.0, 70.0},
				{940.0, 3308.0, 70.0},
				{-3072.0, 4096.0, 73.0},
				{3072.0, 4096.0, 73.0},
				{-1792.0, 4184.0, 70.0},
				{1792.0, 4184.0, 70.0},
				{0.0, 4240.0, 70.0}
		};

		constexpr int BOOST_LOCATIONS_AMOUNT_HOOPS = 20;
		constexpr Vec BOOST_LOCATIONS_HOOPS[BOOST_LOCATIONS_AMOUNT_HOOPS] = {
				{1536,	-1024, 64 },
				{-1280,	-2304, 64 },
				{0,		-2816, 64 },
				{-1536,	-1024, 64 },
				{1280,	-2304, 64 },
				{-512,	  512, 64 },
				{-1536,  1024, 64 },
				{1536,	 1024, 64 },
				{1280,	 2304, 64 },
				{0,		 2816, 64 },
				{512,	  512, 64 },
				{512,	 -512, 64 },
				{-512,	 -512, 64 },
				{-1280,	 2304, 64 },
				{-2176,		 2944, 72 },
				{ 2176,		-2944, 72 },
				{-2176,		-2944, 72 },
				{-2432,			0, 72 },
				{ 2432,			0, 72 },
				{ 2175.99f,	 2944, 72 }
		};
	}
}