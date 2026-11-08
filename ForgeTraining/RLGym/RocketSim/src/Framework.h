#pragma once

#include <RocketSim/Framework.h>

// The vendored v2 engine lives in RocketSimV2 and sees the shared value types unqualified.
#ifndef RS_NO_NAMESPACE
#define RS_NS_START namespace RocketSimV2 { using namespace RocketSim;
#define RS_NS_END }
#else
#define RS_NS_START using namespace RocketSim;
#define RS_NS_END
#endif
