#pragma once

#include <RLGymCPP/BasicTypes/Lists.h>
using RLGC::FList;
using RLGC::IList;

#if defined(_MSC_VER)
// MSVC
#define RG_EXPORTED __declspec(dllexport)
#define RG_IMPORTED __declspec(dllimport)
#else
// Everything else (?)
#define RG_EXPORTED __attribute__((visibility("default")))
#define RG_IMPORTED
#endif

#ifdef GIGALEARN_INFER_STATIC
#define RG_IMEXPORT
#elif defined(WITHIN_GGL)
#define RG_IMEXPORT RG_EXPORTED
#else
#define RG_IMEXPORT RG_IMPORTED
#endif

#ifdef GIGALEARN_TRAIN_STATIC
#define GGL_TRAIN_IMEXPORT
#elif defined(WITHIN_GIGALEARN)
#define GGL_TRAIN_IMEXPORT RG_EXPORTED
#else
#define GGL_TRAIN_IMEXPORT RG_IMPORTED
#endif

#define RG_SLEEP(ms) std::this_thread::sleep_for(std::chrono::milliseconds(ms))

#define THREAD_WAIT() RG_SLEEP(2)
