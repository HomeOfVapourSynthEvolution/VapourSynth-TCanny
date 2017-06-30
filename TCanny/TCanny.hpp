#pragma once

#include <algorithm>
#include <limits>
#include <thread>
#include <type_traits>
#include <unordered_map>

#include <VapourSynth.h>
#include <VSHelper.h>

#ifdef VS_TARGET_CPU_X86
#include "vectorclass/vectormath_trig.h"
#endif

static constexpr float M_PIF = 3.14159265358979323846f;
static constexpr float M_1_PIF = 0.318309886183790671538f;
static constexpr float fltMax = std::numeric_limits<float>::max();
static constexpr float fltLowest = std::numeric_limits<float>::lowest();

struct TCannyData {
    VSNodeRef * node;
    const VSVideoInfo * vi;
    float t_h, t_l;
    int mode, op;
    bool process[3];
    float * weightsHorizontal[3], * weightsVertical[3];
    int radiusHorizontal[3], radiusVertical[3];
    float magnitude;
    unsigned radiusAlign, bins;
    uint16_t peak;
    float offset[3], lower[3], upper[3];
    std::unordered_map<std::thread::id, float *> buffer, blur, gradient, direction;
    std::unordered_map<std::thread::id, bool *> label;
};
