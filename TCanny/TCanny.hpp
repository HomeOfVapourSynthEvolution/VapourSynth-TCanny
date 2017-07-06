#pragma once

#include <algorithm>
#include <cmath>
#include <limits>
#include <memory>
#include <string>
#include <type_traits>

#include <VapourSynth.h>
#include <VSHelper.h>

#ifdef VS_TARGET_CPU_X86
#include "vectorclass/vectormath_trig.h"
#endif

static constexpr float M_PIF = 3.14159265358979323846f;
static constexpr float M_1_PIF = 0.318309886183790671538f;
static constexpr float fltMax = std::numeric_limits<float>::max();
static constexpr float fltLowest = std::numeric_limits<float>::lowest();

static float * gaussianWeights(const float sigma, int * radius) noexcept {
    const unsigned diameter = std::max<unsigned>(sigma * 3.f + 0.5f, 1) * 2 + 1;
    *radius = diameter / 2;
    float sum = 0.f;

    float * VS_RESTRICT weights = new (std::nothrow) float[diameter];
    if (!weights)
        return nullptr;

    for (int k = -(*radius); k <= *radius; k++) {
        const float w = std::exp(-(k * k) / (2.f * sigma * sigma));
        weights[k + *radius] = w;
        sum += w;
    }

    for (unsigned k = 0; k < diameter; k++)
        weights[k] /= sum;

    return weights;
}
