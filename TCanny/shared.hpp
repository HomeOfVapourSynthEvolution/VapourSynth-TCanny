#pragma once

#include <algorithm>
#include <memory>
#include <string>
#include <thread>
#include <unordered_map>

#include <VapourSynth.h>
#include <VSHelper.h>

static inline float * gaussianWeights(const float sigma, int & radius) noexcept {
    const int diameter = std::max<int>(sigma * 3.f + 0.5f, 1) * 2 + 1;
    radius = diameter / 2;

    float * VS_RESTRICT weights = new (std::nothrow) float[diameter];
    if (!weights)
        return nullptr;

    float sum = 0.f;

    for (int k = -radius; k <= radius; k++) {
        const float w = std::exp(-(k * k) / (2.f * sigma * sigma));
        weights[k + radius] = w;
        sum += w;
    }

    for (int k = 0; k < diameter; k++)
        weights[k] /= sum;

    return weights;
}
