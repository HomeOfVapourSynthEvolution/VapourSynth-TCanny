#pragma once

#include <cmath>
#include <limits>
#include <type_traits>
#include <vector>

#include "shared.hpp"

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
    float * weightsH[3], * weightsV[3], magnitude, offset[3];
    int vectorSize, alignment, radiusH[3], radiusV[3], radiusAlign;
    uint16_t peak;
    std::unordered_map<std::thread::id, float *> blur, gradient;
    std::unordered_map<std::thread::id, unsigned *> direction;
    std::unordered_map<std::thread::id, bool *> found;
    void (*filter)(const VSFrameRef *, VSFrameRef *, const TCannyData * const VS_RESTRICT, const VSAPI *);
};

static void hysteresis(float * VS_RESTRICT srcp, bool * VS_RESTRICT found, const int width, const int height, const int stride, const float t_h, const float t_l) noexcept {
    std::fill_n(found, width * height, false);
    std::vector<std::pair<int, int>> coordinates;

    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            if (!found[width * y + x] && srcp[stride * y + x] >= t_h) {
                srcp[stride * y + x] = fltMax;
                found[width * y + x] = true;

                coordinates.emplace_back(std::make_pair(x, y));

                while (!coordinates.empty()) {
                    const auto pos = coordinates.back();
                    coordinates.pop_back();

                    const int xxStart = std::max(pos.first - 1, 0);
                    const int xxStop = std::min(pos.first + 1, width - 1);
                    const int yyStart = std::max(pos.second - 1, 0);
                    const int yyStop = std::min(pos.second + 1, height - 1);

                    for (int yy = yyStart; yy <= yyStop; yy++) {
                        for (int xx = xxStart; xx <= xxStop; xx++) {
                            if (!found[width * yy + xx] && srcp[stride * yy + xx] >= t_l) {
                                srcp[stride * yy + xx] = fltMax;
                                found[width * yy + xx] = true;

                                coordinates.emplace_back(std::make_pair(xx, yy));
                            }
                        }
                    }
                }
            }
        }
    }
}
