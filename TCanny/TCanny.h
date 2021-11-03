#pragma once

#include <algorithm>
#include <limits>
#include <memory>
#include <thread>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>

#include <VapourSynth4.h>
#include <VSHelper4.h>

#ifdef TCANNY_X86
#include "VCL2/vectormath_trig.h"
#endif

static constexpr float M_PIF = 3.14159265358979323846f;
static constexpr float M_1_PIF = 0.318309886183790671538f;
static constexpr float fltMax = std::numeric_limits<float>::max();
static constexpr float fltLowest = std::numeric_limits<float>::lowest();

using unique_float = std::unique_ptr<float[], decltype(&vsh::vsh_aligned_free)>;
using unique_int = std::unique_ptr<int[], decltype(&vsh::vsh_aligned_free)>;

struct TCannyData final {
    VSNode* node;
    const VSVideoInfo* vi;
    float t_h;
    float t_l;
    int mode;
    int op;
    bool process[3];
    float magnitude;
    float offset[3];
    int alignment;
    int peak;
    int radiusAlign;
    int radiusH[3];
    int radiusV[3];
    std::unique_ptr<float[]> weightsH[3];
    std::unique_ptr<float[]> weightsV[3];
    std::unordered_map<std::thread::id, std::unique_ptr<bool[]>> found;
    std::unordered_map<std::thread::id, unique_float> blur;
    std::unordered_map<std::thread::id, unique_float> gradient;
    std::unordered_map<std::thread::id, unique_int> direction;
    void (*filter)(const VSFrame* src, VSFrame* dst, const TCannyData* const VS_RESTRICT d, const VSAPI* vsapi) noexcept;
};

static void hysteresis(float* VS_RESTRICT srcp, bool* VS_RESTRICT found, const int width, const int height, const ptrdiff_t stride,
                       const float t_h, const float t_l) noexcept {
    std::fill_n(found, width * height, false);
    std::vector<std::pair<int, int>> coordinates;

    for (auto y{ 0 }; y < height; y++) {
        for (auto x{ 0 }; x < width; x++) {
            if (!found[width * y + x] && srcp[stride * y + x] >= t_h) {
                srcp[stride * y + x] = fltMax;
                found[width * y + x] = true;

                coordinates.emplace_back(std::make_pair(x, y));

                while (!coordinates.empty()) {
                    const auto& pos{ coordinates.back() };
                    coordinates.pop_back();

                    const auto xxStart{ std::max(pos.first - 1, 0) };
                    const auto xxStop{ std::min(pos.first + 1, width - 1) };
                    const auto yyStart{ std::max(pos.second - 1, 0) };
                    const auto yyStop{ std::min(pos.second + 1, height - 1) };

                    for (auto yy{ yyStart }; yy <= yyStop; yy++) {
                        for (auto xx{ xxStart }; xx <= xxStop; xx++) {
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
