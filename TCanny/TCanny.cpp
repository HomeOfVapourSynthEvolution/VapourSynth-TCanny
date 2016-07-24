/*
**   VapourSynth port by HolyWu
**
**                 tcanny v1.0 for Avisynth 2.5.x
**
**   Copyright (C) 2009 Kevin Stone
**
**   This program is free software; you can redistribute it and/or modify
**   it under the terms of the GNU General Public License as published by
**   the Free Software Foundation; either version 2 of the License, or
**   (at your option) any later version.
**
**   This program is distributed in the hope that it will be useful,
**   but WITHOUT ANY WARRANTY; without even the implied warranty of
**   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
**   GNU General Public License for more details.
**
**   You should have received a copy of the GNU General Public License
**   along with this program; if not, write to the Free Software
**   Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
*/

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstdlib>
#include <vapoursynth/VapourSynth.h>
#include <vapoursynth/VSHelper.h>
#ifdef VS_TARGET_CPU_X86
#include "vectorclass/vectormath_trig.h"
#endif

static constexpr float M_PIF = 3.14159265358979323846f;
static constexpr float M_1_PIF = 0.318309886183790671538f;

struct TCannyData {
    VSNodeRef * node;
    const VSVideoInfo * vi;
    float sigma, t_h, t_l, gmmax;
    int mode, op;
    bool process[3];
    int radius, radiusAlign, bins;
    float * weights;
    float magnitude;
    int peak;
    float lower[3], upper[3];
#ifdef VS_TARGET_CPU_X86
    void (*gaussianBlurVertical)(const uint8_t * srcp, float * buffer, float * dstp, const float * weights, const int width, const int height, const int stride, const int blurStride, const int radius, const float offset);
#endif
};

struct Stack {
    uint8_t * map;
    std::pair<int, int> * pos;
    int index;
};

static inline void push(Stack & s, const int x, const int y) {
    s.pos[++s.index].first = x;
    s.pos[s.index].second = y;
}

static inline std::pair<int, int> pop(Stack & s) {
    return s.pos[s.index--];
}

static inline float scale(const float val, const int bits) {
    return val * ((1 << bits) - 1) / 255.f;
}

static float * gaussianWeights(const float sigma, int * radius) {
    const int diameter = std::max(static_cast<int>(sigma * 3.f + 0.5f), 1) * 2 + 1;
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

    for (int k = 0; k < diameter; k++)
        weights[k] /= sum;

    return weights;
}

template<typename T>
static T getBin(const float dir, const int n) {
    const int bin = static_cast<int>(dir * n * M_1_PIF + 0.5f);
    return (bin >= n) ? 0 : bin;
}

template<>
float getBin<float>(const float dir, const int n) {
    const float bin = dir * M_1_PIF;
    return (bin > n) ? 0.f : bin;
}

#ifdef VS_TARGET_CPU_X86
static void gaussianBlurHorizontal(float * buffer, float * blur, const float * weights, const int width, const int radius) {
    for (int i = 1; i <= radius; i++) {
        buffer[-i] = buffer[i - 1];
        buffer[width - 1 + i] = buffer[width - i];
    }

    for (int x = 0; x < width; x += 8) {
        Vec8f sum { 0.f };

        for (int i = -radius; i <= radius; i++) {
            const Vec8f srcp = Vec8f().load(buffer + x + i);
            sum = mul_add(srcp, weights[i], sum);
        }

        sum.store_a(blur + x);
    }
}

static void gaussianBlurVertical_uint8(const uint8_t * __srcp, float * buffer, float * blur, const float * weights, const int width, const int height,
                                       const int stride, const int blurStride, const int radius, const float offset) {
    const int diameter = radius * 2 + 1;
    const uint8_t ** _srcp = new const uint8_t *[diameter];

    _srcp[radius] = __srcp;
    for (int i = 1; i <= radius; i++) {
        _srcp[radius - i] = _srcp[radius + i - 1];
        _srcp[radius + i] = _srcp[radius] + stride * i;
    }

    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x += 8) {
            Vec8f sum { 0.f };

            for (int i = 0; i < diameter; i++) {
#if defined(__AVX2__)
                const Vec8i srcp_8i = _mm256_cvtepu8_epi32(_mm_loadl_epi64(reinterpret_cast<const __m128i *>(_srcp[i] + x)));
#elif defined(__SSE4_1__)
                const Vec8i srcp_8i = Vec8i(_mm_cvtepu8_epi32(_mm_loadl_epi64(reinterpret_cast<const __m128i *>(_srcp[i] + x))),
                                            _mm_cvtepu8_epi32(_mm_loadl_epi64(reinterpret_cast<const __m128i *>(_srcp[i] + x + 4))));
#else
                const Vec16uc srcp_16uc { _mm_loadl_epi64(reinterpret_cast<const __m128i *>(_srcp[i] + x)) };
                const Vec8us srcp_8us = extend_low(srcp_16uc);
                const Vec8i srcp_8i = Vec8i(extend_low(srcp_8us), extend_high(srcp_8us));
#endif
                const Vec8f srcp = to_float(srcp_8i);
                sum = mul_add(srcp, weights[i], sum);
            }

            sum.store_a(buffer + x);
        }

        gaussianBlurHorizontal(buffer, blur, weights + radius, width, radius);

        for (int i = 0; i < diameter - 1; i++)
            _srcp[i] = _srcp[i + 1];
        if (y < height - 1 - radius)
            _srcp[diameter - 1] += stride;
        else if (y > height - 1 - radius)
            _srcp[diameter - 1] -= stride;
        blur += blurStride;
    }

    delete[] _srcp;
}

static void gaussianBlurVertical_uint16(const uint8_t * __srcp, float * buffer, float * blur, const float * weights, const int width, const int height,
                                        const int stride, const int blurStride, const int radius, const float offset) {
    const int diameter = radius * 2 + 1;
    const uint16_t ** _srcp = new const uint16_t *[diameter];

    _srcp[radius] = reinterpret_cast<const uint16_t *>(__srcp);
    for (int i = 1; i <= radius; i++) {
        _srcp[radius - i] = _srcp[radius + i - 1];
        _srcp[radius + i] = _srcp[radius] + stride * i;
    }

    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x += 8) {
            Vec8f sum { 0.f };

            for (int i = 0; i < diameter; i++) {
                const Vec8us srcp_8us = Vec8us().load_a(_srcp[i] + x);
#if defined(__AVX2__)
                const Vec8i srcp_8i = _mm256_cvtepu16_epi32(srcp_8us);
#else
                const Vec8i srcp_8i = Vec8i(extend_low(srcp_8us), extend_high(srcp_8us));
#endif
                const Vec8f srcp = to_float(srcp_8i);
                sum = mul_add(srcp, weights[i], sum);
            }

            sum.store_a(buffer + x);
        }

        gaussianBlurHorizontal(buffer, blur, weights + radius, width, radius);

        for (int i = 0; i < diameter - 1; i++)
            _srcp[i] = _srcp[i + 1];
        if (y < height - 1 - radius)
            _srcp[diameter - 1] += stride;
        else if (y > height - 1 - radius)
            _srcp[diameter - 1] -= stride;
        blur += blurStride;
    }

    delete[] _srcp;
}

static void gaussianBlurVertical_float(const uint8_t * __srcp, float * buffer, float * blur, const float * weights, const int width, const int height,
                                       const int stride, const int blurStride, const int radius, const float offset) {
    const int diameter = radius * 2 + 1;
    const float ** _srcp = new const float *[diameter];

    _srcp[radius] = reinterpret_cast<const float *>(__srcp);
    for (int i = 1; i <= radius; i++) {
        _srcp[radius - i] = _srcp[radius + i - 1];
        _srcp[radius + i] = _srcp[radius] + stride * i;
    }

    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x += 8) {
            Vec8f sum { 0.f };

            for (int i = 0; i < diameter; i++) {
                const Vec8f srcp = Vec8f().load_a(_srcp[i] + x);
                sum = mul_add(srcp + offset, weights[i], sum);
            }

            sum.store_a(buffer + x);
        }

        gaussianBlurHorizontal(buffer, blur, weights + radius, width, radius);

        for (int i = 0; i < diameter - 1; i++)
            _srcp[i] = _srcp[i + 1];
        if (y < height - 1 - radius)
            _srcp[diameter - 1] += stride;
        else if (y > height - 1 - radius)
            _srcp[diameter - 1] -= stride;
        blur += blurStride;
    }

    delete[] _srcp;
}

static void detectEdge(float * blur, float * gradient, float * direction, const int width, const int height, const int stride, const int blurStride, const int mode, const int op) {
    float * srcpp = blur;
    float * srcp = blur;
    float * srcpn = blur + blurStride;

    srcp[-1] = srcp[0];
    srcp[width] = srcp[width - 1];

    for (int y = 0; y < height; y++) {
        srcpn[-1] = srcpn[0];
        srcpn[width] = srcpn[width - 1];

        for (int x = 0; x < width; x += 8) {
            Vec8f gx, gy;

            if (op == 0) {
                gx = Vec8f().load(srcp + x + 1) - Vec8f().load(srcp + x - 1);
                gy = Vec8f().load_a(srcpp + x) - Vec8f().load_a(srcpn + x);
            } else if (op == 1) {
                gx = (Vec8f().load(srcpp + x + 1) + Vec8f().load(srcp + x + 1) + Vec8f().load(srcpn + x + 1)
                    - Vec8f().load(srcpp + x - 1) - Vec8f().load(srcp + x - 1) - Vec8f().load(srcpn + x - 1)) * 0.5f;
                gy = (Vec8f().load(srcpp + x - 1) + Vec8f().load_a(srcpp + x) + Vec8f().load(srcpp + x + 1)
                    - Vec8f().load(srcpn + x - 1) - Vec8f().load_a(srcpn + x) - Vec8f().load(srcpn + x + 1)) * 0.5f;
            } else if (op == 2) {
                gx = Vec8f().load(srcpp + x + 1) + mul_add(2.f, Vec8f().load(srcp + x + 1), Vec8f().load(srcpn + x + 1))
                    - Vec8f().load(srcpp + x - 1) - mul_add(2.f, Vec8f().load(srcp + x - 1), Vec8f().load(srcpn + x - 1));
                gy = Vec8f().load(srcpp + x - 1) + mul_add(2.f, Vec8f().load_a(srcpp + x), Vec8f().load(srcpp + x + 1))
                    - Vec8f().load(srcpn + x - 1) - mul_add(2.f, Vec8f().load_a(srcpn + x), Vec8f().load(srcpn + x + 1));
            } else {
                gx = mul_add(3.f, Vec8f().load(srcpp + x + 1), mul_add(10.f, Vec8f().load(srcp + x + 1), 3.f * Vec8f().load(srcpn + x + 1)))
                    - mul_add(3.f, Vec8f().load(srcpp + x - 1), mul_add(10.f, Vec8f().load(srcp + x - 1), 3.f * Vec8f().load(srcpn + x - 1)));
                gy = mul_add(3.f, Vec8f().load(srcpp + x - 1), mul_add(10.f, Vec8f().load_a(srcpp + x), 3.f * Vec8f().load(srcpp + x + 1)))
                    - mul_add(3.f, Vec8f().load(srcpn + x - 1), mul_add(10.f, Vec8f().load_a(srcpn + x), 3.f * Vec8f().load(srcpn + x + 1)));
            }

            sqrt(mul_add(gx, gx, gy * gy)).store_a(gradient + x);

            if (mode != 1) {
                const Vec8f dr = atan2(gy, gx);
                if_add(dr < 0.f, dr, M_PIF).store_a(direction + x);
            }
        }

        srcpp = srcp;
        srcp = srcpn;
        if (y < height - 2)
            srcpn += blurStride;
        gradient += stride;
        direction += stride;
    }
}

static void nonMaximumSuppression(const float * _gradient, const float * _direction, float * blur, const int width, const int height, const int stride, const int blurStride) {
    for (int x = 0; x < width; x += 8)
        Vec8f(-FLT_MAX).store_a(blur + x);

    for (int y = 1; y < height - 1; y++) {
        _gradient += stride;
        _direction += stride;
        blur += blurStride;

        for (int x = 1; x < width - 1; x += 8) {
            const Vec8f direction = Vec8f().load(_direction + x);
            const Vec8i bin = truncate_to_int(mul_add(direction, 4.f * M_1_PIF, 0.5f));

            Vec8fb mask { bin == 0 | bin >= 4 };
            Vec8f gradient = max(Vec8f().load(_gradient + x + 1), Vec8f().load_a(_gradient + x - 1));
            Vec8f result { gradient & mask };

            mask = bin == 1;
            gradient = max(Vec8f().load(_gradient + x - stride + 1), Vec8f().load_a(_gradient + x + stride - 1));
            result |= gradient & mask;

            mask = bin == 2;
            gradient = max(Vec8f().load(_gradient + x - stride), Vec8f().load(_gradient + x + stride));
            result |= gradient & mask;

            mask = bin == 3;
            gradient = max(Vec8f().load_a(_gradient + x - stride - 1), Vec8f().load(_gradient + x + stride + 1));
            result |= gradient & mask;

            gradient = Vec8f().load(_gradient + x);
            select(gradient >= result, gradient, -FLT_MAX).store(blur + x);
        }

        blur[0] = blur[width - 1] = -FLT_MAX;
    }

    blur += blurStride;

    for (int x = 0; x < width; x += 8)
        Vec8f(-FLT_MAX).store_a(blur + x);
}
#else
static void gaussianBlurHorizontal(float * VS_RESTRICT buffer, float * VS_RESTRICT blur, const float * weights, const int width, const int radius) {
    for (int i = 1; i <= radius; i++) {
        buffer[-i] = buffer[i - 1];
        buffer[width - 1 + i] = buffer[width - i];
    }

    for (int x = 0; x < width; x++) {
        float sum = 0.f;

        for (int i = -radius; i <= radius; i++)
            sum += buffer[x + i] * weights[i];

        blur[x] = sum;
    }
}

template<typename T>
static void gaussianBlurVertical(const T * _srcp, float * VS_RESTRICT buffer, float * VS_RESTRICT blur, const float * weights, const int width, const int height,
                                 const int stride, const int blurStride, const int radius, const float offset) {
    const int diameter = radius * 2 + 1;
    const T ** srcp = new const T *[diameter];

    srcp[radius] = _srcp;
    for (int i = 1; i <= radius; i++) {
        srcp[radius - i] = srcp[radius + i - 1];
        srcp[radius + i] = srcp[radius] + stride * i;
    }

    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            float sum = 0.f;

            for (int i = 0; i < diameter; i++)
                sum += srcp[i][x] * weights[i];

            buffer[x] = sum;
        }

        gaussianBlurHorizontal(buffer, blur, weights + radius, width, radius);

        for (int i = 0; i < diameter - 1; i++)
            srcp[i] = srcp[i + 1];
        if (y < height - 1 - radius)
            srcp[diameter - 1] += stride;
        else if (y > height - 1 - radius)
            srcp[diameter - 1] -= stride;
        blur += blurStride;
    }

    delete[] srcp;
}

template<>
void gaussianBlurVertical<float>(const float * _srcp, float * VS_RESTRICT buffer, float * VS_RESTRICT blur, const float * weights, const int width, const int height,
                                 const int stride, const int blurStride, const int radius, const float offset) {
    const int diameter = radius * 2 + 1;
    const float ** srcp = new const float *[diameter];

    srcp[radius] = _srcp;
    for (int i = 1; i <= radius; i++) {
        srcp[radius - i] = srcp[radius + i - 1];
        srcp[radius + i] = srcp[radius] + stride * i;
    }

    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            float sum = 0.f;

            for (int i = 0; i < diameter; i++)
                sum += (srcp[i][x] + offset) * weights[i];

            buffer[x] = sum;
        }

        gaussianBlurHorizontal(buffer, blur, weights + radius, width, radius);

        for (int i = 0; i < diameter - 1; i++)
            srcp[i] = srcp[i + 1];
        if (y < height - 1 - radius)
            srcp[diameter - 1] += stride;
        else if (y > height - 1 - radius)
            srcp[diameter - 1] -= stride;
        blur += blurStride;
    }

    delete[] srcp;
}

static void detectEdge(float * blur, float * VS_RESTRICT gradient, float * VS_RESTRICT direction, const int width, const int height,
                       const int stride, const int blurStride, const int mode, const int op) {
    float * VS_RESTRICT srcpp = blur;
    float * VS_RESTRICT srcp = blur;
    float * VS_RESTRICT srcpn = blur + blurStride;

    srcp[-1] = srcp[0];
    srcp[width] = srcp[width - 1];

    for (int y = 0; y < height; y++) {
        srcpn[-1] = srcpn[0];
        srcpn[width] = srcpn[width - 1];

        for (int x = 0; x < width; x++) {
            float gx, gy;

            if (op == 0) {
                gx = srcp[x + 1] - srcp[x - 1];
                gy = srcpp[x] - srcpn[x];
            } else if (op == 1) {
                gx = (srcpp[x + 1] + srcp[x + 1] + srcpn[x + 1] - srcpp[x - 1] - srcp[x - 1] - srcpn[x - 1]) / 2.f;
                gy = (srcpp[x - 1] + srcpp[x] + srcpp[x + 1] - srcpn[x - 1] - srcpn[x] - srcpn[x + 1]) / 2.f;
            } else if (op == 2) {
                gx = srcpp[x + 1] + 2.f * srcp[x + 1] + srcpn[x + 1] - srcpp[x - 1] - 2.f * srcp[x - 1] - srcpn[x - 1];
                gy = srcpp[x - 1] + 2.f * srcpp[x] + srcpp[x + 1] - srcpn[x - 1] - 2.f * srcpn[x] - srcpn[x + 1];
            } else {
                gx = 3.f * srcpp[x + 1] + 10.f * srcp[x + 1] + 3.f * srcpn[x + 1] - 3.f * srcpp[x - 1] - 10.f * srcp[x - 1] - 3.f * srcpn[x - 1];
                gy = 3.f * srcpp[x - 1] + 10.f * srcpp[x] + 3.f * srcpp[x + 1] - 3.f * srcpn[x - 1] - 10.f * srcpn[x] - 3.f * srcpn[x + 1];
            }

            gradient[x] = std::sqrt(gx * gx + gy * gy);

            if (mode != 1) {
                float dr = std::atan2(gy, gx);
                if (dr < 0.f)
                    dr += M_PIF;
                direction[x] = dr;
            }
        }

        srcpp = srcp;
        srcp = srcpn;
        if (y < height - 2)
            srcpn += blurStride;
        gradient += stride;
        direction += stride;
    }
}

static void nonMaximumSuppression(const float * gradient, const float * direction, float * VS_RESTRICT blur, const int width, const int height, const int stride, const int blurStride) {
    const int offsets[] { 1, -stride + 1, -stride, -stride - 1 };

    std::fill_n(blur, width, -FLT_MAX);

    for (int y = 1; y < height - 1; y++) {
        gradient += stride;
        direction += stride;
        blur += blurStride;

        for (int x = 1; x < width - 1; x++) {
            const int offset = offsets[getBin<int>(direction[x], 4)];
            blur[x] = (gradient[x] >= std::max(gradient[x + offset], gradient[x - offset])) ? gradient[x] : -FLT_MAX;
        }

        blur[0] = blur[width - 1] = -FLT_MAX;
    }

    std::fill_n(blur + blurStride, width, -FLT_MAX);
}
#endif

static void hysteresis(float * VS_RESTRICT blur, Stack & VS_RESTRICT stack, const int width, const int height, const int blurStride, const float t_h, const float t_l) {
    memset(stack.map, 0, width * height);
    stack.index = -1;

    for (int y = 1; y < height - 1; y++) {
        for (int x = 1; x < width - 1; x++) {
            if (blur[x + blurStride * y] < t_h || stack.map[x + width * y])
                continue;

            blur[x + blurStride * y] = FLT_MAX;
            stack.map[x + width * y] = UINT8_MAX;
            push(stack, x, y);

            while (stack.index > -1) {
                const std::pair<int, int> pos = pop(stack);
                const int xMin = (pos.first > 1) ? pos.first - 1 : 1;
                const int xMax = (pos.first < width - 2) ? pos.first + 1 : pos.first;
                const int yMin = (pos.second > 1) ? pos.second - 1 : 1;
                const int yMax = (pos.second < height - 2) ? pos.second + 1 : pos.second;

                for (int yy = yMin; yy <= yMax; yy++) {
                    for (int xx = xMin; xx <= xMax; xx++) {
                        if (blur[xx + blurStride * yy] > t_l && !stack.map[xx + width * yy]) {
                            blur[xx + blurStride * yy] = FLT_MAX;
                            stack.map[xx + width * yy] = UINT8_MAX;
                            push(stack, xx, yy);
                        }
                    }
                }
            }
        }
    }
}

template<typename T>
static void outputGB(const float * blur, T * VS_RESTRICT dstp, const int width, const int height, const int stride, const int blurStride,
                     const int peak, const float offset, const float lower, const float upper) {
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++)
            dstp[x] = std::min(std::max(static_cast<int>(blur[x] + 0.5f), 0), peak);

        blur += blurStride;
        dstp += stride;
    }
}

template<>
void outputGB<float>(const float * blur, float * VS_RESTRICT dstp, const int width, const int height, const int stride, const int blurStride,
                     const int peak, const float offset, const float lower, const float upper) {
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++)
            dstp[x] = std::min(std::max(blur[x] - offset, lower), upper);

        blur += blurStride;
        dstp += stride;
    }
}

template<typename T>
static void binarizeCE(const float * blur, T * VS_RESTRICT dstp, const int width, const int height, const int stride, const int blurStride,
                       const float t_h, const T peak, const float lower, const float upper) {
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++)
            dstp[x] = (blur[x] >= t_h) ? peak : 0;

        blur += blurStride;
        dstp += stride;
    }
}

template<>
void binarizeCE<float>(const float * blur, float * VS_RESTRICT dstp, const int width, const int height, const int stride, const int blurStride,
                       const float t_h, const float peak, const float lower, const float upper) {
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++)
            dstp[x] = (blur[x] >= t_h) ? upper : lower;

        blur += blurStride;
        dstp += stride;
    }
}

template<typename T>
static void discretizeGM(const float * gradient, T * VS_RESTRICT dstp, const int width, const int height, const int stride,
                         const float magnitude, const int peak, const float offset, const float upper) {
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++)
            dstp[x] = std::min(static_cast<int>(gradient[x] * magnitude + 0.5f), peak);

        gradient += stride;
        dstp += stride;
    }
}

template<>
void discretizeGM<float>(const float * gradient, float * VS_RESTRICT dstp, const int width, const int height, const int stride,
                         const float magnitude, const int peak, const float offset, const float upper) {
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++)
            dstp[x] = std::min(gradient[x] * magnitude - offset, upper);

        gradient += stride;
        dstp += stride;
    }
}

template<typename T>
static void discretizeDM_T(const float * blur, const float * direction, T * VS_RESTRICT dstp, const int width, const int height, const int stride, const int blurStride,
                           const float t_h, const int bins, const float offset, const float lower) {
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++)
            dstp[x] = (blur[x] >= t_h) ? getBin<T>(direction[x], bins) : 0;

        blur += blurStride;
        direction += stride;
        dstp += stride;
    }
}

template<>
void discretizeDM_T<float>(const float * blur, const float * direction, float * VS_RESTRICT dstp, const int width, const int height, const int stride, const int blurStride,
                           const float t_h, const int bins, const float offset, const float lower) {
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++)
            dstp[x] = (blur[x] >= t_h) ? getBin<float>(direction[x], bins) - offset : lower;

        blur += blurStride;
        direction += stride;
        dstp += stride;
    }
}

template<typename T>
static void discretizeDM(const float * direction, T * VS_RESTRICT dstp, const int width, const int height, const int stride, const int bins, const float offset) {
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++)
            dstp[x] = getBin<T>(direction[x], bins);

        direction += stride;
        dstp += stride;
    }
}

template<>
void discretizeDM<float>(const float * direction, float * VS_RESTRICT dstp, const int width, const int height, const int stride, const int bins, const float offset) {
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++)
            dstp[x] = getBin<float>(direction[x], bins) - offset;

        direction += stride;
        dstp += stride;
    }
}

template<typename T>
static void process(const VSFrameRef * src, VSFrameRef * dst, float * buffer, float * blur, float * gradient, float * direction,
                    Stack & stack, const TCannyData * d, const VSAPI * vsapi) {
    for (int plane = 0; plane < d->vi->format->numPlanes; plane++) {
        if (d->process[plane]) {
            const int width = vsapi->getFrameWidth(src, plane);
            const int height = vsapi->getFrameHeight(src, plane);
            const int stride = vsapi->getStride(src, plane) / sizeof(T);
            const int blurStride = stride + 16;
            const uint8_t * srcp = vsapi->getReadPtr(src, plane);
            T * dstp = reinterpret_cast<T *>(vsapi->getWritePtr(dst, plane));
            const float offset = (d->vi->format->sampleType == stInteger || plane == 0 || d->vi->format->colorFamily == cmRGB) ? 0.f : 0.5f;

#ifdef VS_TARGET_CPU_X86
            d->gaussianBlurVertical(srcp, buffer, blur, d->weights, width, height, stride, blurStride, d->radius, offset);
#else
            gaussianBlurVertical<T>(reinterpret_cast<const T *>(srcp), buffer, blur, d->weights, width, height, stride, blurStride, d->radius, offset);
#endif

            if (d->mode != -1)
                detectEdge(blur, gradient, direction, width, height, stride, blurStride, d->mode, d->op);

            if (!(d->mode & 1)) {
                nonMaximumSuppression(gradient, direction, blur, width, height, stride, blurStride);
                hysteresis(blur, stack, width, height, blurStride, d->t_h, d->t_l);
            }

            if (d->mode == -1)
                outputGB<T>(blur, dstp, width, height, stride, blurStride, d->peak, offset, d->lower[plane], d->upper[plane]);
            else if (d->mode == 0)
                binarizeCE<T>(blur, dstp, width, height, stride, blurStride, d->t_h, d->peak, d->lower[plane], d->upper[plane]);
            else if (d->mode == 1)
                discretizeGM<T>(gradient, dstp, width, height, stride, d->magnitude, d->peak, offset, d->upper[plane]);
            else if (d->mode == 2)
                discretizeDM_T<T>(blur, direction, dstp, width, height, stride, blurStride, d->t_h, d->bins, offset, d->lower[plane]);
            else
                discretizeDM<T>(direction, dstp, width, height, stride, d->bins, offset);
        }
    }
}

static void VS_CC tcannyInit(VSMap *in, VSMap *out, void **instanceData, VSNode *node, VSCore *core, const VSAPI *vsapi) {
    TCannyData * d = static_cast<TCannyData *>(*instanceData);
    vsapi->setVideoInfo(d->vi, 1, node);
}

static const VSFrameRef *VS_CC tcannyGetFrame(int n, int activationReason, void **instanceData, void **frameData, VSFrameContext *frameCtx, VSCore *core, const VSAPI *vsapi) {
    const TCannyData * d = static_cast<const TCannyData *>(*instanceData);

    if (activationReason == arInitial) {
        vsapi->requestFrameFilter(n, d->node, frameCtx);
    } else if (activationReason == arAllFramesReady) {
#ifdef VS_TARGET_CPU_X86
        no_subnormals();
#endif

        const VSFrameRef * src = vsapi->getFrameFilter(n, d->node, frameCtx);
        const VSFrameRef * fr[] { d->process[0] ? nullptr : src, d->process[1] ? nullptr : src, d->process[2] ? nullptr : src };
        const int pl[] { 0, 1, 2 };
        VSFrameRef * dst = vsapi->newVideoFrame2(d->vi->format, d->vi->width, d->vi->height, fr, pl, src, core);

        float * buffer = vs_aligned_malloc<float>((d->vi->width + d->radiusAlign * 2) * sizeof(float), 32);
        if (!buffer) {
            vsapi->setFilterError("TCanny: malloc failure (buffer)", frameCtx);
            vsapi->freeFrame(src);
            vsapi->freeFrame(dst);
            return nullptr;
        }

        float * blur;
        blur = vs_aligned_malloc<float>((vsapi->getStride(src, 0) / d->vi->format->bytesPerSample + 16) * d->vi->height * sizeof(float), 32);
        if (!blur) {
            vsapi->setFilterError("TCanny: malloc failure (blur)", frameCtx);
            vsapi->freeFrame(src);
            vsapi->freeFrame(dst);
            return nullptr;
        }

        float * gradient = nullptr, * direction = nullptr;
        if (d->mode != -1) {
            gradient = vs_aligned_malloc<float>(vsapi->getStride(src, 0) / d->vi->format->bytesPerSample * (d->vi->height + 1) * sizeof(float), 32);
            if (!gradient) {
                vsapi->setFilterError("TCanny: malloc failure (gradient)", frameCtx);
                vsapi->freeFrame(src);
                vsapi->freeFrame(dst);
                return nullptr;
            }

            if (d->mode != 1) {
                direction = vs_aligned_malloc<float>(vsapi->getStride(src, 0) / d->vi->format->bytesPerSample * (d->vi->height + 1) * sizeof(float), 32);
                if (!direction) {
                    vsapi->setFilterError("TCanny: malloc failure (direction)", frameCtx);
                    vsapi->freeFrame(src);
                    vsapi->freeFrame(dst);
                    return nullptr;
                }
            }
        }

        Stack stack {};
        if (!(d->mode & 1)) {
            stack.map = vs_aligned_malloc<uint8_t>(d->vi->width * d->vi->height, 32);
            stack.pos = vs_aligned_malloc<std::pair<int, int>>(d->vi->width * d->vi->height * sizeof(std::pair<int, int>), 32);
            if (!stack.map || !stack.pos) {
                vsapi->setFilterError("TCanny: malloc failure (stack)", frameCtx);
                vsapi->freeFrame(src);
                vsapi->freeFrame(dst);
                return nullptr;
            }
        }

        if (d->vi->format->sampleType == stInteger) {
            if (d->vi->format->bitsPerSample == 8)
                process<uint8_t>(src, dst, buffer + d->radiusAlign, blur + 8, gradient, direction, stack, d, vsapi);
            else
                process<uint16_t>(src, dst, buffer + d->radiusAlign, blur + 8, gradient, direction, stack, d, vsapi);
        } else {
            process<float>(src, dst, buffer + d->radiusAlign, blur + 8, gradient, direction, stack, d, vsapi);
        }

        vsapi->freeFrame(src);
        vs_aligned_free(buffer);
        vs_aligned_free(blur);
        vs_aligned_free(gradient);
        vs_aligned_free(direction);
        vs_aligned_free(stack.map);
        vs_aligned_free(stack.pos);
        return dst;
    }

    return nullptr;
}

static void VS_CC tcannyFree(void *instanceData, VSCore *core, const VSAPI *vsapi) {
    TCannyData * d = static_cast<TCannyData *>(instanceData);

    vsapi->freeNode(d->node);

    delete[] d->weights;
    delete d;
}

static void VS_CC tcannyCreate(const VSMap *in, VSMap *out, void *userData, VSCore *core, const VSAPI *vsapi) {
    TCannyData d;
    int err;

    d.sigma = static_cast<float>(vsapi->propGetFloat(in, "sigma", 0, &err));
    if (err)
        d.sigma = 1.5f;

    d.t_h = static_cast<float>(vsapi->propGetFloat(in, "t_h", 0, &err));
    if (err)
        d.t_h = 8.f;

    d.t_l = static_cast<float>(vsapi->propGetFloat(in, "t_l", 0, &err));
    if (err)
        d.t_l = 1.f;

    d.mode = int64ToIntS(vsapi->propGetInt(in, "mode", 0, &err));

    d.op = int64ToIntS(vsapi->propGetInt(in, "op", 0, &err));
    if (err)
        d.op = 1;

    d.gmmax = static_cast<float>(vsapi->propGetFloat(in, "gmmax", 0, &err));
    if (err)
        d.gmmax = 50.f;

    if (d.sigma <= 0.f) {
        vsapi->setError(out, "TCanny: sigma must be greater than 0.0");
        return;
    }

    if (d.mode < -1 || d.mode > 3) {
        vsapi->setError(out, "TCanny: mode must be -1, 0, 1, 2 or 3");
        return;
    }

    if (d.op < 0 || d.op > 3) {
        vsapi->setError(out, "TCanny: op must be 0, 1, 2 or 3");
        return;
    }

    if (d.gmmax < 1.f) {
        vsapi->setError(out, "TCanny: gmmax must be greater than or equal to 1.0");
        return;
    }

    d.node = vsapi->propGetNode(in, "clip", 0, nullptr);
    d.vi = vsapi->getVideoInfo(d.node);

    if (!isConstantFormat(d.vi) || (d.vi->format->sampleType == stInteger && d.vi->format->bitsPerSample > 16) ||
        (d.vi->format->sampleType == stFloat && d.vi->format->bitsPerSample != 32)) {
        vsapi->setError(out, "TCanny: only constant format 8-16 bits integer and 32 bits float input supported");
        vsapi->freeNode(d.node);
        return;
    }

    const int m = vsapi->propNumElements(in, "planes");

    for (int i = 0; i < 3; i++)
        d.process[i] = m <= 0;

    for (int i = 0; i < m; i++) {
        const int n = int64ToIntS(vsapi->propGetInt(in, "planes", i, nullptr));

        if (n < 0 || n >= d.vi->format->numPlanes) {
            vsapi->setError(out, "TCanny: plane index out of range");
            vsapi->freeNode(d.node);
            return;
        }

        if (d.process[n]) {
            vsapi->setError(out, "TCanny: plane specified twice");
            vsapi->freeNode(d.node);
            return;
        }

        d.process[n] = true;
    }

    if (d.vi->format->sampleType == stInteger) {
        d.t_h = scale(d.t_h, d.vi->format->bitsPerSample);
        d.t_l = scale(d.t_l, d.vi->format->bitsPerSample);
        d.bins = 1 << d.vi->format->bitsPerSample;
        d.peak = d.bins - 1;

#ifdef VS_TARGET_CPU_X86
        if (d.vi->format->bitsPerSample == 8)
            d.gaussianBlurVertical = gaussianBlurVertical_uint8;
        else
            d.gaussianBlurVertical = gaussianBlurVertical_uint16;
#endif
    } else {
        d.t_h /= 255.f;
        d.t_l /= 255.f;
        d.bins = 1;

        for (int plane = 0; plane < d.vi->format->numPlanes; plane++) {
            if (d.process[plane]) {
                if (plane == 0 || d.vi->format->colorFamily == cmRGB) {
                    d.lower[plane] = 0.f;
                    d.upper[plane] = 1.f;
                } else {
                    d.lower[plane] = -0.5f;
                    d.upper[plane] = 0.5f;
                }
            }
        }

#ifdef VS_TARGET_CPU_X86
        d.gaussianBlurVertical = gaussianBlurVertical_float;
#endif
    }

    d.weights = gaussianWeights(d.sigma, &d.radius);
    if (!d.weights) {
        vsapi->setError(out, "TCanny: malloc failure (weights)");
        vsapi->freeNode(d.node);
        return;
    }
    d.radiusAlign = (d.radius + 7) & -8;

    d.magnitude = 255.f / d.gmmax;

    TCannyData * data = new TCannyData { d };

    vsapi->createFilter(in, out, "TCanny", tcannyInit, tcannyGetFrame, tcannyFree, fmParallel, 0, data, core);
}

//////////////////////////////////////////
// Init

VS_EXTERNAL_API(void) VapourSynthPluginInit(VSConfigPlugin configFunc, VSRegisterFunction registerFunc, VSPlugin *plugin) {
    configFunc("com.holywu.tcanny", "tcanny", "Build an edge map using canny edge detection", VAPOURSYNTH_API_VERSION, 1, plugin);
    registerFunc("TCanny",
                 "clip:clip;"
                 "sigma:float:opt;"
                 "t_h:float:opt;"
                 "t_l:float:opt;"
                 "mode:int:opt;"
                 "op:int:opt;"
                 "gmmax:float:opt;"
                 "planes:int[]:opt;",
                 tcannyCreate, nullptr, plugin);
}
