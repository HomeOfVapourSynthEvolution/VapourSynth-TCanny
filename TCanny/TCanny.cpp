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
#include <cmath>
#include <limits>
#include <thread>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>

#include <VapourSynth.h>
#include <VSHelper.h>

#ifdef VS_TARGET_CPU_X86
#include "vectorclass/vectorclass.h"

extern void gaussianBlurHorizontal_SSE2(float *, float *, const float *, const int, const int);
extern void gaussianBlurHorizontal_AVX(float *, float *, const float *, const int, const int);
extern void gaussianBlurHorizontal_AVX2(float *, float *, const float *, const int, const int);

template<typename T> extern void gaussianBlurVertical_SSE2(const T *, float *, float *, const float *, const unsigned, const int, const unsigned, const unsigned, const int, const float);
template<typename T> extern void gaussianBlurVertical_AVX(const T *, float *, float *, const float *, const unsigned, const int, const unsigned, const unsigned, const int, const float);
template<typename T> extern void gaussianBlurVertical_AVX2(const T *, float *, float *, const float *, const unsigned, const int, const unsigned, const unsigned, const int, const float);

extern void detectEdge_SSE2(float *, float *, float *, const int, const unsigned, const unsigned, const unsigned, const int, const unsigned);
extern void detectEdge_AVX(float *, float *, float *, const int, const unsigned, const unsigned, const unsigned, const int, const unsigned);
extern void detectEdge_AVX2(float *, float *, float *, const int, const unsigned, const unsigned, const unsigned, const int, const unsigned);

extern void nonMaximumSuppression_SSE2(const float *, const float *, float *, const int, const unsigned, const int, const unsigned);
extern void nonMaximumSuppression_AVX(const float *, const float *, float *, const int, const unsigned, const int, const unsigned);
extern void nonMaximumSuppression_AVX2(const float *, const float *, float *, const int, const unsigned, const int, const unsigned);

template<typename T> extern void outputGB_SSE2(const float *, T *, const unsigned, const unsigned, const unsigned, const unsigned, const uint16_t, const float, const float);
template<typename T> extern void outputGB_AVX(const float *, T *, const unsigned, const unsigned, const unsigned, const unsigned, const uint16_t, const float, const float);
template<typename T> extern void outputGB_AVX2(const float *, T *, const unsigned, const unsigned, const unsigned, const unsigned, const uint16_t, const float, const float);

template<typename T> extern void binarizeCE_SSE2(const float *, T *, const unsigned, const unsigned, const unsigned, const unsigned, const uint16_t, const float, const float);
template<typename T> extern void binarizeCE_AVX(const float *, T *, const unsigned, const unsigned, const unsigned, const unsigned, const uint16_t, const float, const float);
template<typename T> extern void binarizeCE_AVX2(const float *, T *, const unsigned, const unsigned, const unsigned, const unsigned, const uint16_t, const float, const float);

template<typename T> extern void discretizeGM_SSE2(const float *, T *, const unsigned, const unsigned, const unsigned, const float, const uint16_t, const float, const float);
template<typename T> extern void discretizeGM_AVX(const float *, T *, const unsigned, const unsigned, const unsigned, const float, const uint16_t, const float, const float);
template<typename T> extern void discretizeGM_AVX2(const float *, T *, const unsigned, const unsigned, const unsigned, const float, const uint16_t, const float, const float);
#endif

static constexpr float M_PIF = 3.14159265358979323846f;
static constexpr float M_1_PIF = 0.318309886183790671538f;

static void (*gaussianBlurHorizontal)(float *, float *, const float *, const int, const int);
template<typename T> void (*gaussianBlurVertical)(const T *, float *, float *, const float *, const unsigned, const int, const unsigned, const unsigned, const int, const float);
static void (*detectEdge)(float *, float *, float *, const int, const unsigned, const unsigned, const unsigned, const int, const unsigned);
static void (*nonMaximumSuppression)(const float *, const float *, float *, const int, const unsigned, const int, const unsigned);
template<typename T> void (*outputGB)(const float *, T *, const unsigned, const unsigned, const unsigned, const unsigned, const uint16_t, const float, const float);
template<typename T> void (*binarizeCE)(const float *, T *, const unsigned, const unsigned, const unsigned, const unsigned, const uint16_t, const float, const float);
template<typename T> void (*discretizeGM)(const float *, T *, const unsigned, const unsigned, const unsigned, const float, const uint16_t, const float, const float);

struct TCannyData {
    VSNodeRef * node;
    const VSVideoInfo * vi;
    float sigma, t_h, t_l, gmmax;
    int mode, op;
    bool process[3];
    int radius;
    float * weights;
    float magnitude;
    unsigned radiusAlign, bins;
    uint16_t peak;
    float offset[3], lower[3], upper[3];
    std::unordered_map<std::thread::id, float *> buffer, blur, gradient, direction;
    std::unordered_map<std::thread::id, uint8_t *> label;
};

static float * gaussianWeights(const float sigma, int * radius) noexcept {
    const unsigned diameter = std::max(static_cast<int>(sigma * 3.f + 0.5f), 1) * 2 + 1;
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

template<typename T>
static inline T getBin(const float dir, const unsigned n) noexcept {
    if (!std::is_same<T, float>::value) {
        const unsigned bin = static_cast<unsigned>(dir * n * M_1_PIF + 0.5f);
        return (bin >= n) ? 0 : bin;
    } else {
        const float bin = dir * M_1_PIF;
        return (bin > n) ? 0.f : bin;
    }
}

static void gaussianBlurHorizontal_C(float * VS_RESTRICT buffer, float * VS_RESTRICT blur, const float * weights, const int width, const int radius) noexcept {
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
static void gaussianBlurVertical_C(const T * _srcp, float * VS_RESTRICT buffer, float * VS_RESTRICT blur, const float * weights, const unsigned width, const int height,
                                   const unsigned stride, const unsigned blurStride, const int radius, const float offset) noexcept {
    const unsigned diameter = radius * 2 + 1;
    const T ** srcp = new const T *[diameter];

    srcp[radius] = _srcp;
    for (int i = 1; i <= radius; i++) {
        srcp[radius - i] = srcp[radius + i - 1];
        srcp[radius + i] = srcp[radius] + stride * i;
    }

    for (int y = 0; y < height; y++) {
        for (unsigned x = 0; x < width; x++) {
            float sum = 0.f;

            for (unsigned i = 0; i < diameter; i++) {
                if (!std::is_same<T, float>::value)
                    sum += srcp[i][x] * weights[i];
                else
                    sum += (srcp[i][x] + offset) * weights[i];
            }

            buffer[x] = sum;
        }

        gaussianBlurHorizontal_C(buffer, blur, weights + radius, width, radius);

        for (unsigned i = 0; i < diameter - 1; i++)
            srcp[i] = srcp[i + 1];
        if (y < height - 1 - radius)
            srcp[diameter - 1] += stride;
        else if (y > height - 1 - radius)
            srcp[diameter - 1] -= stride;
        blur += blurStride;
    }

    delete[] srcp;
}

static void detectEdge_C(float * blur, float * VS_RESTRICT gradient, float * VS_RESTRICT direction, const int width, const unsigned height,
                         const unsigned stride, const unsigned blurStride, const int mode, const unsigned op) noexcept {
    float * VS_RESTRICT srcpp = blur;
    float * VS_RESTRICT srcp = blur;
    float * VS_RESTRICT srcpn = blur + blurStride;

    srcp[-1] = srcp[0];
    srcp[width] = srcp[width - 1];

    for (unsigned y = 0; y < height; y++) {
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

static void nonMaximumSuppression_C(const float * gradient, const float * direction, float * VS_RESTRICT blur, const int width, const unsigned height,
                                    const int stride, const unsigned blurStride) noexcept {
    const int offsets[] { 1, -stride + 1, -stride, -stride - 1 };

    std::fill_n(blur, width, std::numeric_limits<float>::lowest());

    for (unsigned y = 1; y < height - 1; y++) {
        gradient += stride;
        direction += stride;
        blur += blurStride;

        for (int x = 1; x < width - 1; x++) {
            const int offset = offsets[getBin<unsigned>(direction[x], 4)];
            blur[x] = (gradient[x] >= std::max(gradient[x + offset], gradient[x - offset])) ? gradient[x] : std::numeric_limits<float>::lowest();
        }

        blur[0] = blur[width - 1] = std::numeric_limits<float>::lowest();
    }

    std::fill_n(blur + blurStride, width, std::numeric_limits<float>::lowest());
}

static void hysteresis(float * VS_RESTRICT blur, uint8_t * VS_RESTRICT label, const unsigned width, const unsigned height, const unsigned blurStride,
                       const float t_h, const float t_l) noexcept {
    std::fill_n(label, width * height, static_cast<uint8_t>(0));

    std::vector<std::pair<unsigned, unsigned>> coordinates;

    for (unsigned y = 1; y < height - 1; y++) {
        for (unsigned x = 1; x < width - 1; x++) {
            if (!label[width * y + x] && blur[blurStride * y + x] >= t_h) {
                label[width * y + x] = std::numeric_limits<uint8_t>::max();
                blur[blurStride * y + x] = std::numeric_limits<float>::max();

                coordinates.emplace_back(std::make_pair(x, y));

                while (!coordinates.empty()) {
                    const auto pos = coordinates.back();
                    coordinates.pop_back();

                    for (unsigned yy = pos.second - 1; yy <= pos.second + 1; yy++) {
                        for (unsigned xx = pos.first - 1; xx <= pos.first + 1; xx++) {
                            if (!label[width * yy + xx] && blur[blurStride * yy + xx] >= t_l) {
                                label[width * yy + xx] = std::numeric_limits<uint8_t>::max();
                                blur[blurStride * yy + xx] = std::numeric_limits<float>::max();

                                coordinates.emplace_back(std::make_pair(xx, yy));
                            }
                        }
                    }
                }
            }
        }
    }
}

template<typename T>
static void outputGB_C(const float * blur, T * VS_RESTRICT dstp, const unsigned width, const unsigned height, const unsigned stride, const unsigned blurStride,
                       const uint16_t peak, const float offset, const float upper) noexcept {
    for (unsigned y = 0; y < height; y++) {
        for (unsigned x = 0; x < width; x++) {
            if (!std::is_same<T, float>::value)
                dstp[x] = std::min<unsigned>(static_cast<unsigned>(blur[x] + 0.5f), peak);
            else
                dstp[x] = std::min(blur[x] - offset, upper);
        }

        blur += blurStride;
        dstp += stride;
    }
}

template<typename T>
static void binarizeCE_C(const float * blur, T * VS_RESTRICT dstp, const unsigned width, const unsigned height, const unsigned stride, const unsigned blurStride,
                         const uint16_t peak, const float lower, const float upper) noexcept {
    for (unsigned y = 0; y < height; y++) {
        for (unsigned x = 0; x < width; x++) {
            if (!std::is_same<T, float>::value)
                dstp[x] = (blur[x] == std::numeric_limits<float>::max()) ? peak : 0;
            else
                dstp[x] = (blur[x] == std::numeric_limits<float>::max()) ? upper : lower;
        }

        blur += blurStride;
        dstp += stride;
    }
}

template<typename T>
static void discretizeGM_C(const float * gradient, T * VS_RESTRICT dstp, const unsigned width, const unsigned height, const unsigned stride, const float magnitude,
                           const uint16_t peak, const float offset, const float upper) noexcept {
    for (unsigned y = 0; y < height; y++) {
        for (unsigned x = 0; x < width; x++) {
            if (!std::is_same<T, float>::value)
                dstp[x] = std::min<unsigned>(static_cast<unsigned>(gradient[x] * magnitude + 0.5f), peak);
            else
                dstp[x] = std::min(gradient[x] * magnitude - offset, upper);
        }

        gradient += stride;
        dstp += stride;
    }
}

template<typename T>
static void discretizeDM_T(const float * blur, const float * direction, T * VS_RESTRICT dstp, const unsigned width, const unsigned height,
                           const unsigned stride, const unsigned blurStride, const unsigned bins, const float offset, const float lower) noexcept {
    for (unsigned y = 0; y < height; y++) {
        for (unsigned x = 0; x < width; x++) {
            if (!std::is_same<T, float>::value)
                dstp[x] = (blur[x] == std::numeric_limits<float>::max()) ? getBin<T>(direction[x], bins) : 0;
            else
                dstp[x] = (blur[x] == std::numeric_limits<float>::max()) ? getBin<float>(direction[x], bins) - offset : lower;
        }

        blur += blurStride;
        direction += stride;
        dstp += stride;
    }
}

template<typename T>
static void discretizeDM(const float * direction, T * VS_RESTRICT dstp, const unsigned width, const unsigned height, const unsigned stride,
                         const unsigned bins, const float offset) noexcept {
    for (unsigned y = 0; y < height; y++) {
        for (unsigned x = 0; x < width; x++) {
            if (!std::is_same<T, float>::value)
                dstp[x] = getBin<T>(direction[x], bins);
            else
                dstp[x] = getBin<float>(direction[x], bins) - offset;
        }

        direction += stride;
        dstp += stride;
    }
}

template<typename T>
static void process(const VSFrameRef * src, VSFrameRef * dst, const TCannyData * d, const VSAPI * vsapi) noexcept {
    for (int plane = 0; plane < d->vi->format->numPlanes; plane++) {
        if (d->process[plane]) {
            const unsigned width = vsapi->getFrameWidth(src, plane);
            const unsigned height = vsapi->getFrameHeight(src, plane);
            const unsigned stride = vsapi->getStride(src, plane) / sizeof(T);
            const unsigned blurStride = stride + 16;
            const T * srcp = reinterpret_cast<const T *>(vsapi->getReadPtr(src, plane));
            T * dstp = reinterpret_cast<T *>(vsapi->getWritePtr(dst, plane));

            const auto threadId = std::this_thread::get_id();
            float * buffer = d->buffer.at(threadId) + d->radiusAlign;
            float * blur = d->blur.at(threadId) + 8;
            float * gradient = d->gradient.at(threadId);
            float * direction = d->direction.at(threadId);
            uint8_t * label = d->label.at(threadId);

            gaussianBlurVertical<T>(srcp, buffer, blur, d->weights, width, height, stride, blurStride, d->radius, d->offset[plane]);

            if (d->mode != -1)
                detectEdge(blur, gradient, direction, width, height, stride, blurStride, d->mode, d->op);

            if (!(d->mode & 1)) {
                nonMaximumSuppression(gradient, direction, blur, width, height, stride, blurStride);
                hysteresis(blur, label, width, height, blurStride, d->t_h, d->t_l);
            }

            if (d->mode == -1)
                outputGB<T>(blur, dstp, width, height, stride, blurStride, d->peak, d->offset[plane], d->upper[plane]);
            else if (d->mode == 0)
                binarizeCE<T>(blur, dstp, width, height, stride, blurStride, d->peak, d->lower[plane], d->upper[plane]);
            else if (d->mode == 1)
                discretizeGM<T>(gradient, dstp, width, height, stride, d->magnitude, d->peak, d->offset[plane], d->upper[plane]);
            else if (d->mode == 2)
                discretizeDM_T(blur, direction, dstp, width, height, stride, blurStride, d->bins, d->offset[plane], d->lower[plane]);
            else
                discretizeDM(direction, dstp, width, height, stride, d->bins, d->offset[plane]);
        }
    }
}

static void selectFunctions(const unsigned opt) noexcept {
    gaussianBlurHorizontal = gaussianBlurHorizontal_C;

    gaussianBlurVertical<uint8_t> = gaussianBlurVertical_C;
    gaussianBlurVertical<uint16_t> = gaussianBlurVertical_C;
    gaussianBlurVertical<float> = gaussianBlurVertical_C;

    detectEdge = detectEdge_C;

    nonMaximumSuppression = nonMaximumSuppression_C;

    outputGB<uint8_t> = outputGB_C;
    outputGB<uint16_t> = outputGB_C;
    outputGB<float> = outputGB_C;

    binarizeCE<uint8_t> = binarizeCE_C;
    binarizeCE<uint16_t> = binarizeCE_C;
    binarizeCE<float> = binarizeCE_C;

    discretizeGM<uint8_t> = discretizeGM_C;
    discretizeGM<uint16_t> = discretizeGM_C;
    discretizeGM<float> = discretizeGM_C;

#ifdef VS_TARGET_CPU_X86
    const int iset = instrset_detect();
    if (opt == 4 || (opt == 0 && iset >= 8)) {
        gaussianBlurHorizontal = gaussianBlurHorizontal_AVX2;

        gaussianBlurVertical<uint8_t> = gaussianBlurVertical_AVX2;
        gaussianBlurVertical<uint16_t> = gaussianBlurVertical_AVX2;
        gaussianBlurVertical<float> = gaussianBlurVertical_AVX2;

        detectEdge = detectEdge_AVX2;

        nonMaximumSuppression = nonMaximumSuppression_AVX2;

        outputGB<uint8_t> = outputGB_AVX2;
        outputGB<uint16_t> = outputGB_AVX2;
        outputGB<float> = outputGB_AVX2;

        binarizeCE<uint8_t> = binarizeCE_AVX2;
        binarizeCE<uint16_t> = binarizeCE_AVX2;
        binarizeCE<float> = binarizeCE_AVX2;

        discretizeGM<uint8_t> = discretizeGM_AVX2;
        discretizeGM<uint16_t> = discretizeGM_AVX2;
        discretizeGM<float> = discretizeGM_AVX2;
    } else if (opt == 3 || (opt == 0 && iset == 7)) {
        gaussianBlurHorizontal = gaussianBlurHorizontal_AVX;

        gaussianBlurVertical<uint8_t> = gaussianBlurVertical_AVX;
        gaussianBlurVertical<uint16_t> = gaussianBlurVertical_AVX;
        gaussianBlurVertical<float> = gaussianBlurVertical_AVX;

        detectEdge = detectEdge_AVX;

        nonMaximumSuppression = nonMaximumSuppression_AVX;

        outputGB<uint8_t> = outputGB_AVX;
        outputGB<uint16_t> = outputGB_AVX;
        outputGB<float> = outputGB_AVX;

        binarizeCE<uint8_t> = binarizeCE_AVX;
        binarizeCE<uint16_t> = binarizeCE_AVX;
        binarizeCE<float> = binarizeCE_AVX;

        discretizeGM<uint8_t> = discretizeGM_AVX;
        discretizeGM<uint16_t> = discretizeGM_AVX;
        discretizeGM<float> = discretizeGM_AVX;
    } else if (opt == 2 || (opt == 0 && iset >= 2)) {
        gaussianBlurHorizontal = gaussianBlurHorizontal_SSE2;

        gaussianBlurVertical<uint8_t> = gaussianBlurVertical_SSE2;
        gaussianBlurVertical<uint16_t> = gaussianBlurVertical_SSE2;
        gaussianBlurVertical<float> = gaussianBlurVertical_SSE2;

        detectEdge = detectEdge_SSE2;

        nonMaximumSuppression = nonMaximumSuppression_SSE2;

        outputGB<uint8_t> = outputGB_SSE2;
        outputGB<uint16_t> = outputGB_SSE2;
        outputGB<float> = outputGB_SSE2;

        binarizeCE<uint8_t> = binarizeCE_SSE2;
        binarizeCE<uint16_t> = binarizeCE_SSE2;
        binarizeCE<float> = binarizeCE_SSE2;

        discretizeGM<uint8_t> = discretizeGM_SSE2;
        discretizeGM<uint16_t> = discretizeGM_SSE2;
        discretizeGM<float> = discretizeGM_SSE2;
    }
#endif
}

static void VS_CC tcannyInit(VSMap *in, VSMap *out, void **instanceData, VSNode *node, VSCore *core, const VSAPI *vsapi) {
    TCannyData * d = static_cast<TCannyData *>(*instanceData);
    vsapi->setVideoInfo(d->vi, 1, node);
}

static const VSFrameRef *VS_CC tcannyGetFrame(int n, int activationReason, void **instanceData, void **frameData, VSFrameContext *frameCtx, VSCore *core, const VSAPI *vsapi) {
    TCannyData * d = static_cast<TCannyData *>(*instanceData);

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

        auto threadId = std::this_thread::get_id();

        if (!d->buffer.count(threadId)) {
            float * buffer = vs_aligned_malloc<float>((d->vi->width + d->radiusAlign * 2) * sizeof(float), 32);
            if (!buffer) {
                vsapi->setFilterError("TCanny: malloc failure (buffer)", frameCtx);
                vsapi->freeFrame(src);
                vsapi->freeFrame(dst);
                return nullptr;
            }
            d->buffer.emplace(threadId, buffer);
        }

        if (!d->blur.count(threadId)) {
            float * blur = vs_aligned_malloc<float>((vsapi->getStride(src, 0) / d->vi->format->bytesPerSample + 16) * d->vi->height * sizeof(float), 32);
            if (!blur) {
                vsapi->setFilterError("TCanny: malloc failure (blur)", frameCtx);
                vsapi->freeFrame(src);
                vsapi->freeFrame(dst);
                return nullptr;
            }
            d->blur.emplace(threadId, blur);
        }

        if (!d->gradient.count(threadId)) {
            if (d->mode != -1) {
                float * gradient = vs_aligned_malloc<float>(vsapi->getStride(src, 0) / d->vi->format->bytesPerSample * (d->vi->height + 1) * sizeof(float), 32);
                if (!gradient) {
                    vsapi->setFilterError("TCanny: malloc failure (gradient)", frameCtx);
                    vsapi->freeFrame(src);
                    vsapi->freeFrame(dst);
                    return nullptr;
                }
                d->gradient.emplace(threadId, gradient);
            } else {
                d->gradient.emplace(threadId, nullptr);
            }

            if (!d->direction.count(threadId)) {
                if (d->mode != 1) {
                    float * direction = vs_aligned_malloc<float>(vsapi->getStride(src, 0) / d->vi->format->bytesPerSample * (d->vi->height + 1) * sizeof(float), 32);
                    if (!direction) {
                        vsapi->setFilterError("TCanny: malloc failure (direction)", frameCtx);
                        vsapi->freeFrame(src);
                        vsapi->freeFrame(dst);
                        return nullptr;
                    }
                    d->direction.emplace(threadId, direction);
                } else {
                    d->direction.emplace(threadId, nullptr);
                }
            }
        }

        if (!d->label.count(threadId)) {
            if (!(d->mode & 1)) {
                uint8_t * label = new (std::nothrow) uint8_t[d->vi->width * d->vi->height];
                if (!label) {
                    vsapi->setFilterError("TCanny: malloc failure (label)", frameCtx);
                    vsapi->freeFrame(src);
                    vsapi->freeFrame(dst);
                    return nullptr;
                }
                d->label.emplace(threadId, label);
            } else {
                d->label.emplace(threadId, nullptr);
            }
        }

        if (d->vi->format->bytesPerSample == 1)
            process<uint8_t>(src, dst, d, vsapi);
        else if (d->vi->format->bytesPerSample == 2)
            process<uint16_t>(src, dst, d, vsapi);
        else
            process<float>(src, dst, d, vsapi);

        vsapi->freeFrame(src);
        return dst;
    }

    return nullptr;
}

static void VS_CC tcannyFree(void *instanceData, VSCore *core, const VSAPI *vsapi) {
    TCannyData * d = static_cast<TCannyData *>(instanceData);

    vsapi->freeNode(d->node);

    delete[] d->weights;

    for (auto & element : d->buffer)
        vs_aligned_free(element.second);

    for (auto & element : d->blur)
        vs_aligned_free(element.second);

    for (auto & element : d->gradient)
        vs_aligned_free(element.second);

    for (auto & element : d->direction)
        vs_aligned_free(element.second);

    for (auto & element : d->label)
        delete[] element.second;

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

    const int opt = int64ToIntS(vsapi->propGetInt(in, "opt", 0, &err));

    if (d.sigma <= 0.f) {
        vsapi->setError(out, "TCanny: sigma must be greater than 0.0");
        return;
    }

    if (d.t_l >= d.t_h) {
        vsapi->setError(out, "TCanny: t_h must be greater than t_l");
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

    if (opt < 0 || opt > 4) {
        vsapi->setError(out, "TCanny: opt must be 0, 1, 2, 3 or 4");
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

    if (d.vi->height < 2) {
        vsapi->setError(out, "TCanny: the clip's height must be greater than or equal to 2");
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
        d.bins = 1 << d.vi->format->bitsPerSample;
        d.peak = d.bins - 1;
        const float scale = d.peak / 255.f;
        d.t_h *= scale;
        d.t_l *= scale;
    } else {
        d.t_h /= 255.f;
        d.t_l /= 255.f;
        d.bins = 1;

        for (int plane = 0; plane < d.vi->format->numPlanes; plane++) {
            if (plane == 0 || d.vi->format->colorFamily == cmRGB) {
                d.offset[plane] = 0.f;
                d.lower[plane] = 0.f;
                d.upper[plane] = 1.f;
            } else {
                d.offset[plane] = 0.5f;
                d.lower[plane] = -0.5f;
                d.upper[plane] = 0.5f;
            }
        }
    }

    d.weights = gaussianWeights(d.sigma, &d.radius);
    if (!d.weights) {
        vsapi->setError(out, "TCanny: malloc failure (weights)");
        vsapi->freeNode(d.node);
        return;
    }
    d.radiusAlign = (d.radius + 7) & -8;

    d.magnitude = 255.f / d.gmmax;

    const int numThreads = vsapi->getCoreInfo(core)->numThreads;
    d.buffer.reserve(numThreads);
    d.blur.reserve(numThreads);
    d.gradient.reserve(numThreads);
    d.direction.reserve(numThreads);
    d.label.reserve(numThreads);

    selectFunctions(opt);

    TCannyData * data = new TCannyData { std::move(d) };

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
                 "opt:int:opt;"
                 "planes:int[]:opt;",
                 tcannyCreate, nullptr, plugin);
}
