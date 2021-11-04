/*
**   VapourSynth port by HolyWu
**
**                 tcanny v1.0 for Avisynth 2.5.x
**
**   Copyright (C) 2009 Kevin Stone
**
**   This program is free software: you can redistribute it and/or modify
**   it under the terms of the GNU General Public License as published by
**   the Free Software Foundation, either version 3 of the License, or
**   (at your option) any later version.
**
**   This program is distributed in the hope that it will be useful,
**   but WITHOUT ANY WARRANTY; without even the implied warranty of
**   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
**   GNU General Public License for more details.
**
**   You should have received a copy of the GNU General Public License
**   along with this program.  If not, see <https://www.gnu.org/licenses/>.
*/

#include <cmath>

#include <string>

#include "TCanny.h"

using namespace std::literals;

#ifdef TCANNY_X86
template<typename pixel_t> extern void filter_sse2(const VSFrame* src, VSFrame* dst, const TCannyData* const VS_RESTRICT d, const VSAPI* vsapi) noexcept;
template<typename pixel_t> extern void filter_avx2(const VSFrame* src, VSFrame* dst, const TCannyData* const VS_RESTRICT d, const VSAPI* vsapi) noexcept;
template<typename pixel_t> extern void filter_avx512(const VSFrame* src, VSFrame* dst, const TCannyData* const VS_RESTRICT d, const VSAPI* vsapi) noexcept;
#endif

static auto gaussianWeights(const float sigma, int& radius) noexcept {
    auto diameter{ std::max(static_cast<int>(sigma * 3.0f + 0.5f), 1) * 2 + 1 };
    radius = diameter / 2;
    auto weights{ new float[diameter]() };
    auto sum{ 0.0f };

    for (auto k{ -radius }; k <= radius; k++) {
        auto w{ std::exp(-(k * k) / (2.0f * sigma * sigma)) };
        weights[k + radius] = w;
        sum += w;
    }

    for (auto k{ 0 }; k < diameter; k++)
        weights[k] /= sum;

    return weights;
}

template<typename pixel_t>
static void gaussianBlur(const pixel_t* _srcp, float* VS_RESTRICT temp, float* VS_RESTRICT dstp, const int width, const int height,
                         const ptrdiff_t srcStride, const ptrdiff_t dstStride, const int radiusH, const int radiusV,
                         const float* weightsH, const float* weightsV) noexcept {
    auto diameter{ radiusV * 2 + 1 };
    auto srcp{ std::make_unique<const pixel_t* []>(diameter) };

    srcp[radiusV] = _srcp;
    for (auto i{ 1 }; i <= radiusV; i++)
        srcp[radiusV - i] = srcp[radiusV + i] = srcp[radiusV] + srcStride * i;

    weightsH += radiusH;

    for (auto y{ 0 }; y < height; y++) {
        for (auto x{ 0 }; x < width; x++) {
            auto sum{ 0.0f };

            for (auto v{ 0 }; v < diameter; v++)
                sum += srcp[v][x] * weightsV[v];

            temp[x] = sum;
        }

        for (auto i{ 1 }; i <= radiusH; i++) {
            temp[-i] = temp[i];
            temp[width - 1 + i] = temp[width - 1 - i];
        }

        for (auto x{ 0 }; x < width; x++) {
            auto sum{ 0.0f };

            for (auto v{ -radiusH }; v <= radiusH; v++)
                sum += temp[x + v] * weightsH[v];

            dstp[x] = sum;
        }

        for (auto i{ 0 }; i < diameter - 1; i++)
            srcp[i] = srcp[i + 1];
        srcp[diameter - 1] += (y < height - 1 - radiusV) ? srcStride : -srcStride;
        dstp += dstStride;
    }
}

template<typename pixel_t>
static void gaussianBlurH(const pixel_t* srcp, float* VS_RESTRICT temp, float* VS_RESTRICT dstp, const int width, const int height,
                          const ptrdiff_t srcStride, const ptrdiff_t dstStride, const int radius, const float* weights) noexcept {
    weights += radius;

    for (auto y{ 0 }; y < height; y++) {
        for (auto x{ 0 }; x < width; x++)
            temp[x] = srcp[x];

        for (auto i{ 1 }; i <= radius; i++) {
            temp[-i] = temp[i];
            temp[width - 1 + i] = temp[width - 1 - i];
        }

        for (auto x{ 0 }; x < width; x++) {
            auto sum{ 0.0f };

            for (auto v{ -radius }; v <= radius; v++)
                sum += temp[x + v] * weights[v];

            dstp[x] = sum;
        }

        srcp += srcStride;
        dstp += dstStride;
    }
}

template<typename pixel_t>
static void gaussianBlurV(const pixel_t* _srcp, float* VS_RESTRICT dstp, const int width, const int height,
                          const ptrdiff_t srcStride, const ptrdiff_t dstStride, const int radius, const float* weights) noexcept {
    auto diameter{ radius * 2 + 1 };
    auto srcp{ std::make_unique<const pixel_t* []>(diameter) };

    srcp[radius] = _srcp;
    for (auto i{ 1 }; i <= radius; i++)
        srcp[radius - i] = srcp[radius + i] = srcp[radius] + srcStride * i;

    for (auto y{ 0 }; y < height; y++) {
        for (auto x{ 0 }; x < width; x++) {
            auto sum{ 0.0f };

            for (auto v{ 0 }; v < diameter; v++)
                sum += srcp[v][x] * weights[v];

            dstp[x] = sum;
        }

        for (auto i{ 0 }; i < diameter - 1; i++)
            srcp[i] = srcp[i + 1];
        srcp[diameter - 1] += (y < height - 1 - radius) ? srcStride : -srcStride;
        dstp += dstStride;
    }
}

template<typename pixel_t>
static void copyPlane(const pixel_t* srcp, float* VS_RESTRICT dstp, const int width, const int height,
                      const ptrdiff_t srcStride, const ptrdiff_t dstStride) noexcept {
    for (auto y{ 0 }; y < height; y++) {
        for (auto x{ 0 }; x < width; x++)
            dstp[x] = srcp[x];

        srcp += srcStride;
        dstp += dstStride;
    }
}

static void detectEdge(float* VS_RESTRICT blur, float* VS_RESTRICT gradient, int* VS_RESTRICT direction, const int width, const int height,
                       const ptrdiff_t stride, const ptrdiff_t bgStride, const int mode, const int op, const float scale) noexcept {
    auto cur{ blur };
    auto next{ blur + bgStride };
    auto next2{ blur + bgStride * 2 };
    auto prev{ next };
    auto prev2{ next2 };

    cur[-1] = cur[1];
    cur[width] = cur[width - 2];
    if (op == FDOG) {
        cur[-2] = cur[2];
        cur[width + 1] = cur[width - 3];
    }

    for (auto y{ 0 }; y < height; y++) {
        next[-1] = next[1];
        next[width] = next[width - 2];
        if (op == FDOG) {
            next[-2] = next[2];
            next[width + 1] = next[width - 3];

            next2[-1] = next2[1];
            next2[-2] = next2[2];
            next2[width] = next2[width - 2];
            next2[width + 1] = next2[width - 3];
        }

        for (auto x{ 0 }; x < width; x++) {
            float gx{}, gy{};

            if (op != FDOG) {
                auto c1{ prev[x - 1] };
                auto c2{ prev[x] };
                auto c3{ prev[x + 1] };
                auto c4{ cur[x - 1] };
                auto c6{ cur[x + 1] };
                auto c7{ next[x - 1] };
                auto c8{ next[x] };
                auto c9{ next[x + 1] };

                switch (op) {
                case TRITICAL:
                    gx = c6 - c4;
                    gy = c2 - c8;
                    break;
                case PREWITT:
                    gx = (c3 + c6 + c9 - c1 - c4 - c7) / 2.0f;
                    gy = (c1 + c2 + c3 - c7 - c8 - c9) / 2.0f;
                    break;
                case SOBEL:
                    gx = c3 + 2.0f * c6 + c9 - c1 - 2.0f * c4 - c7;
                    gy = c1 + 2.0f * c2 + c3 - c7 - 2.0f * c8 - c9;
                    break;
                case SCHARR:
                    gx = 3.0f * c3 + 10.0f * c6 + 3.0f * c9 - 3.0f * c1 - 10.0f * c4 - 3.0f * c7;
                    gy = 3.0f * c1 + 10.0f * c2 + 3.0f * c3 - 3.0f * c7 - 10.0f * c8 - 3.0f * c9;
                    break;
                case KROON:
                    gx = 17.0f * c3 + 61.0f * c6 + 17.0f * c9 - 17.0f * c1 - 61.0f * c4 - 17.0f * c7;
                    gy = 17.0f * c1 + 61.0f * c2 + 17.0f * c3 - 17.0f * c7 - 61.0f * c8 - 17.0f * c9;
                    break;
                case KIRSCH:
                    auto g1{ 5.0f * c1 + 5.0f * c2 + 5.0f * c3 - 3.0f * c4 - 3.0f * c6 - 3.0f * c7 - 3.0f * c8 - 3.0f * c9 };
                    auto g2{ 5.0f * c1 + 5.0f * c2 - 3.0f * c3 + 5.0f * c4 - 3.0f * c6 - 3.0f * c7 - 3.0f * c8 - 3.0f * c9 };
                    auto g3{ 5.0f * c1 - 3.0f * c2 - 3.0f * c3 + 5.0f * c4 - 3.0f * c6 + 5.0f * c7 - 3.0f * c8 - 3.0f * c9 };
                    auto g4{ -3.0f * c1 - 3.0f * c2 - 3.0f * c3 + 5.0f * c4 - 3.0f * c6 + 5.0f * c7 + 5.0f * c8 - 3.0f * c9 };
                    auto g5{ -3.0f * c1 - 3.0f * c2 - 3.0f * c3 - 3.0f * c4 - 3.0f * c6 + 5.0f * c7 + 5.0f * c8 + 5.0f * c9 };
                    auto g6{ -3.0f * c1 - 3.0f * c2 - 3.0f * c3 - 3.0f * c4 + 5.0f * c6 - 3.0f * c7 + 5.0f * c8 + 5.0f * c9 };
                    auto g7{ -3.0f * c1 - 3.0f * c2 + 5.0f * c3 - 3.0f * c4 + 5.0f * c6 - 3.0f * c7 - 3.0f * c8 + 5.0f * c9 };
                    auto g8{ -3.0f * c1 + 5.0f * c2 + 5.0f * c3 - 3.0f * c4 + 5.0f * c6 - 3.0f * c7 - 3.0f * c8 - 3.0f * c9 };
                    auto g{ std::max({ std::abs(g1), std::abs(g2), std::abs(g3), std::abs(g4), std::abs(g5), std::abs(g6), std::abs(g7), std::abs(g8) }) };
                    gradient[x] = g * scale;
                    break;
                }
            } else {
                auto c1{ prev2[x - 2] };
                auto c2{ prev2[x - 1] };
                auto c3{ prev2[x] };
                auto c4{ prev2[x + 1] };
                auto c5{ prev2[x + 2] };
                auto c6{ prev[x - 2] };
                auto c7{ prev[x - 1] };
                auto c8{ prev[x] };
                auto c9{ prev[x + 1] };
                auto c10{ prev[x + 2] };
                auto c11{ cur[x - 2] };
                auto c12{ cur[x - 1] };
                auto c14{ cur[x + 1] };
                auto c15{ cur[x + 2] };
                auto c16{ next[x - 2] };
                auto c17{ next[x - 1] };
                auto c18{ next[x] };
                auto c19{ next[x + 1] };
                auto c20{ next[x + 2] };
                auto c21{ next2[x - 2] };
                auto c22{ next2[x - 1] };
                auto c23{ next2[x] };
                auto c24{ next2[x + 1] };
                auto c25{ next2[x + 2] };

                gx = c5 + 2.0f * c10 + 3.0f * c15 + 2.0f * c20 + c25 + c4 + 2.0f * c9 + 3.0f * c14 + 2.0f * c19 + c24
                    - c2 - 2.0f * c7 - 3.0f * c12 - 2.0f * c17 - c22 - c1 - 2.0f * c6 - 3.0f * c11 - 2.0f * c16 - c21;
                gy = c1 + 2.0f * c2 + 3.0f * c3 + 2.0f * c4 + c5 + c6 + 2.0f * c7 + 3.0f * c8 + 2.0f * c9 + c10
                    - c16 - 2.0f * c17 - 3.0f * c18 - 2.0f * c19 - c20 - c21 - 2.0f * c22 - 3.0f * c23 - 2.0f * c24 - c25;
            }

            if (op != KIRSCH) {
                gx *= scale;
                gy *= scale;
                gradient[x] = std::sqrt(gx * gx + gy * gy);
            }

            if (mode == 0) {
                auto dr{ std::atan2(gy, gx) };
                if (dr < 0.0f)
                    dr += M_PIF;

                auto bin{ static_cast<int>(dr * 4.0f * M_1_PIF + 0.5f) };
                direction[x] = (bin >= 4) ? 0 : bin;
            }
        }

        prev2 = prev;
        prev = cur;
        cur = next;
        if (op != FDOG) {
            next += (y < height - 2) ? bgStride : -bgStride;
        } else {
            next = next2;
            next2 += (y < height - 3) ? bgStride : -bgStride;
        }
        gradient += bgStride;
        direction += stride;
    }
}

static void nonMaximumSuppression(const int* direction, float* VS_RESTRICT gradient, float* VS_RESTRICT blur, const int width, const int height,
                                  const ptrdiff_t stride, const ptrdiff_t bgStride, const int radiusAlign) noexcept {
    const ptrdiff_t offsets[]{ 1, -bgStride + 1, -bgStride, -bgStride - 1 };

    gradient[-1] = gradient[1];
    gradient[-1 + bgStride * (height - 1)] = gradient[1 + bgStride * (height - 1)];
    gradient[width] = gradient[width - 2];
    gradient[width + bgStride * (height - 1)] = gradient[width - 2 + bgStride * (height - 1)];
    std::copy_n(gradient - radiusAlign + bgStride, width + radiusAlign * 2, gradient - radiusAlign - bgStride);
    std::copy_n(gradient - radiusAlign + bgStride * (height - 2), width + radiusAlign * 2, gradient - radiusAlign + bgStride * height);

    for (auto y{ 0 }; y < height; y++) {
        for (auto x{ 0 }; x < width; x++) {
            auto offset{ offsets[direction[x]] };
            blur[x] = (gradient[x] >= std::max(gradient[x + offset], gradient[x - offset])) ? gradient[x] : fltLowest;
        }

        direction += stride;
        gradient += bgStride;
        blur += bgStride;
    }
}

template<typename pixel_t>
static void binarizeCE(const float* srcp, pixel_t* VS_RESTRICT dstp, const int width, const int height, const ptrdiff_t srcStride, const ptrdiff_t dstStride,
                       const int peak) noexcept {
    for (auto y{ 0 }; y < height; y++) {
        for (auto x{ 0 }; x < width; x++) {
            if constexpr (std::is_integral_v<pixel_t>)
                dstp[x] = (srcp[x] == fltMax) ? static_cast<pixel_t>(peak) : 0;
            else
                dstp[x] = (srcp[x] == fltMax) ? 1.0f : 0.0f;
        }

        srcp += srcStride;
        dstp += dstStride;
    }
}

template<typename pixel_t>
static void discretizeGM(const float* srcp, pixel_t* VS_RESTRICT dstp, const int width, const int height, const ptrdiff_t srcStride, const ptrdiff_t dstStride,
                         const int peak) noexcept {
    for (auto y{ 0 }; y < height; y++) {
        for (auto x{ 0 }; x < width; x++) {
            if constexpr (std::is_integral_v<pixel_t>)
                dstp[x] = static_cast<pixel_t>(std::min(static_cast<int>(srcp[x] + 0.5f), peak));
            else
                dstp[x] = srcp[x];
        }

        srcp += srcStride;
        dstp += dstStride;
    }
}

template<typename pixel_t>
static void filter_c(const VSFrame* src, VSFrame* dst, const TCannyData* const VS_RESTRICT d, const VSAPI* vsapi) noexcept {
    for (auto plane{ 0 }; plane < d->vi->format.numPlanes; plane++) {
        if (d->process[plane]) {
            const auto width{ vsapi->getFrameWidth(src, plane) };
            const auto height{ vsapi->getFrameHeight(src, plane) };
            const auto stride{ vsapi->getStride(src, plane) / d->vi->format.bytesPerSample };
            const auto bgStride{ stride + d->radiusAlign * 2 };
            auto srcp{ reinterpret_cast<const pixel_t*>(vsapi->getReadPtr(src, plane)) };
            auto dstp{ reinterpret_cast<pixel_t*>(vsapi->getWritePtr(dst, plane)) };

            const auto threadId{ std::this_thread::get_id() };
            auto blur{ d->blur.at(threadId).get() + d->radiusAlign };
            auto gradient{ d->gradient.at(threadId).get() + bgStride + d->radiusAlign };
            auto direction{ d->direction.at(threadId).get() };
            auto found{ d->found.at(threadId).get() };

            if (d->radiusH[plane] && d->radiusV[plane])
                gaussianBlur(srcp, gradient, blur, width, height, stride, bgStride, d->radiusH[plane], d->radiusV[plane],
                             d->weightsH[plane].get(), d->weightsV[plane].get());
            else if (d->radiusH[plane])
                gaussianBlurH(srcp, gradient, blur, width, height, stride, bgStride, d->radiusH[plane], d->weightsH[plane].get());
            else if (d->radiusV[plane])
                gaussianBlurV(srcp, blur, width, height, stride, bgStride, d->radiusV[plane], d->weightsV[plane].get());
            else
                copyPlane(srcp, blur, width, height, stride, bgStride);

            if (d->mode != -1) {
                detectEdge(blur, gradient, direction, width, height, stride, bgStride, d->mode, d->op, d->scale);

                if (d->mode == 0) {
                    nonMaximumSuppression(direction, gradient, blur, width, height, stride, bgStride, d->radiusAlign);
                    hysteresis(blur, found, width, height, bgStride, d->t_h, d->t_l);
                }
            }

            if (d->mode == 0)
                binarizeCE(blur, dstp, width, height, bgStride, stride, d->peak);
            else
                discretizeGM(d->mode == 1 ? gradient : blur, dstp, width, height, bgStride, stride, d->peak);
        }
    }
}

static const VSFrame* VS_CC tcannyGetFrame(int n, int activationReason, void* instanceData, [[maybe_unused]] void** frameData, VSFrameContext* frameCtx, VSCore* core, const VSAPI* vsapi) {
    auto d{ static_cast<TCannyData*>(instanceData) };

    if (activationReason == arInitial) {
        vsapi->requestFrameFilter(n, d->node, frameCtx);
    } else if (activationReason == arAllFramesReady) {
        auto src{ vsapi->getFrameFilter(n, d->node, frameCtx) };
        const VSFrame* fr[]{ d->process[0] ? nullptr : src, d->process[1] ? nullptr : src, d->process[2] ? nullptr : src };
        const int pl[]{ 0, 1, 2 };
        auto dst{ vsapi->newVideoFrame2(&d->vi->format, d->vi->width, d->vi->height, fr, pl, src, core) };

        try {
            auto threadId{ std::this_thread::get_id() };
            auto stride{ vsapi->getStride(src, 0) / d->vi->format.bytesPerSample };

            if (!d->blur.count(threadId)) {
                auto blur{ vsh::vsh_aligned_malloc<float>((stride + d->radiusAlign * 2) * d->vi->height * sizeof(float), d->alignment) };
                if (!blur)
                    throw "malloc failure (blur)"s;
                d->blur.emplace(threadId, unique_float{ blur, vsh::vsh_aligned_free });

                auto gradient{ vsh::vsh_aligned_malloc<float>((stride + d->radiusAlign * 2) * (d->vi->height + 2) * sizeof(float), d->alignment) };
                if (!gradient)
                    throw "malloc failure (gradient)"s;
                d->gradient.emplace(threadId, unique_float{ gradient, vsh::vsh_aligned_free });

                if (d->mode == 0) {
                    auto direction{ vsh::vsh_aligned_malloc<int>(stride * d->vi->height * sizeof(int), d->alignment) };
                    if (!direction)
                        throw "malloc failure (direction)"s;
                    d->direction.emplace(threadId, unique_int{ direction, vsh::vsh_aligned_free });

                    auto found{ new (std::nothrow) bool[d->vi->width * d->vi->height] };
                    if (!found)
                        throw "malloc failure (found)"s;
                    d->found.emplace(threadId, found);
                } else {
                    d->direction.emplace(threadId, unique_int{ nullptr, vsh::vsh_aligned_free });
                    d->found.emplace(threadId, nullptr);
                }
            }
        } catch (const std::string& error) {
            vsapi->setFilterError(("TCanny: " + error).c_str(), frameCtx);
            vsapi->freeFrame(src);
            vsapi->freeFrame(dst);
            return nullptr;
        }

        d->filter(src, dst, d, vsapi);

        vsapi->freeFrame(src);
        return dst;
    }

    return nullptr;
}

static void VS_CC tcannyFree(void* instanceData, [[maybe_unused]] VSCore* core, const VSAPI* vsapi) {
    auto d{ static_cast<TCannyData*>(instanceData) };
    vsapi->freeNode(d->node);
    delete d;
}

static void VS_CC tcannyCreate(const VSMap* in, VSMap* out, [[maybe_unused]] void* userData, VSCore* core, const VSAPI* vsapi) {
    auto d{ std::make_unique<TCannyData>() };

    try {
        d->node = vsapi->mapGetNode(in, "clip", 0, nullptr);
        d->vi = vsapi->getVideoInfo(d->node);
        auto err{ 0 };

        if (!vsh::isConstantVideoFormat(d->vi) ||
            (d->vi->format.sampleType == stInteger && d->vi->format.bitsPerSample > 16) ||
            (d->vi->format.sampleType == stFloat && d->vi->format.bitsPerSample != 32))
            throw "only constant format 8-16 bit integer and 32 bit float input supported"s;

        if (d->vi->height < 3)
            throw "clip's height must be at least 3"s;

        const auto numSigmaH{ vsapi->mapNumElements(in, "sigma") };
        if (numSigmaH > d->vi->format.numPlanes)
            throw "more sigma given than there are planes"s;

        const auto numSigmaV{ vsapi->mapNumElements(in, "sigma_v") };
        if (numSigmaV > d->vi->format.numPlanes)
            throw "more sigma_v given than there are planes"s;

        float sigmaH[3]{}, sigmaV[3]{};

        for (auto i{ 0 }; i < d->vi->format.numPlanes; i++) {
            if (i < numSigmaH)
                sigmaH[i] = vsapi->mapGetFloatSaturated(in, "sigma", i, nullptr);
            else if (i == 0)
                sigmaH[0] = 1.5f;
            else if (i == 1)
                sigmaH[1] = sigmaH[0] / (1 << d->vi->format.subSamplingW);
            else
                sigmaH[2] = sigmaH[1];

            if (i < numSigmaV)
                sigmaV[i] = vsapi->mapGetFloatSaturated(in, "sigma_v", i, nullptr);
            else if (i < numSigmaH)
                sigmaV[i] = sigmaH[i];
            else if (i == 0)
                sigmaV[0] = 1.5f;
            else if (i == 1)
                sigmaV[1] = sigmaV[0] / (1 << d->vi->format.subSamplingH);
            else
                sigmaV[2] = sigmaV[1];
        }

        d->t_h = vsapi->mapGetFloatSaturated(in, "t_h", 0, &err);
        if (err)
            d->t_h = 8.0f;

        d->t_l = vsapi->mapGetFloatSaturated(in, "t_l", 0, &err);
        if (err)
            d->t_l = 1.0f;

        d->mode = vsapi->mapGetIntSaturated(in, "mode", 0, &err);

        d->op = vsapi->mapGetIntSaturated(in, "op", 0, &err);
        if (err)
            d->op = PREWITT;

        d->scale = vsapi->mapGetFloatSaturated(in, "scale", 0, &err);
        if (err)
            d->scale = 1.0f;

        auto opt{ vsapi->mapGetIntSaturated(in, "opt", 0, &err) };

        const auto m{ vsapi->mapNumElements(in, "planes") };

        for (auto i{ 0 }; i < 3; i++)
            d->process[i] = (m <= 0);

        for (auto i{ 0 }; i < m; i++) {
            auto n{ vsapi->mapGetIntSaturated(in, "planes", i, nullptr) };

            if (n < 0 || n >= d->vi->format.numPlanes)
                throw "plane index out of range"s;

            if (d->process[n])
                throw "plane specified twice"s;

            d->process[n] = true;
        }

        for (auto i{ 0 }; i < d->vi->format.numPlanes; i++) {
            if (sigmaH[i] < 0.0f)
                throw "sigma must be greater than or equal to 0.0"s;

            if (sigmaV[i] < 0.0f)
                throw "sigma_v must be greater than or equal to 0.0"s;
        }

        if (d->t_l >= d->t_h)
            throw "t_h must be greater than t_l"s;

        if (d->mode < -1 || d->mode > 1)
            throw "mode must be -1, 0, or 1"s;

        if (d->op < 0 || d->op > 6)
            throw "op must be 0, 1, 2, 3, 4, 5, or 6"s;

        if (d->op == 5 && d->mode == 0)
            throw "op=5 cannot be used when mode=0"s;

        if (d->scale <= 0.0f)
            throw "scale must be greater than 0.0"s;

        if (opt < 0 || opt > 4)
            throw "opt must be 0, 1, 2, 3, or 4"s;

        auto vectorSize{ 1 };
        {
            d->alignment = 4;

#ifdef TCANNY_X86
            const auto iset{ instrset_detect() };

            if ((opt == 0 && iset >= 10) || opt == 4) {
                vectorSize = 16;
                d->alignment = 64;
            } else if ((opt == 0 && iset >= 8) || opt == 3) {
                vectorSize = 8;
                d->alignment = 32;
            } else if ((opt == 0 && iset >= 2) || opt == 2) {
                vectorSize = 4;
                d->alignment = 16;
            }
#endif

            if (d->vi->format.bytesPerSample == 1) {
                d->filter = filter_c<uint8_t>;

#ifdef TCANNY_X86
                if ((opt == 0 && iset >= 10) || opt == 4)
                    d->filter = filter_avx512<uint8_t>;
                else if ((opt == 0 && iset >= 8) || opt == 3)
                    d->filter = filter_avx2<uint8_t>;
                else if ((opt == 0 && iset >= 2) || opt == 2)
                    d->filter = filter_sse2<uint8_t>;
#endif
            } else if (d->vi->format.bytesPerSample == 2) {
                d->filter = filter_c<uint16_t>;

#ifdef TCANNY_X86
                if ((opt == 0 && iset >= 10) || opt == 4)
                    d->filter = filter_avx512<uint16_t>;
                else if ((opt == 0 && iset >= 8) || opt == 3)
                    d->filter = filter_avx2<uint16_t>;
                else if ((opt == 0 && iset >= 2) || opt == 2)
                    d->filter = filter_sse2<uint16_t>;
#endif
            } else {
                d->filter = filter_c<float>;

#ifdef TCANNY_X86
                if ((opt == 0 && iset >= 10) || opt == 4)
                    d->filter = filter_avx512<float>;
                else if ((opt == 0 && iset >= 8) || opt == 3)
                    d->filter = filter_avx2<float>;
                else if ((opt == 0 && iset >= 2) || opt == 2)
                    d->filter = filter_sse2<float>;
#endif
            }
        }

        VSCoreInfo info;
        vsapi->getCoreInfo(core, &info);

        d->blur.reserve(info.numThreads);
        d->gradient.reserve(info.numThreads);
        d->direction.reserve(info.numThreads);
        d->found.reserve(info.numThreads);

        if (d->vi->format.sampleType == stInteger) {
            d->peak = (1 << d->vi->format.bitsPerSample) - 1;
            auto scale{ d->peak / 255.0f };
            d->t_h *= scale;
            d->t_l *= scale;
        } else {
            d->t_h /= 255.0f;
            d->t_l /= 255.0f;
        }

        for (auto plane{ 0 }; plane < d->vi->format.numPlanes; plane++) {
            if (d->process[plane]) {
                auto planeOrder{ plane == 0 ? "first" : (plane == 1 ? "second" : "third") };

                if (sigmaH[plane]) {
                    d->weightsH[plane].reset(gaussianWeights(sigmaH[plane], d->radiusH[plane]));

                    auto width{ d->vi->width >> (plane ? d->vi->format.subSamplingW : 0) };
                    if (width < d->radiusH[plane] + 1)
                        throw "the "s + planeOrder + " plane's width must be at least " + std::to_string(d->radiusH[plane] + 1) + " for specified sigma";
                }

                if (sigmaV[plane]) {
                    d->weightsV[plane].reset(gaussianWeights(sigmaV[plane], d->radiusV[plane]));

                    auto height{ d->vi->height >> (plane ? d->vi->format.subSamplingH : 0) };
                    if (height < d->radiusV[plane] + 1)
                        throw "the "s + planeOrder + " plane's height must be at least " + std::to_string(d->radiusV[plane] + 1) + " for specified sigma_v";
                }
            }
        }

        d->radiusAlign = (std::max({ d->radiusH[0], d->radiusH[1], d->radiusH[2], d->op == FDOG ? 2 : 1 }) + vectorSize - 1) & ~(vectorSize - 1);
    } catch (const std::string& error) {
        vsapi->mapSetError(out, ("TCanny: " + error).c_str());
        vsapi->freeNode(d->node);
        return;
    }

    VSFilterDependency deps[]{ {d->node, rpStrictSpatial} };
    vsapi->createVideoFilter(out, "TCanny", d->vi, tcannyGetFrame, tcannyFree, fmParallel, deps, 1, d.get(), core);
    d.release();
}

//////////////////////////////////////////
// Init

VS_EXTERNAL_API(void) VapourSynthPluginInit2(VSPlugin* plugin, const VSPLUGINAPI* vspapi) {
    vspapi->configPlugin("com.holywu.tcanny", "tcanny", "Build an edge map using canny edge detection", VS_MAKE_VERSION(13, 0), VAPOURSYNTH_API_VERSION, 0, plugin);
    vspapi->registerFunction("TCanny",
                             "clip:vnode;"
                             "sigma:float[]:opt;"
                             "sigma_v:float[]:opt;"
                             "t_h:float:opt;"
                             "t_l:float:opt;"
                             "mode:int:opt;"
                             "op:int:opt;"
                             "scale:float:opt;"
                             "opt:int:opt;"
                             "planes:int[]:opt;",
                             "clip:vnode;",
                             tcannyCreate, nullptr, plugin);
}
