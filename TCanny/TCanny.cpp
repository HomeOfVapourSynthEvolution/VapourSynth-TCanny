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

#include "TCanny.hpp"

#ifdef VS_TARGET_CPU_X86
template<typename T> extern void filter_sse2(const VSFrameRef *, VSFrameRef *, const TCannyData * const VS_RESTRICT, const VSAPI *) noexcept;
template<typename T> extern void filter_avx(const VSFrameRef *, VSFrameRef *, const TCannyData * const VS_RESTRICT, const VSAPI *) noexcept;
template<typename T> extern void filter_avx2(const VSFrameRef *, VSFrameRef *, const TCannyData * const VS_RESTRICT, const VSAPI *) noexcept;
#endif

template<typename T>
static void copyPlane(const T * srcp, float * VS_RESTRICT dstp, const int width, const int height, const int srcStride, const int dstStride, const float offset) noexcept {
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            if (std::is_integral<T>::value)
                dstp[x] = srcp[x];
            else
                dstp[x] = srcp[x] + offset;
        }

        srcp += srcStride;
        dstp += dstStride;
    }
}

template<typename T>
static void gaussianBlur(const T * _srcp, float * VS_RESTRICT temp, float * VS_RESTRICT dstp, const float * weightsH, const float * weightsV, const int width, const int height,
                         const int srcStride, const int dstStride, const int radiusH, const int radiusV, const float offset) noexcept {
    const int diameter = radiusV * 2 + 1;
    const T ** srcp = new const T *[diameter];

    srcp[radiusV] = _srcp;
    for (int i = 1; i <= radiusV; i++) {
        srcp[radiusV - i] = srcp[radiusV - 1 + i];
        srcp[radiusV + i] = srcp[radiusV] + srcStride * i;
    }

    weightsH += radiusH;

    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            float sum = 0.f;

            for (int i = 0; i < diameter; i++) {
                if (std::is_integral<T>::value)
                    sum += srcp[i][x] * weightsV[i];
                else
                    sum += (srcp[i][x] + offset) * weightsV[i];
            }

            temp[x] = sum;
        }

        for (int i = 1; i <= radiusH; i++) {
            temp[-i] = temp[-1 + i];
            temp[width - 1 + i] = temp[width - i];
        }

        for (int x = 0; x < width; x++) {
            float sum = 0.f;

            for (int i = -radiusH; i <= radiusH; i++)
                sum += temp[x + i] * weightsH[i];

            dstp[x] = sum;
        }

        for (int i = 0; i < diameter - 1; i++)
            srcp[i] = srcp[i + 1];
        if (y < height - 1 - radiusV)
            srcp[diameter - 1] += srcStride;
        else if (y > height - 1 - radiusV)
            srcp[diameter - 1] -= srcStride;
        dstp += dstStride;
    }

    delete[] srcp;
}

template<typename T>
static void gaussianBlurV(const T * _srcp, float * VS_RESTRICT dstp, const float * weights, const int width, const int height, const int srcStride, const int dstStride,
                          const int radius, const float offset) noexcept {
    const int diameter = radius * 2 + 1;
    const T ** srcp = new const T *[diameter];

    srcp[radius] = _srcp;
    for (int i = 1; i <= radius; i++) {
        srcp[radius - i] = srcp[radius - 1 + i];
        srcp[radius + i] = srcp[radius] + srcStride * i;
    }

    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            float sum = 0.f;

            for (int i = 0; i < diameter; i++) {
                if (std::is_integral<T>::value)
                    sum += srcp[i][x] * weights[i];
                else
                    sum += (srcp[i][x] + offset) * weights[i];
            }

            dstp[x] = sum;
        }

        for (int i = 0; i < diameter - 1; i++)
            srcp[i] = srcp[i + 1];
        if (y < height - 1 - radius)
            srcp[diameter - 1] += srcStride;
        else if (y > height - 1 - radius)
            srcp[diameter - 1] -= srcStride;
        dstp += dstStride;
    }

    delete[] srcp;
}

template<typename T>
static void gaussianBlurH(const T * srcp, float * VS_RESTRICT temp, float * VS_RESTRICT dstp, const float * weights, const int width, const int height,
                          const int srcStride, const int dstStride, const int radius, const float offset) noexcept {
    weights += radius;

    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            if (std::is_integral<T>::value)
                temp[x] = srcp[x];
            else
                temp[x] = srcp[x] + offset;
        }

        for (int i = 1; i <= radius; i++) {
            temp[-i] = temp[-1 + i];
            temp[width - 1 + i] = temp[width - i];
        }

        for (int x = 0; x < width; x++) {
            float sum = 0.f;

            for (int i = -radius; i <= radius; i++)
                sum += temp[x + i] * weights[i];

            dstp[x] = sum;
        }

        srcp += srcStride;
        dstp += dstStride;
    }
}

static void detectEdge(float * blur, float * VS_RESTRICT gradient, unsigned * VS_RESTRICT direction, const int width, const int height, const int stride, const int bgStride,
                       const int mode, const int op) noexcept {
    float * VS_RESTRICT srcpp = blur;
    float * VS_RESTRICT srcp = blur;
    float * VS_RESTRICT srcpn = blur + bgStride;

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

            if (mode == 0) {
                float dr = std::atan2(gy, gx);
                if (dr < 0.f)
                    dr += M_PIF;

                const unsigned bin = static_cast<unsigned>(dr * 4.f * M_1_PIF + 0.5f);
                direction[x] = (bin >= 4) ? 0 : bin;
            }
        }

        srcpp = srcp;
        srcp = srcpn;
        if (y < height - 2)
            srcpn += bgStride;
        gradient += bgStride;
        direction += stride;
    }
}

static void nonMaximumSuppression(const unsigned * direction, float * VS_RESTRICT gradient, float * VS_RESTRICT blur, const int width, const int height,
                                  const int stride, const int bgStride, const int radiusAlign) noexcept {
    const int offsets[] = { 1, -bgStride + 1, -bgStride, -bgStride - 1 };

    gradient[-1] = gradient[0];
    gradient[-1 + bgStride * (height - 1)] = gradient[bgStride * (height - 1)];
    gradient[width] = gradient[width - 1];
    gradient[width + bgStride * (height - 1)] = gradient[width - 1 + bgStride * (height - 1)];
    std::copy_n(gradient - radiusAlign, width + radiusAlign * 2, gradient - radiusAlign - bgStride);
    std::copy_n(gradient - radiusAlign + bgStride * (height - 1), width + radiusAlign * 2, gradient - radiusAlign + bgStride * height);

    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            const int offset = offsets[direction[x]];
            blur[x] = (gradient[x] >= std::max(gradient[x + offset], gradient[x - offset])) ? gradient[x] : fltLowest;
        }

        direction += stride;
        gradient += bgStride;
        blur += bgStride;
    }
}

template<typename T>
static void outputGB(const float * srcp, T * VS_RESTRICT dstp, const int width, const int height, const int srcStride, const int dstStride,
                     const uint16_t peak, const float offset) noexcept {
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            if (std::is_integral<T>::value)
                dstp[x] = std::min<unsigned>(srcp[x] + 0.5f, peak);
            else
                dstp[x] = srcp[x] - offset;
        }

        srcp += srcStride;
        dstp += dstStride;
    }
}

template<typename T>
static void binarizeCE(const float * srcp, T * VS_RESTRICT dstp, const int width, const int height, const int srcStride, const int dstStride,
                       const uint16_t peak) noexcept {
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            if (std::is_integral<T>::value)
                dstp[x] = (srcp[x] == fltMax) ? peak : 0;
            else
                dstp[x] = (srcp[x] == fltMax) ? 1.0f : 0.0f;
        }

        srcp += srcStride;
        dstp += dstStride;
    }
}

template<typename T>
static void discretizeGM(const float * srcp, T * VS_RESTRICT dstp, const int width, const int height, const int srcStride, const int dstStride,
                         const float magnitude, const uint16_t peak) noexcept {
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            if (std::is_integral<T>::value)
                dstp[x] = std::min<unsigned>(srcp[x] * magnitude + 0.5f, peak);
            else
                dstp[x] = srcp[x] * magnitude;
        }

        srcp += srcStride;
        dstp += dstStride;
    }
}

template<typename T>
static void filter_c(const VSFrameRef * src, VSFrameRef * dst, const TCannyData * const VS_RESTRICT d, const VSAPI * vsapi) noexcept {
    for (int plane = 0; plane < d->vi->format->numPlanes; plane++) {
        if (d->process[plane]) {
            const int width = vsapi->getFrameWidth(src, plane);
            const int height = vsapi->getFrameHeight(src, plane);
            const int stride = vsapi->getStride(src, plane) / sizeof(T);
            const int bgStride = stride + d->radiusAlign * 2;
            const T * srcp = reinterpret_cast<const T *>(vsapi->getReadPtr(src, plane));
            T * dstp = reinterpret_cast<T *>(vsapi->getWritePtr(dst, plane));

            const auto threadId = std::this_thread::get_id();
            float * blur = d->blur.at(threadId) + d->radiusAlign;
            float * gradient = d->gradient.at(threadId) + bgStride + d->radiusAlign;
            unsigned * direction = d->direction.at(threadId);
            bool * found = d->found.at(threadId);

            if (d->radiusV[plane] && d->radiusH[plane])
                gaussianBlur(srcp, gradient, blur, d->weightsH[plane], d->weightsV[plane], width, height, stride, bgStride, d->radiusH[plane], d->radiusV[plane], d->offset[plane]);
            else if (d->radiusV[plane])
                gaussianBlurV(srcp, blur, d->weightsV[plane], width, height, stride, bgStride, d->radiusV[plane], d->offset[plane]);
            else if (d->radiusH[plane])
                gaussianBlurH(srcp, gradient, blur, d->weightsH[plane], width, height, stride, bgStride, d->radiusH[plane], d->offset[plane]);
            else
                copyPlane(srcp, blur, width, height, stride, bgStride, d->offset[plane]);

            if (d->mode != -1) {
                detectEdge(blur, gradient, direction, width, height, stride, bgStride, d->mode, d->op);

                if (d->mode == 0) {
                    nonMaximumSuppression(direction, gradient, blur, width, height, stride, bgStride, d->radiusAlign);
                    hysteresis(blur, found, width, height, bgStride, d->t_h, d->t_l);
                }
            }

            if (d->mode == -1)
                outputGB(blur, dstp, width, height, bgStride, stride, d->peak, d->offset[plane]);
            else if (d->mode == 0)
                binarizeCE(blur, dstp, width, height, bgStride, stride, d->peak);
            else
                discretizeGM(gradient, dstp, width, height, bgStride, stride, d->magnitude, d->peak);
        }
    }
}

static void selectFunctions(const unsigned opt, TCannyData * d) noexcept {
    d->vectorSize = 1;
    d->alignment = 4;

#ifdef VS_TARGET_CPU_X86
    const int iset = instrset_detect();

    if ((opt == 0 && iset >= 7) || opt >= 3) {
        d->vectorSize = 8;
        d->alignment = 32;
    } else if ((opt == 0 && iset >= 2) || opt == 2) {
        d->vectorSize = 4;
        d->alignment = 16;
    }
#endif

    if (d->vi->format->bytesPerSample == 1) {
        d->filter = filter_c<uint8_t>;

#ifdef VS_TARGET_CPU_X86
        if ((opt == 0 && iset >= 8) || opt == 4)
            d->filter = filter_avx2<uint8_t>;
        else if ((opt == 0 && iset == 7) || opt == 3)
            d->filter = filter_avx<uint8_t>;
        else if ((opt == 0 && iset >= 2) || opt == 2)
            d->filter = filter_sse2<uint8_t>;
#endif
    } else if (d->vi->format->bytesPerSample == 2) {
        d->filter = filter_c<uint16_t>;

#ifdef VS_TARGET_CPU_X86
        if ((opt == 0 && iset >= 8) || opt == 4)
            d->filter = filter_avx2<uint16_t>;
        else if ((opt == 0 && iset == 7) || opt == 3)
            d->filter = filter_avx<uint16_t>;
        else if ((opt == 0 && iset >= 2) || opt == 2)
            d->filter = filter_sse2<uint16_t>;
#endif
    } else {
        d->filter = filter_c<float>;

#ifdef VS_TARGET_CPU_X86
        if ((opt == 0 && iset >= 8) || opt == 4)
            d->filter = filter_avx2<float>;
        else if ((opt == 0 && iset == 7) || opt == 3)
            d->filter = filter_avx<float>;
        else if ((opt == 0 && iset >= 2) || opt == 2)
            d->filter = filter_sse2<float>;
#endif
    }
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
        const VSFrameRef * fr[] = { d->process[0] ? nullptr : src, d->process[1] ? nullptr : src, d->process[2] ? nullptr : src };
        const int pl[] = { 0, 1, 2 };
        VSFrameRef * dst = vsapi->newVideoFrame2(d->vi->format, d->vi->width, d->vi->height, fr, pl, src, core);

        try {
            auto threadId = std::this_thread::get_id();

            if (!d->blur.count(threadId)) {
                float * blur = vs_aligned_malloc<float>((vsapi->getStride(src, 0) / d->vi->format->bytesPerSample + d->radiusAlign * 2) * d->vi->height * sizeof(float), d->alignment);
                if (!blur)
                    throw std::string{ "malloc failure (blur)" };
                d->blur.emplace(threadId, blur);

                float * gradient = vs_aligned_malloc<float>((vsapi->getStride(src, 0) / d->vi->format->bytesPerSample + d->radiusAlign * 2) * (d->vi->height + 2) * sizeof(float), d->alignment);
                if (!gradient)
                    throw std::string{ "malloc failure (gradient)" };
                d->gradient.emplace(threadId, gradient);

                if (d->mode == 0) {
                    unsigned * direction = vs_aligned_malloc<unsigned>(vsapi->getStride(src, 0) / d->vi->format->bytesPerSample * d->vi->height * sizeof(unsigned), d->alignment);
                    if (!direction)
                        throw std::string{ "malloc failure (direction)" };
                    d->direction.emplace(threadId, direction);

                    bool * found = new (std::nothrow) bool[d->vi->width * d->vi->height];
                    if (!found)
                        throw std::string{ "malloc failure (found)" };
                    d->found.emplace(threadId, found);
                } else {
                    d->direction.emplace(threadId, nullptr);
                    d->found.emplace(threadId, nullptr);
                }
            }
        } catch (const std::string & error) {
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

static void VS_CC tcannyFree(void *instanceData, VSCore *core, const VSAPI *vsapi) {
    TCannyData * d = static_cast<TCannyData *>(instanceData);

    vsapi->freeNode(d->node);

    for (int i = 0; i < 3; i++) {
        delete[] d->weightsH[i];
        delete[] d->weightsV[i];
    }

    for (auto & iter : d->blur)
        vs_aligned_free(iter.second);

    for (auto & iter : d->gradient)
        vs_aligned_free(iter.second);

    for (auto & iter : d->direction)
        vs_aligned_free(iter.second);

    for (auto & iter : d->found)
        delete[] iter.second;

    delete d;
}

static void VS_CC tcannyCreate(const VSMap *in, VSMap *out, void *userData, VSCore *core, const VSAPI *vsapi) {
    std::unique_ptr<TCannyData> d = std::make_unique<TCannyData>();
    int err;

    d->node = vsapi->propGetNode(in, "clip", 0, nullptr);
    d->vi = vsapi->getVideoInfo(d->node);

    try {
        if (!isConstantFormat(d->vi) ||
            (d->vi->format->sampleType == stInteger && d->vi->format->bitsPerSample > 16) ||
            (d->vi->format->sampleType == stFloat && d->vi->format->bitsPerSample != 32))
            throw std::string{ "only constant format 8-16 bit integer and 32 bit float input supported" };

        if (d->vi->height < 2)
            throw std::string{ "the clip's height must be greater than or equal to 2" };

        const int numSigmaH = vsapi->propNumElements(in, "sigma");
        if (numSigmaH > d->vi->format->numPlanes)
            throw std::string{ "more sigma given than there are planes" };

        const int numSigmaV = vsapi->propNumElements(in, "sigma_v");
        if (numSigmaV > d->vi->format->numPlanes)
            throw std::string{ "more sigma_v given than there are planes" };

        float sigmaH[3], sigmaV[3];

        for (int i = 0; i < 3; i++) {
            if (i < numSigmaH)
                sigmaH[i] = static_cast<float>(vsapi->propGetFloat(in, "sigma", i, nullptr));
            else if (i == 0)
                sigmaH[0] = 1.5f;
            else if (i == 1)
                sigmaH[1] = sigmaH[0] / (1 << d->vi->format->subSamplingW);
            else
                sigmaH[2] = sigmaH[1];

            if (i < numSigmaV)
                sigmaV[i] = static_cast<float>(vsapi->propGetFloat(in, "sigma_v", i, nullptr));
            else if (i < numSigmaH)
                sigmaV[i] = sigmaH[i];
            else if (i == 0)
                sigmaV[0] = 1.5f;
            else if (i == 1)
                sigmaV[1] = sigmaV[0] / (1 << d->vi->format->subSamplingH);
            else
                sigmaV[2] = sigmaV[1];
        }

        d->t_h = static_cast<float>(vsapi->propGetFloat(in, "t_h", 0, &err));
        if (err)
            d->t_h = 8.f;

        d->t_l = static_cast<float>(vsapi->propGetFloat(in, "t_l", 0, &err));
        if (err)
            d->t_l = 1.f;

        d->mode = int64ToIntS(vsapi->propGetInt(in, "mode", 0, &err));

        d->op = int64ToIntS(vsapi->propGetInt(in, "op", 0, &err));
        if (err)
            d->op = 1;

        float gmmax = static_cast<float>(vsapi->propGetFloat(in, "gmmax", 0, &err));
        if (err)
            gmmax = 50.f;

        const int opt = int64ToIntS(vsapi->propGetInt(in, "opt", 0, &err));

        const int m = vsapi->propNumElements(in, "planes");

        for (int i = 0; i < 3; i++)
            d->process[i] = (m <= 0);

        for (int i = 0; i < m; i++) {
            const int n = int64ToIntS(vsapi->propGetInt(in, "planes", i, nullptr));

            if (n < 0 || n >= d->vi->format->numPlanes)
                throw std::string{ "plane index out of range" };

            if (d->process[n])
                throw std::string{ "plane specified twice" };

            d->process[n] = true;
        }

        for (int i = 0; i < 3; i++) {
            if (sigmaH[i] < 0.f)
                throw std::string{ "sigma must be greater than or equal to 0.0" };

            if (sigmaV[i] < 0.f)
                throw std::string{ "sigma_v must be greater than or equal to 0.0" };
        }

        if (d->t_l >= d->t_h)
            throw std::string{ "t_h must be greater than t_l" };

        if (d->mode < -1 || d->mode > 1)
            throw std::string{ "mode must be -1, 0, or 1" };

        if (d->op < 0 || d->op > 3)
            throw std::string{ "op must be 0, 1, 2, or 3" };

        if (gmmax < 1.f)
            throw std::string{ "gmmax must be greater than or equal to 1.0" };

        if (opt < 0 || opt > 4)
            throw std::string{ "opt must be 0, 1, 2, 3, or 4" };

        const unsigned numThreads = vsapi->getCoreInfo(core)->numThreads;
        d->blur.reserve(numThreads);
        d->gradient.reserve(numThreads);
        d->direction.reserve(numThreads);
        d->found.reserve(numThreads);

        selectFunctions(opt, d.get());

        if (d->vi->format->sampleType == stInteger) {
            d->peak = (1 << d->vi->format->bitsPerSample) - 1;
            const float scale = d->peak / 255.f;
            d->t_h *= scale;
            d->t_l *= scale;
        } else {
            d->t_h /= 255.f;
            d->t_l /= 255.f;

            for (int plane = 0; plane < d->vi->format->numPlanes; plane++) {
                if (plane == 0 || d->vi->format->colorFamily == cmRGB)
                    d->offset[plane] = 0.f;
                else
                    d->offset[plane] = 0.5f;
            }
        }

        for (int plane = 0; plane < d->vi->format->numPlanes; plane++) {
            if (d->process[plane]) {
                if (sigmaH[plane]) {
                    d->weightsH[plane] = gaussianWeights(sigmaH[plane], d->radiusH[plane]);
                    if (!d->weightsH[plane])
                        throw std::string{ "malloc failure (weightsH)" };

                    const int width = d->vi->width >> (plane ? d->vi->format->subSamplingW : 0);
                    const std::string planeOrder{ plane == 0 ? "first" : (plane == 1 ? "second" : "third") };
                    if (width < d->radiusH[plane] + 1)
                        throw std::string{ "the " + planeOrder + " plane's width must be greater than or equal to " + std::to_string(d->radiusH[plane] + 1) + " for specified sigma" };
                }

                if (sigmaV[plane]) {
                    d->weightsV[plane] = gaussianWeights(sigmaV[plane], d->radiusV[plane]);
                    if (!d->weightsV[plane])
                        throw std::string{ "malloc failure (weightsV)" };

                    const int height = d->vi->height >> (plane ? d->vi->format->subSamplingH : 0);
                    const std::string planeOrder{ plane == 0 ? "first" : (plane == 1 ? "second" : "third") };
                    if (height < d->radiusV[plane] + 1)
                        throw std::string{ "the " + planeOrder + " plane's height must be greater than or equal to " + std::to_string(d->radiusV[plane] + 1) + " for specified sigma_v" };
                }
            }
        }

        d->radiusAlign = (std::max({ d->radiusH[0], d->radiusH[1], d->radiusH[2], 1 }) + d->vectorSize - 1) & -d->vectorSize;

        d->magnitude = 255.f / gmmax;
    } catch (const std::string & error) {
        vsapi->setError(out, ("TCanny: " + error).c_str());
        vsapi->freeNode(d->node);
        return;
    }

    vsapi->createFilter(in, out, "TCanny", tcannyInit, tcannyGetFrame, tcannyFree, fmParallel, 0, d.release(), core);
}

//////////////////////////////////////////
// Init

#ifdef HAVE_OPENCL
extern void VS_CC tcannyclCreate(const VSMap *in, VSMap *out, void *userData, VSCore *core, const VSAPI *vsapi);
#endif

VS_EXTERNAL_API(void) VapourSynthPluginInit(VSConfigPlugin configFunc, VSRegisterFunction registerFunc, VSPlugin *plugin) {
    configFunc("com.holywu.tcanny", "tcanny", "Build an edge map using canny edge detection", VAPOURSYNTH_API_VERSION, 1, plugin);

    registerFunc("TCanny",
                 "clip:clip;"
                 "sigma:float[]:opt;"
                 "sigma_v:float[]:opt;"
                 "t_h:float:opt;"
                 "t_l:float:opt;"
                 "mode:int:opt;"
                 "op:int:opt;"
                 "gmmax:float:opt;"
                 "opt:int:opt;"
                 "planes:int[]:opt;",
                 tcannyCreate, nullptr, plugin);

#ifdef HAVE_OPENCL
    registerFunc("TCannyCL",
                 "clip:clip;"
                 "sigma:float[]:opt;"
                 "sigma_v:float[]:opt;"
                 "t_h:float:opt;"
                 "t_l:float:opt;"
                 "mode:int:opt;"
                 "op:int:opt;"
                 "gmmax:float:opt;"
                 "device:int:opt;"
                 "list_device:int:opt;"
                 "info:int:opt;"
                 "planes:int[]:opt;",
                 tcannyclCreate, nullptr, plugin);
#endif
}
