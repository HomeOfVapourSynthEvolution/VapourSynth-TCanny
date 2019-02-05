#ifdef HAVE_OPENCL
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

#include <clocale>
#include <cstdio>

#include "shared.hpp"
#include "TCanny.cl"

#define BOOST_COMPUTE_DEBUG_KERNEL_COMPILATION
#define BOOST_COMPUTE_HAVE_THREAD_LOCAL
#define BOOST_COMPUTE_THREAD_SAFE
#define BOOST_COMPUTE_USE_OFFLINE_CACHE
#include <boost/compute/core.hpp>
namespace compute = boost::compute;

struct TCannyCLData {
    VSNodeRef * node;
    const VSVideoInfo * vi;
    int mode;
    bool process[3];
    int radiusH[3], radiusV[3];
    unsigned peak;
    float offset[3], lower[3], upper[3];
    compute::device gpu;
    compute::context ctx;
    compute::program program;
    compute::buffer weightsH[3], weightsV[3];
    cl_image_format clImageFormat;
    std::unordered_map<std::thread::id, compute::command_queue> queue;
    std::unordered_map<std::thread::id, compute::kernel> copyPlane, gaussianBlurH, gaussianBlurV, detectEdge, nonMaximumSuppression, hysteresis, outputGB, binarizeCE, discretizeGM;
    std::unordered_map<std::thread::id, compute::image2d> src[3], dst[3], blur[3], gradient[3], direction[3];
    std::unordered_map<std::thread::id, compute::buffer> buffer, found;
};

static void VS_CC tcannyclInit(VSMap *in, VSMap *out, void **instanceData, VSNode *node, VSCore *core, const VSAPI *vsapi) {
    TCannyCLData * d = static_cast<TCannyCLData *>(*instanceData);
    vsapi->setVideoInfo(d->vi, 1, node);
}

static const VSFrameRef *VS_CC tcannyclGetFrame(int n, int activationReason, void **instanceData, void **frameData, VSFrameContext *frameCtx, VSCore *core, const VSAPI *vsapi) {
    TCannyCLData * d = static_cast<TCannyCLData *>(*instanceData);

    if (activationReason == arInitial) {
        vsapi->requestFrameFilter(n, d->node, frameCtx);
    } else if (activationReason == arAllFramesReady) {
        const VSFrameRef * src = vsapi->getFrameFilter(n, d->node, frameCtx);
        const VSFrameRef * fr[] = { d->process[0] ? nullptr : src, d->process[1] ? nullptr : src, d->process[2] ? nullptr : src };
        const int pl[] = { 0, 1, 2 };
        VSFrameRef * dst = vsapi->newVideoFrame2(d->vi->format, d->vi->width, d->vi->height, fr, pl, src, core);

        try {
            auto threadId = std::this_thread::get_id();

            if (!d->queue.count(threadId)) {
                d->queue.emplace(threadId, compute::command_queue{ d->ctx, d->gpu });

                if (d->vi->format->sampleType == stInteger) {
                    d->copyPlane.emplace(threadId, d->program.create_kernel("copyPlane_uint"));
                    d->gaussianBlurV.emplace(threadId, d->program.create_kernel("gaussianBlurV_uint"));
                    d->outputGB.emplace(threadId, d->program.create_kernel("outputGB_uint"));
                    d->binarizeCE.emplace(threadId, d->program.create_kernel("binarizeCE_uint"));
                    d->discretizeGM.emplace(threadId, d->program.create_kernel("discretizeGM_uint"));
                } else {
                    d->copyPlane.emplace(threadId, d->program.create_kernel("copyPlane_float"));
                    d->gaussianBlurV.emplace(threadId, d->program.create_kernel("gaussianBlurV_float"));
                    d->outputGB.emplace(threadId, d->program.create_kernel("outputGB_float"));
                    d->binarizeCE.emplace(threadId, d->program.create_kernel("binarizeCE_float"));
                    d->discretizeGM.emplace(threadId, d->program.create_kernel("discretizeGM_float"));
                }
                d->gaussianBlurH.emplace(threadId, d->program.create_kernel("gaussianBlurH"));
                d->detectEdge.emplace(threadId, d->program.create_kernel("detectEdge"));
                d->nonMaximumSuppression.emplace(threadId, d->program.create_kernel("nonMaximumSuppression"));
                d->hysteresis.emplace(threadId, d->program.create_kernel("hysteresis"));

                size_t width = d->vi->width;
                size_t height = d->vi->height;

                d->src[0].emplace(threadId, compute::image2d{ d->ctx, width, height, compute::image_format{ d->clImageFormat }, CL_MEM_READ_ONLY | CL_MEM_HOST_WRITE_ONLY });
                d->dst[0].emplace(threadId, compute::image2d{ d->ctx, width, height, compute::image_format{ d->clImageFormat }, CL_MEM_WRITE_ONLY | CL_MEM_HOST_READ_ONLY });
                d->blur[0].emplace(threadId, compute::image2d{ d->ctx, width, height, compute::image_format{ CL_R, CL_FLOAT }, CL_MEM_READ_WRITE | CL_MEM_HOST_NO_ACCESS });
                d->gradient[0].emplace(threadId, compute::image2d{ d->ctx, width, height, compute::image_format{ CL_R, CL_FLOAT }, CL_MEM_READ_WRITE | CL_MEM_HOST_NO_ACCESS });
                d->direction[0].emplace(threadId, compute::image2d{ d->ctx, width, height, compute::image_format{ CL_R, CL_UNSIGNED_INT8 }, CL_MEM_READ_WRITE | CL_MEM_HOST_NO_ACCESS });

                if (d->vi->format->subSamplingW || d->vi->format->subSamplingH) {
                    width >>= d->vi->format->subSamplingW;
                    height >>= d->vi->format->subSamplingH;

                    for (int i = 1; i <= 2; i++) {
                        d->src[i].emplace(threadId, compute::image2d{ d->ctx, width, height, compute::image_format{ d->clImageFormat }, CL_MEM_READ_ONLY | CL_MEM_HOST_WRITE_ONLY });
                        d->dst[i].emplace(threadId, compute::image2d{ d->ctx, width, height, compute::image_format{ d->clImageFormat }, CL_MEM_WRITE_ONLY | CL_MEM_HOST_READ_ONLY });
                        d->blur[i].emplace(threadId, compute::image2d{ d->ctx, width, height, compute::image_format{ CL_R, CL_FLOAT }, CL_MEM_READ_WRITE | CL_MEM_HOST_NO_ACCESS });
                        d->gradient[i].emplace(threadId, compute::image2d{ d->ctx, width, height, compute::image_format{ CL_R, CL_FLOAT }, CL_MEM_READ_WRITE | CL_MEM_HOST_NO_ACCESS });
                        d->direction[i].emplace(threadId, compute::image2d{ d->ctx, width, height, compute::image_format{ CL_R, CL_UNSIGNED_INT8 }, CL_MEM_READ_WRITE | CL_MEM_HOST_NO_ACCESS });
                    }
                } else {
                    for (int i = 1; i <= 2; i++) {
                        d->src[i].emplace(threadId, d->src[0].at(threadId));
                        d->dst[i].emplace(threadId, d->dst[0].at(threadId));
                        d->blur[i].emplace(threadId, d->blur[0].at(threadId));
                        d->gradient[i].emplace(threadId, d->gradient[0].at(threadId));
                        d->direction[i].emplace(threadId, d->direction[0].at(threadId));
                    }
                }

                d->buffer.emplace(threadId, compute::buffer{ d->ctx, d->vi->width * d->vi->height * sizeof(cl_float), CL_MEM_READ_WRITE | CL_MEM_HOST_NO_ACCESS });
                d->found.emplace(threadId, compute::buffer{ d->ctx, d->vi->width * d->vi->height * sizeof(cl_uchar), CL_MEM_READ_WRITE | CL_MEM_HOST_NO_ACCESS });
            }

            auto queue = d->queue.at(threadId);
            auto copyPlane = d->copyPlane.at(threadId);
            auto gaussianBlurV = d->gaussianBlurV.at(threadId);
            auto gaussianBlurH = d->gaussianBlurH.at(threadId);
            auto detectEdge = d->detectEdge.at(threadId);
            auto nonMaximumSuppression = d->nonMaximumSuppression.at(threadId);
            auto hysteresis = d->hysteresis.at(threadId);
            auto outputGB = d->outputGB.at(threadId);
            auto binarizeCE = d->binarizeCE.at(threadId);
            auto discretizeGM = d->discretizeGM.at(threadId);
            auto buffer = d->buffer.at(threadId);
            auto found = d->found.at(threadId);

            for (int plane = 0; plane < d->vi->format->numPlanes; plane++) {
                if (d->process[plane]) {
                    const size_t width = vsapi->getFrameWidth(src, plane);
                    const size_t height = vsapi->getFrameHeight(src, plane);
                    const size_t stride = vsapi->getStride(src, plane);
                    const uint8_t * srcp = vsapi->getReadPtr(src, plane);
                    uint8_t * dstp = vsapi->getWritePtr(dst, plane);

                    auto srcImage = d->src[plane].at(threadId);
                    auto dstImage = d->dst[plane].at(threadId);
                    auto blurImage = d->blur[plane].at(threadId);
                    auto gradientImage = d->gradient[plane].at(threadId);
                    auto directionImage = d->direction[plane].at(threadId);

                    const size_t origin[] = { 0, 0, 0 };
                    const size_t region[] = { width, height, 1 };
                    const size_t globalWorkSize[] = { width, height };

                    queue.enqueue_write_image(srcImage, origin, region, srcp, stride);

                    if (d->radiusV[plane]) {
                        gaussianBlurV.set_args(srcImage, d->radiusH[plane] ? gradientImage : blurImage, d->weightsV[plane], d->radiusV[plane], d->offset[plane]);
                        queue.enqueue_nd_range_kernel(gaussianBlurV, 2, nullptr, globalWorkSize, nullptr);
                    } else {
                        copyPlane.set_args(srcImage, d->radiusH[plane] ? gradientImage : blurImage, d->offset[plane]);
                        queue.enqueue_nd_range_kernel(copyPlane, 2, nullptr, globalWorkSize, nullptr);
                    }

                    if (d->radiusH[plane]) {
                        gaussianBlurH.set_args(gradientImage, blurImage, d->weightsH[plane], d->radiusH[plane]);
                        queue.enqueue_nd_range_kernel(gaussianBlurH, 2, nullptr, globalWorkSize, nullptr);
                    }

                    if (d->mode != -1) {
                        detectEdge.set_args(blurImage, gradientImage, directionImage);
                        queue.enqueue_nd_range_kernel(detectEdge, 2, nullptr, globalWorkSize, nullptr);

                        if (d->mode == 0) {
                            nonMaximumSuppression.set_args(directionImage, gradientImage, buffer);
                            queue.enqueue_nd_range_kernel(nonMaximumSuppression, 2, nullptr, globalWorkSize, nullptr);

                            constexpr cl_uchar pattern = 0;
                            queue.enqueue_fill_buffer(found, &pattern, sizeof(cl_uchar), 0, width * height * sizeof(cl_uchar));

                            const size_t paddedGlobalWorkSize[] = { (width + 7) & -8, (height + 7) & -8 };
                            const size_t localWorkSize[] = { 8, 8 };

                            hysteresis.set_args(buffer, found, static_cast<int>(width), static_cast<int>(height));
                            queue.enqueue_nd_range_kernel(hysteresis, 2, nullptr, paddedGlobalWorkSize, localWorkSize);
                        }
                    }

                    if (d->mode == -1) {
                        outputGB.set_args(blurImage, dstImage, d->peak, d->offset[plane]);
                        queue.enqueue_nd_range_kernel(outputGB, 2, nullptr, globalWorkSize, nullptr);
                    } else if (d->mode == 0) {
                        binarizeCE.set_args(buffer, dstImage, d->peak, d->lower[plane], d->upper[plane]);
                        queue.enqueue_nd_range_kernel(binarizeCE, 2, nullptr, globalWorkSize, nullptr);
                    } else {
                        discretizeGM.set_args(gradientImage, dstImage, d->peak, d->offset[plane]);
                        queue.enqueue_nd_range_kernel(discretizeGM, 2, nullptr, globalWorkSize, nullptr);
                    }

                    queue.enqueue_read_image(dstImage, origin, region, stride, 0, dstp);
                }
            }
        } catch (const compute::opencl_error & error) {
            vsapi->setFilterError(("TCannyCL: " + error.error_string()).c_str(), frameCtx);
            vsapi->freeFrame(src);
            vsapi->freeFrame(dst);
            return nullptr;
        }

        vsapi->freeFrame(src);
        return dst;
    }

    return nullptr;
}

static void VS_CC tcannyclFree(void *instanceData, VSCore *core, const VSAPI *vsapi) {
    TCannyCLData * d = static_cast<TCannyCLData *>(instanceData);
    vsapi->freeNode(d->node);
    delete d;
}

void VS_CC tcannyclCreate(const VSMap *in, VSMap *out, void *userData, VSCore *core, const VSAPI *vsapi) {
    std::unique_ptr<TCannyCLData> d = std::make_unique<TCannyCLData>();
    int err;

    d->node = vsapi->propGetNode(in, "clip", 0, nullptr);
    d->vi = vsapi->getVideoInfo(d->node);

    try {
        if (!isConstantFormat(d->vi) ||
            (d->vi->format->sampleType == stInteger && d->vi->format->bitsPerSample > 16) ||
            (d->vi->format->sampleType == stFloat && d->vi->format->bitsPerSample != 32))
            throw std::string{ "only constant format 8-16 bit integer and 32 bit float input supported" };

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

        float t_h = static_cast<float>(vsapi->propGetFloat(in, "t_h", 0, &err));
        if (err)
            t_h = 8.f;

        float t_l = static_cast<float>(vsapi->propGetFloat(in, "t_l", 0, &err));
        if (err)
            t_l = 1.f;

        d->mode = int64ToIntS(vsapi->propGetInt(in, "mode", 0, &err));

        int op = int64ToIntS(vsapi->propGetInt(in, "op", 0, &err));
        if (err)
            op = 1;

        float gmmax = static_cast<float>(vsapi->propGetFloat(in, "gmmax", 0, &err));
        if (err)
            gmmax = 50.f;

        int device = int64ToIntS(vsapi->propGetInt(in, "device", 0, &err));
        if (err)
            device = -1;

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

        if (t_l >= t_h)
            throw std::string{ "t_h must be greater than t_l" };

        if (d->mode < -1 || d->mode > 1)
            throw std::string{ "mode must be -1, 0, or 1" };

        if (op < 0 || op > 3)
            throw std::string{ "op must be 0, 1, 2, or 3" };

        if (gmmax < 1.f)
            throw std::string{ "gmmax must be greater than or equal to 1.0" };

        if (device >= static_cast<int>(compute::system::device_count()))
            throw std::string{ "device index out of range" };

        if (!!vsapi->propGetInt(in, "list_device", 0, &err)) {
            const auto devices = compute::system::devices();
            std::string text;

            for (size_t i = 0; i < devices.size(); i++)
                text += std::to_string(i) + ": " + devices[i].name() + " (" + devices[i].platform().name() + ")" + "\n";

            VSMap * args = vsapi->createMap();
            vsapi->propSetNode(args, "clip", d->node, paReplace);
            vsapi->freeNode(d->node);
            vsapi->propSetData(args, "text", text.c_str(), -1, paReplace);

            VSMap * ret = vsapi->invoke(vsapi->getPluginById("com.vapoursynth.text", core), "Text", args);
            if (vsapi->getError(ret)) {
                vsapi->setError(out, vsapi->getError(ret));
                vsapi->freeMap(args);
                vsapi->freeMap(ret);
                return;
            }

            d->node = vsapi->propGetNode(ret, "clip", 0, nullptr);
            vsapi->freeMap(args);
            vsapi->freeMap(ret);
            vsapi->propSetNode(out, "clip", d->node, paReplace);
            vsapi->freeNode(d->node);
            return;
        }

        d->gpu = (device < 0) ? compute::system::default_device() : compute::system::devices().at(device);
        d->ctx = compute::context{ d->gpu };

        if (!!vsapi->propGetInt(in, "info", 0, &err)) {
            std::string text{ "=== Device Info ===\n" };
            text += "Name: " + d->gpu.get_info<CL_DEVICE_NAME>() + "\n";
            text += "Vendor: " + d->gpu.get_info<CL_DEVICE_VENDOR>() + "\n";
            text += "Profile: " + d->gpu.get_info<CL_DEVICE_PROFILE>() + "\n";
            text += "Version: " + d->gpu.get_info<CL_DEVICE_VERSION>() + "\n";
            text += "Global Memory Size: " + std::to_string(d->gpu.get_info<CL_DEVICE_GLOBAL_MEM_SIZE>() / 1024 / 1024) + " MB\n";
            text += "Local Memory Size: " + std::to_string(d->gpu.get_info<CL_DEVICE_LOCAL_MEM_SIZE>() / 1024) + " KB\n";
            text += "Local Memory Type: " + std::string{ d->gpu.get_info<CL_DEVICE_LOCAL_MEM_TYPE>() == CL_LOCAL ? "CL_LOCAL" : "CL_GLOBAL" } + "\n";
            text += "Image Support: " + std::string{ d->gpu.get_info<CL_DEVICE_IMAGE_SUPPORT>() ? "CL_TRUE" : "CL_FALSE" } + "\n";
            text += "1D Image Max Buffer Size: " + std::to_string(d->gpu.get_info<size_t>(CL_DEVICE_IMAGE_MAX_BUFFER_SIZE)) + "\n";
            text += "2D Image Max Width: " + std::to_string(d->gpu.get_info<CL_DEVICE_IMAGE2D_MAX_WIDTH>()) + "\n";
            text += "2D Image Max Height: " + std::to_string(d->gpu.get_info<CL_DEVICE_IMAGE2D_MAX_HEIGHT>()) + "\n";
            text += "Max Constant Arguments: " + std::to_string(d->gpu.get_info<CL_DEVICE_MAX_CONSTANT_ARGS>()) + "\n";
            text += "Max Constant Buffer Size: " + std::to_string(d->gpu.get_info<CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE>() / 1024) + " KB\n";
            text += "Max Work-group Size: " + std::to_string(d->gpu.get_info<CL_DEVICE_MAX_WORK_GROUP_SIZE>()) + "\n";
            const auto MAX_WORK_ITEM_SIZES = d->gpu.get_info<CL_DEVICE_MAX_WORK_ITEM_SIZES>();
            text += "Max Work-item Sizes: (" + std::to_string(MAX_WORK_ITEM_SIZES[0]) + ", " + std::to_string(MAX_WORK_ITEM_SIZES[1]) + ", " + std::to_string(MAX_WORK_ITEM_SIZES[2]) + ")";

            VSMap * args = vsapi->createMap();
            vsapi->propSetNode(args, "clip", d->node, paReplace);
            vsapi->freeNode(d->node);
            vsapi->propSetData(args, "text", text.c_str(), -1, paReplace);

            VSMap * ret = vsapi->invoke(vsapi->getPluginById("com.vapoursynth.text", core), "Text", args);
            if (vsapi->getError(ret)) {
                vsapi->setError(out, vsapi->getError(ret));
                vsapi->freeMap(args);
                vsapi->freeMap(ret);
                return;
            }

            d->node = vsapi->propGetNode(ret, "clip", 0, nullptr);
            vsapi->freeMap(args);
            vsapi->freeMap(ret);
            vsapi->propSetNode(out, "clip", d->node, paReplace);
            vsapi->freeNode(d->node);
            return;
        }

        const unsigned numThreads = vsapi->getCoreInfo(core)->numThreads;
        d->queue.reserve(numThreads);
        d->copyPlane.reserve(numThreads);
        d->gaussianBlurV.reserve(numThreads);
        d->gaussianBlurH.reserve(numThreads);
        d->detectEdge.reserve(numThreads);
        d->nonMaximumSuppression.reserve(numThreads);
        d->hysteresis.reserve(numThreads);
        d->outputGB.reserve(numThreads);
        d->binarizeCE.reserve(numThreads);
        d->discretizeGM.reserve(numThreads);
        d->buffer.reserve(numThreads);
        d->found.reserve(numThreads);
        for (int i = 0; i < 3; i++) {
            d->src[i].reserve(numThreads);
            d->dst[i].reserve(numThreads);
            d->blur[i].reserve(numThreads);
            d->gradient[i].reserve(numThreads);
            d->direction[i].reserve(numThreads);
        }

        if (d->vi->format->sampleType == stInteger) {
            d->peak = (1 << d->vi->format->bitsPerSample) - 1;
            const float scale = d->peak / 255.f;
            t_h *= scale;
            t_l *= scale;
        } else {
            t_h /= 255.f;
            t_l /= 255.f;

            for (int plane = 0; plane < d->vi->format->numPlanes; plane++) {
                if (plane == 0 || d->vi->format->colorFamily == cmRGB) {
                    d->offset[plane] = 0.f;
                    d->lower[plane] = 0.f;
                    d->upper[plane] = 1.f;
                } else {
                    d->offset[plane] = 0.5f;
                    d->lower[plane] = -0.5f;
                    d->upper[plane] = 0.5f;
                }
            }
        }

        for (int plane = 0; plane < d->vi->format->numPlanes; plane++) {
            if (d->process[plane]) {
                if (sigmaH[plane]) {
                    float * weightsH = gaussianWeights(sigmaH[plane], d->radiusH[plane]);
                    if (!weightsH)
                        throw std::string{ "malloc failure (weightsH)" };

                    d->weightsH[plane] = compute::buffer{ d->ctx, (d->radiusH[plane] * 2 + 1) * sizeof(cl_float), CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR | CL_MEM_HOST_NO_ACCESS, weightsH };

                    delete[] weightsH;
                }

                if (sigmaV[plane]) {
                    float * weightsV = gaussianWeights(sigmaV[plane], d->radiusV[plane]);
                    if (!weightsV)
                        throw std::string{ "malloc failure (weightsV)" };

                    d->weightsV[plane] = compute::buffer{ d->ctx, (d->radiusV[plane] * 2 + 1) * sizeof(cl_float), CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR | CL_MEM_HOST_NO_ACCESS, weightsV };

                    delete[] weightsV;
                }
            }
        }

        if (d->vi->format->bytesPerSample == 1)
            d->clImageFormat = { CL_R, CL_UNSIGNED_INT8 };
        else if (d->vi->format->bytesPerSample == 2)
            d->clImageFormat = { CL_R, CL_UNSIGNED_INT16 };
        else
            d->clImageFormat = { CL_R, CL_FLOAT };

        try {
            std::setlocale(LC_ALL, "C");
            char buf[100];
            std::string options{ "-cl-denorms-are-zero -cl-fast-relaxed-math -Werror" };
            std::snprintf(buf, 100, "%.20ff", t_h);
            options += " -D T_H=" + std::string{ buf };
            std::snprintf(buf, 100, "%.20ff", t_l);
            options += " -D T_L=" + std::string{ buf };
            options += " -D MODE=" + std::to_string(d->mode);
            options += " -D OP=" + std::to_string(op);
            std::snprintf(buf, 100, "%.20ff", 255.f / gmmax);
            options += " -D MAGNITUDE=" + std::string{ buf };
            std::setlocale(LC_ALL, "");
            d->program = compute::program::build_with_source(source, d->ctx, options);
        } catch (const compute::opencl_error & error) {
            throw error.error_string() + "\n" + d->program.build_log();
        }
    } catch (const std::string & error) {
        vsapi->setError(out, ("TCannyCL: " + error).c_str());
        vsapi->freeNode(d->node);
        return;
    } catch (const compute::no_device_found & error) {
        vsapi->setError(out, (std::string{ "TCannyCL: " } + error.what()).c_str());
        vsapi->freeNode(d->node);
        return;
    } catch (const compute::opencl_error & error) {
        vsapi->setError(out, ("TCannyCL: " + error.error_string()).c_str());
        vsapi->freeNode(d->node);
        return;
    }

    vsapi->createFilter(in, out, "TCannyCL", tcannyclInit, tcannyclGetFrame, tcannyclFree, fmParallel, 0, d.release(), core);
}
#endif
