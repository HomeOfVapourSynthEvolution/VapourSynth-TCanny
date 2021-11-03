#ifdef TCANNY_X86
#include "TCanny.h"

template<typename pixel_t>
static void gaussianBlur(const pixel_t* __srcp, float* temp, float* dstp, const int width, const int height,
                         const ptrdiff_t srcStride, const ptrdiff_t dstStride, const int radiusH, const int radiusV,
                         const float* weightsH, const float* weightsV, const float offset) noexcept {
    auto diameter{ radiusV * 2 + 1 };
    auto _srcp{ std::make_unique<const pixel_t* []>(diameter) };

    _srcp[radiusV] = __srcp;
    for (auto i{ 1 }; i <= radiusV; i++)
        _srcp[radiusV - i] = _srcp[radiusV + i] = _srcp[radiusV] + srcStride * i;

    weightsH += radiusH;

    for (auto y{ 0 }; y < height; y++) {
        for (auto x{ 0 }; x < width; x += Vec8f().size()) {
            auto sum{ zero_8f() };

            for (auto v{ 0 }; v < diameter; v++) {
                if constexpr (std::is_same_v<pixel_t, uint8_t>) {
                    auto srcp{ to_float(Vec8i().load_8uc(_srcp[v] + x)) };
                    sum = mul_add(srcp, weightsV[v], sum);
                } else if constexpr (std::is_same_v<pixel_t, uint16_t>) {
                    auto srcp{ to_float(Vec8i().load_8us(_srcp[v] + x)) };
                    sum = mul_add(srcp, weightsV[v], sum);
                } else {
                    auto& srcp{ Vec8f().load_a(_srcp[v] + x) };
                    sum = mul_add(srcp + offset, weightsV[v], sum);
                }
            }

            sum.store_a(temp + x);
        }

        for (auto i{ 1 }; i <= radiusH; i++) {
            temp[-i] = temp[i];
            temp[width - 1 + i] = temp[width - 1 - i];
        }

        for (auto x{ 0 }; x < width; x += Vec8f().size()) {
            auto sum{ zero_8f() };

            for (auto v{ -radiusH }; v <= radiusH; v++) {
                auto& srcp{ Vec8f().load(temp + x + v) };
                sum = mul_add(srcp, weightsH[v], sum);
            }

            sum.store_nt(dstp + x);
        }

        for (auto i{ 0 }; i < diameter - 1; i++)
            _srcp[i] = _srcp[i + 1];
        _srcp[diameter - 1] += (y < height - 1 - radiusV) ? srcStride : -srcStride;
        dstp += dstStride;
    }
}

template<typename pixel_t>
static void gaussianBlurH(const pixel_t* _srcp, float* temp, float* dstp, const int width, const int height,
                          const ptrdiff_t srcStride, const ptrdiff_t dstStride, const int radius, const float* weights, const float offset) noexcept {
    weights += radius;

    for (auto y{ 0 }; y < height; y++) {
        for (auto x{ 0 }; x < width; x += Vec8f().size()) {
            if constexpr (std::is_same_v<pixel_t, uint8_t>)
                to_float(Vec8i().load_8uc(_srcp + x)).store_a(temp + x);
            else if constexpr (std::is_same_v<pixel_t, uint16_t>)
                to_float(Vec8i().load_8us(_srcp + x)).store_a(temp + x);
            else
                (Vec8f().load_a(_srcp + x) + offset).store_a(temp + x);
        }

        for (auto i{ 1 }; i <= radius; i++) {
            temp[-i] = temp[i];
            temp[width - 1 + i] = temp[width - 1 - i];
        }

        for (auto x{ 0 }; x < width; x += Vec8f().size()) {
            auto sum{ zero_8f() };

            for (auto v{ -radius }; v <= radius; v++) {
                auto& srcp{ Vec8f().load(temp + x + v) };
                sum = mul_add(srcp, weights[v], sum);
            }

            sum.store_nt(dstp + x);
        }

        _srcp += srcStride;
        dstp += dstStride;
    }
}

template<typename pixel_t>
static void gaussianBlurV(const pixel_t* __srcp, float* dstp, const int width, const int height, const ptrdiff_t srcStride, const ptrdiff_t dstStride,
                          const int radius, const float* weights, const float offset) noexcept {
    auto diameter{ radius * 2 + 1 };
    auto _srcp{ std::make_unique<const pixel_t* []>(diameter) };

    _srcp[radius] = __srcp;
    for (auto i{ 1 }; i <= radius; i++)
        _srcp[radius - i] = _srcp[radius + i] = _srcp[radius] + srcStride * i;

    for (auto y{ 0 }; y < height; y++) {
        for (auto x{ 0 }; x < width; x += Vec8f().size()) {
            auto sum{ zero_8f() };

            for (auto v{ 0 }; v < diameter; v++) {
                if constexpr (std::is_same_v<pixel_t, uint8_t>) {
                    auto srcp{ to_float(Vec8i().load_8uc(_srcp[v] + x)) };
                    sum = mul_add(srcp, weights[v], sum);
                } else if constexpr (std::is_same_v<pixel_t, uint16_t>) {
                    auto srcp{ to_float(Vec8i().load_8us(_srcp[v] + x)) };
                    sum = mul_add(srcp, weights[v], sum);
                } else {
                    auto& srcp{ Vec8f().load_a(_srcp[v] + x) };
                    sum = mul_add(srcp + offset, weights[v], sum);
                }
            }

            sum.store_nt(dstp + x);
        }

        for (auto i{ 0 }; i < diameter - 1; i++)
            _srcp[i] = _srcp[i + 1];
        _srcp[diameter - 1] += (y < height - 1 - radius) ? srcStride : -srcStride;
        dstp += dstStride;
    }
}

template<typename pixel_t>
static void copyPlane(const pixel_t* srcp, float* dstp, const int width, const int height, const ptrdiff_t srcStride, const ptrdiff_t dstStride,
                      const float offset) noexcept {
    for (auto y{ 0 }; y < height; y++) {
        for (auto x{ 0 }; x < width; x += Vec8f().size()) {
            if constexpr (std::is_same_v<pixel_t, uint8_t>)
                to_float(Vec8i().load_8uc(srcp + x)).store_nt(dstp + x);
            else if constexpr (std::is_same_v<pixel_t, uint16_t>)
                to_float(Vec8i().load_8us(srcp + x)).store_nt(dstp + x);
            else
                (Vec8f().load_a(srcp + x) + offset).store_nt(dstp + x);
        }

        srcp += srcStride;
        dstp += dstStride;
    }
}

static void detectEdge(float* blur, float* gradient, int* direction, const int width, const int height, const ptrdiff_t stride, const ptrdiff_t bgStride,
                       const int mode, const int op) noexcept {
    auto prev{ blur + bgStride };
    auto cur{ blur };
    auto next{ blur + bgStride };

    cur[-1] = cur[1];
    cur[width] = cur[width - 2];

    for (auto y{ 0 }; y < height; y++) {
        next[-1] = next[1];
        next[width] = next[width - 2];

        for (auto x{ 0 }; x < width; x += Vec8f().size()) {
            auto& topLeft{ Vec8f().load(prev + x - 1) };
            auto& top{ Vec8f().load_a(prev + x) };
            auto& topRight{ Vec8f().load(prev + x + 1) };
            auto& left{ Vec8f().load(cur + x - 1) };
            auto& right{ Vec8f().load(cur + x + 1) };
            auto& bottomLeft{ Vec8f().load(next + x - 1) };
            auto& bottom{ Vec8f().load_a(next + x) };
            auto& bottomRight{ Vec8f().load(next + x + 1) };

            Vec8f gx, gy;

            switch (op) {
            case TRITICAL:
                gx = right - left;
                gy = top - bottom;
                break;
            case PREWITT:
                gx = (topRight + right + bottomRight - topLeft - left - bottomLeft) * 0.5f;
                gy = (topLeft + top + topRight - bottomLeft - bottom - bottomRight) * 0.5f;
                break;
            case SOBEL:
                gx = topRight + mul_add(2.0f, right, bottomRight) - topLeft - mul_add(2.0f, left, bottomLeft);
                gy = topLeft + mul_add(2.0f, top, topRight) - bottomLeft - mul_add(2.0f, bottom, bottomRight);
                break;
            case SCHARR:
                gx = mul_add(3.0f, topRight, mul_add(10.0f, right, 3.0f * bottomRight)) - mul_add(3.0f, topLeft, mul_add(10.0f, left, 3.0f * bottomLeft));
                gy = mul_add(3.0f, topLeft, mul_add(10.0f, top, 3.0f * topRight)) - mul_add(3.0f, bottomLeft, mul_add(10.0f, bottom, 3.0f * bottomRight));
                break;
            case KROON:
                gx = mul_add(17.0f, topRight, mul_add(61.0f, right, 17.0f * bottomRight)) - mul_add(17.0f, topLeft, mul_add(61.0f, left, 17.0f * bottomLeft));
                gy = mul_add(17.0f, topLeft, mul_add(61.0f, top, 17.0f * topRight)) - mul_add(17.0f, bottomLeft, mul_add(61.0f, bottom, 17.0f * bottomRight));
                break;
            case ROBINSON: {
                auto g1{ topRight + mul_add(2.0f, right, bottomRight) - topLeft - mul_add(2.0f, left, bottomLeft) };
                auto g2{ top + mul_add(2.0f, topRight, right) - left - mul_add(2.0f, bottomLeft, bottom) };
                auto g3{ topLeft + mul_add(2.0f, top, topRight) - bottomLeft - mul_add(2.0f, bottom, bottomRight) };
                auto g4{ left + mul_add(2.0f, topLeft, top) - bottom - mul_add(2.0f, bottomRight, right) };
                max(max(abs(g1), abs(g2)), max(abs(g3), abs(g4))).store_nt(gradient + x);
                break;
            }
            case KIRSCH: {
                auto g1{ mul_sub(5.0f, topLeft + top + topRight, 3.0f * (left + right + bottomLeft + bottom + bottomRight)) };
                auto g2{ mul_sub(5.0f, topLeft + top + left, 3.0f * (topRight + right + bottomLeft + bottom + bottomRight)) };
                auto g3{ mul_sub(5.0f, topLeft + left + bottomLeft, 3.0f * (top + topRight + right + bottom + bottomRight)) };
                auto g4{ mul_sub(5.0f, left + bottomLeft + bottom, 3.0f * (topLeft + top + topRight + right + bottomRight)) };
                auto g5{ mul_sub(5.0f, bottomLeft + bottom + bottomRight, 3.0f * (topLeft + top + topRight + left + right)) };
                auto g6{ mul_sub(5.0f, right + bottom + bottomRight, 3.0f * (topLeft + top + topRight + left + bottomLeft)) };
                auto g7{ mul_sub(5.0f, topRight + right + bottomRight, 3.0f * (topLeft + top + left + bottomLeft + bottom)) };
                auto g8{ mul_sub(5.0f, top + topRight + right, 3.0f * (topLeft + left + bottomLeft + bottom + bottomRight)) };
                max(max(max(abs(g1), abs(g2)), max(abs(g3), abs(g4))), max(max(abs(g5), abs(g6)), max(abs(g7), abs(g8)))).store_nt(gradient + x);
                break;
            }
            }

            if (op < ROBINSON)
                sqrt(mul_add(gx, gx, gy * gy)).store_nt(gradient + x);

            if (mode == 0) {
                auto dr{ atan2(gy, gx) };
                dr = if_add(dr < 0.0f, dr, M_PIF);

                auto bin{ truncatei(mul_add(dr, 4.0f * M_1_PIF, 0.5f)) };
                select(bin >= 4, zero_si256(), bin).store_nt(direction + x);
            }
        }

        prev = cur;
        cur = next;
        next += (y < height - 2) ? bgStride : -bgStride;
        gradient += bgStride;
        direction += stride;
    }
}

static void nonMaximumSuppression(const int* _direction, float* _gradient, float* blur, const int width, const int height,
                                  const ptrdiff_t stride, const ptrdiff_t bgStride, const int radiusAlign) noexcept {
    _gradient[-1] = _gradient[1];
    _gradient[-1 + bgStride * (height - 1)] = _gradient[1 + bgStride * (height - 1)];
    _gradient[width] = _gradient[width - 2];
    _gradient[width + bgStride * (height - 1)] = _gradient[width - 2 + bgStride * (height - 1)];
    std::copy_n(_gradient - radiusAlign + bgStride, width + radiusAlign * 2, _gradient - radiusAlign - bgStride);
    std::copy_n(_gradient - radiusAlign + bgStride * (height - 2), width + radiusAlign * 2, _gradient - radiusAlign + bgStride * height);

    for (auto y{ 0 }; y < height; y++) {
        for (auto x{ 0 }; x < width; x += Vec8f().size()) {
            auto& direction{ Vec8i().load_a(_direction + x) };

            auto mask{ Vec8fb(direction == 0) };
            auto gradient{ max(Vec8f().load(_gradient + x + 1), Vec8f().load(_gradient + x - 1)) };
            auto result{ gradient & mask };

            mask = Vec8fb(direction == 1);
            gradient = max(Vec8f().load(_gradient + x - bgStride + 1), Vec8f().load(_gradient + x + bgStride - 1));
            result |= gradient & mask;

            mask = Vec8fb(direction == 2);
            gradient = max(Vec8f().load_a(_gradient + x - bgStride), Vec8f().load_a(_gradient + x + bgStride));
            result |= gradient & mask;

            mask = Vec8fb(direction == 3);
            gradient = max(Vec8f().load(_gradient + x - bgStride - 1), Vec8f().load(_gradient + x + bgStride + 1));
            result |= gradient & mask;

            gradient = Vec8f().load_a(_gradient + x);
            select(gradient >= result, gradient, fltLowest).store_nt(blur + x);
        }

        _direction += stride;
        _gradient += bgStride;
        blur += bgStride;
    }
}

template<typename pixel_t>
static void outputGB(const float* _srcp, pixel_t* dstp, const int width, const int height, const ptrdiff_t srcStride, const ptrdiff_t dstStride,
                     const int peak, const float offset) noexcept {
    for (auto y{ 0 }; y < height; y++) {
        for (auto x{ 0 }; x < width; x += Vec8f().size()) {
            auto& srcp{ Vec8f().load_a(_srcp + x) };

            if constexpr (std::is_same_v<pixel_t, uint8_t>) {
                auto result{ compress_saturated_s2u(compress_saturated(truncatei(srcp + 0.5f), zero_si256()), zero_si256()).get_low() };
                result.storel(dstp + x);
            } else if constexpr (std::is_same_v<pixel_t, uint16_t>) {
                auto result{ compress_saturated_s2u(truncatei(srcp + 0.5f), zero_si256()).get_low() };
                min(result, peak).store_nt(dstp + x);
            } else {
                (srcp - offset).store_nt(dstp + x);
            }
        }

        _srcp += srcStride;
        dstp += dstStride;
    }
}

template<typename pixel_t>
static void binarizeCE(const float* _srcp, pixel_t* dstp, const int width, const int height, const ptrdiff_t srcStride, const ptrdiff_t dstStride,
                       const int peak) noexcept {
    for (auto y{ 0 }; y < height; y++) {
        for (auto x{ 0 }; x < width; x += Vec8f().size()) {
            auto& srcp{ Vec8f().load_a(_srcp + x) };

            if constexpr (std::is_same_v<pixel_t, uint8_t>) {
                auto mask{ Vec16cb(compress_saturated(compress_saturated(Vec8ib(srcp == fltMax), zero_si256()), zero_si256()).get_low()) };
                select(mask, Vec16uc(255), zero_si128()).storel(dstp + x);
            } else if constexpr (std::is_same_v<pixel_t, uint16_t>) {
                auto mask{ Vec8sb(compress_saturated(Vec8ib(srcp == fltMax), zero_si256()).get_low()) };
                select(mask, Vec8us(peak), zero_si128()).store_nt(dstp + x);
            } else {
                auto mask{ srcp == fltMax };
                select(mask, Vec8f(1.0f), Vec8f(0.0f)).store_nt(dstp + x);
            }
        }

        _srcp += srcStride;
        dstp += dstStride;
    }
}

template<typename pixel_t>
static void discretizeGM(const float* _srcp, pixel_t* dstp, const int width, const int height, const ptrdiff_t srcStride, const ptrdiff_t dstStride,
                         const float gmmax, const int peak) noexcept {
    for (auto y{ 0 }; y < height; y++) {
        for (auto x{ 0 }; x < width; x += Vec8f().size()) {
            auto& srcp{ Vec8f().load_a(_srcp + x) };

            if constexpr (std::is_same_v<pixel_t, uint8_t>) {
                auto result{ compress_saturated_s2u(compress_saturated(truncatei(mul_add(srcp, gmmax, 0.5f)), zero_si256()), zero_si256()).get_low() };
                result.storel(dstp + x);
            } else if constexpr (std::is_same_v<pixel_t, uint16_t>) {
                auto result{ compress_saturated_s2u(truncatei(mul_add(srcp, gmmax, 0.5f)), zero_si256()).get_low() };
                min(result, peak).store_nt(dstp + x);
            } else {
                (srcp * gmmax).store_nt(dstp + x);
            }
        }

        _srcp += srcStride;
        dstp += dstStride;
    }
}

template<typename pixel_t>
void filter_avx2(const VSFrame* src, VSFrame* dst, const TCannyData* const VS_RESTRICT d, const VSAPI* vsapi) noexcept {
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
                             d->weightsH[plane].get(), d->weightsV[plane].get(), d->offset[plane]);
            else if (d->radiusH[plane])
                gaussianBlurH(srcp, gradient, blur, width, height, stride, bgStride, d->radiusH[plane], d->weightsH[plane].get(), d->offset[plane]);
            else if (d->radiusV[plane])
                gaussianBlurV(srcp, blur, width, height, stride, bgStride, d->radiusV[plane], d->weightsV[plane].get(), d->offset[plane]);
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
                discretizeGM(gradient, dstp, width, height, bgStride, stride, d->gmmax, d->peak);
        }
    }
}

template void filter_avx2<uint8_t>(const VSFrame* src, VSFrame* dst, const TCannyData* const VS_RESTRICT d, const VSAPI* vsapi) noexcept;
template void filter_avx2<uint16_t>(const VSFrame* src, VSFrame* dst, const TCannyData* const VS_RESTRICT d, const VSAPI* vsapi) noexcept;
template void filter_avx2<float>(const VSFrame* src, VSFrame* dst, const TCannyData* const VS_RESTRICT d, const VSAPI* vsapi) noexcept;
#endif
