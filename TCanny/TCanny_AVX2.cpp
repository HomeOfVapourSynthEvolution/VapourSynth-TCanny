#ifdef TCANNY_X86
#include "TCanny.h"

template<typename pixel_t>
static void gaussianBlur(const pixel_t* __srcp, float* temp, float* dstp, const int width, const int height,
                         const ptrdiff_t srcStride, const ptrdiff_t dstStride, const int radiusH, const int radiusV,
                         const float* weightsH, const float* weightsV) noexcept {
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
                    sum = mul_add(srcp, weightsV[v], sum);
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
                          const ptrdiff_t srcStride, const ptrdiff_t dstStride, const int radius, const float* weights) noexcept {
    weights += radius;

    for (auto y{ 0 }; y < height; y++) {
        for (auto x{ 0 }; x < width; x += Vec8f().size()) {
            if constexpr (std::is_same_v<pixel_t, uint8_t>)
                to_float(Vec8i().load_8uc(_srcp + x)).store_a(temp + x);
            else if constexpr (std::is_same_v<pixel_t, uint16_t>)
                to_float(Vec8i().load_8us(_srcp + x)).store_a(temp + x);
            else
                Vec8f().load_a(_srcp + x).store_a(temp + x);
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
                          const int radius, const float* weights) noexcept {
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
                    sum = mul_add(srcp, weights[v], sum);
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
static void copyPlane(const pixel_t* srcp, float* dstp, const int width, const int height, const ptrdiff_t srcStride, const ptrdiff_t dstStride) noexcept {
    for (auto y{ 0 }; y < height; y++) {
        for (auto x{ 0 }; x < width; x += Vec8f().size()) {
            if constexpr (std::is_same_v<pixel_t, uint8_t>)
                to_float(Vec8i().load_8uc(srcp + x)).store_nt(dstp + x);
            else if constexpr (std::is_same_v<pixel_t, uint16_t>)
                to_float(Vec8i().load_8us(srcp + x)).store_nt(dstp + x);
            else
                Vec8f().load_a(srcp + x).store_nt(dstp + x);
        }

        srcp += srcStride;
        dstp += dstStride;
    }
}

static void detectEdge(float* blur, float* gradient, int* direction, const int width, const int height, const ptrdiff_t stride, const ptrdiff_t bgStride,
                       const int mode, const int op, const float scale) noexcept {
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

        for (auto x{ 0 }; x < width; x += Vec8f().size()) {
            Vec8f gx, gy;

            if (op != FDOG) {
                auto& c1{ Vec8f().load(prev + x - 1) };
                auto& c2{ Vec8f().load_a(prev + x) };
                auto& c3{ Vec8f().load(prev + x + 1) };
                auto& c4{ Vec8f().load(cur + x - 1) };
                auto& c6{ Vec8f().load(cur + x + 1) };
                auto& c7{ Vec8f().load(next + x - 1) };
                auto& c8{ Vec8f().load_a(next + x) };
                auto& c9{ Vec8f().load(next + x + 1) };

                switch (op) {
                case TRITICAL:
                    gx = c6 - c4;
                    gy = c2 - c8;
                    break;
                case PREWITT:
                    gx = (c3 + c6 + c9 - c1 - c4 - c7) * 0.5f;
                    gy = (c1 + c2 + c3 - c7 - c8 - c9) * 0.5f;
                    break;
                case SOBEL:
                    gx = c3 + mul_add(2.0f, c6, c9) - c1 - mul_add(2.0f, c4, c7);
                    gy = c1 + mul_add(2.0f, c2, c3) - c7 - mul_add(2.0f, c8, c9);
                    break;
                case SCHARR:
                    gx = mul_add(3.0f, c3 + c9, 10.0f * c6) - mul_add(3.0f, c1 + c7, 10.0f * c4);
                    gy = mul_add(3.0f, c1 + c3, 10.0f * c2) - mul_add(3.0f, c7 + c9, 10.0f * c8);
                    break;
                case KROON:
                    gx = mul_add(17.0f, c3 + c9, 61.0f * c6) - mul_add(17.0f, c1 + c7, 61.0f * c4);
                    gy = mul_add(17.0f, c1 + c3, 61.0f * c2) - mul_add(17.0f, c7 + c9, 61.0f * c8);
                    break;
                case KIRSCH:
                    auto g1{ mul_sub(5.0f, c1 + c2 + c3, 3.0f * (c4 + c6 + c7 + c8 + c9)) };
                    auto g2{ mul_sub(5.0f, c1 + c2 + c4, 3.0f * (c3 + c6 + c7 + c8 + c9)) };
                    auto g3{ mul_sub(5.0f, c1 + c4 + c7, 3.0f * (c2 + c3 + c6 + c8 + c9)) };
                    auto g4{ mul_sub(5.0f, c4 + c7 + c8, 3.0f * (c1 + c2 + c3 + c6 + c9)) };
                    auto g5{ mul_sub(5.0f, c7 + c8 + c9, 3.0f * (c1 + c2 + c3 + c4 + c6)) };
                    auto g6{ mul_sub(5.0f, c6 + c8 + c9, 3.0f * (c1 + c2 + c3 + c4 + c7)) };
                    auto g7{ mul_sub(5.0f, c3 + c6 + c9, 3.0f * (c1 + c2 + c4 + c7 + c8)) };
                    auto g8{ mul_sub(5.0f, c2 + c3 + c6, 3.0f * (c1 + c4 + c7 + c8 + c9)) };
                    auto g{ max(max(max(abs(g1), abs(g2)), max(abs(g3), abs(g4))), max(max(abs(g5), abs(g6)), max(abs(g7), abs(g8)))) };
                    (g * scale).store_nt(gradient + x);
                    break;
                }
            } else {
                auto& c1{ Vec8f().load(prev2 + x - 2) };
                auto& c2{ Vec8f().load(prev2 + x - 1) };
                auto& c3{ Vec8f().load(prev2 + x) };
                auto& c4{ Vec8f().load(prev2 + x + 1) };
                auto& c5{ Vec8f().load(prev2 + x + 2) };
                auto& c6{ Vec8f().load(prev + x - 2) };
                auto& c7{ Vec8f().load(prev + x - 1) };
                auto& c8{ Vec8f().load(prev + x) };
                auto& c9{ Vec8f().load(prev + x + 1) };
                auto& c10{ Vec8f().load(prev + x + 2) };
                auto& c11{ Vec8f().load(cur + x - 2) };
                auto& c12{ Vec8f().load(cur + x - 1) };
                auto& c14{ Vec8f().load(cur + x + 1) };
                auto& c15{ Vec8f().load(cur + x + 2) };
                auto& c16{ Vec8f().load(next + x - 2) };
                auto& c17{ Vec8f().load(next + x - 1) };
                auto& c18{ Vec8f().load(next + x) };
                auto& c19{ Vec8f().load(next + x + 1) };
                auto& c20{ Vec8f().load(next + x + 2) };
                auto& c21{ Vec8f().load(next2 + x - 2) };
                auto& c22{ Vec8f().load(next2 + x - 1) };
                auto& c23{ Vec8f().load(next2 + x) };
                auto& c24{ Vec8f().load(next2 + x + 1) };
                auto& c25{ Vec8f().load(next2 + x + 2) };

                gx = c5 + c25 + c4 + c24 + mul_add(2.0f, c10 + c20 + c9 + c19, 3.0f * (c15 + c14))
                    - c2 - c22 - c1 - c21 - mul_add(2.0f, c7 + c17 + c6 + c16, 3.0f * (c12 + c11));
                gy = c1 + c5 + c6 + c10 + mul_add(2.0f, c2 + c4 + c7 + c9, 3.0f * (c3 + c8))
                    - c16 - c20 - c21 - c25 - mul_add(2.0f, c17 + c19 + c22 + c24, 3.0f * (c18 + c23));
            }

            if (op != KIRSCH) {
                gx *= scale;
                gy *= scale;
                sqrt(mul_add(gx, gx, gy * gy)).store_nt(gradient + x);
            }

            if (mode == 0) {
                auto dr{ atan2(gy, gx) };
                dr = if_add(dr < 0.0f, dr, M_PIF);

                auto bin{ truncatei(mul_add(dr, 4.0f * M_1_PIF, 0.5f)) };
                select(bin >= 4, zero_si256(), bin).store_nt(direction + x);
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

template<typename pixel_t, bool clampFP = true>
static void discretizeGM(const float* _srcp, pixel_t* dstp, const int width, const int height, const ptrdiff_t srcStride, const ptrdiff_t dstStride,
                         const int peak) noexcept {
    for (auto y{ 0 }; y < height; y++) {
        for (auto x{ 0 }; x < width; x += Vec8f().size()) {
            auto& srcp{ Vec8f().load_a(_srcp + x) };

            if constexpr (std::is_same_v<pixel_t, uint8_t>) {
                auto result{ compress_saturated_s2u(compress_saturated(truncatei(srcp + 0.5f), zero_si256()), zero_si256()).get_low() };
                result.storel(dstp + x);
            } else if constexpr (std::is_same_v<pixel_t, uint16_t>) {
                auto result{ compress_saturated_s2u(truncatei(srcp + 0.5f), zero_si256()).get_low() };
                min(result, peak).store_nt(dstp + x);
            } else if constexpr (clampFP) {
                min(max(srcp, 0.0f), 1.0f).store_nt(dstp + x);
            } else {
                srcp.store_nt(dstp + x);
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
            else if (d->mode == 1)
                discretizeGM(gradient, dstp, width, height, bgStride, stride, d->peak);
            else
                discretizeGM<pixel_t, false>(blur, dstp, width, height, bgStride, stride, d->peak);
        }
    }
}

template void filter_avx2<uint8_t>(const VSFrame* src, VSFrame* dst, const TCannyData* const VS_RESTRICT d, const VSAPI* vsapi) noexcept;
template void filter_avx2<uint16_t>(const VSFrame* src, VSFrame* dst, const TCannyData* const VS_RESTRICT d, const VSAPI* vsapi) noexcept;
template void filter_avx2<float>(const VSFrame* src, VSFrame* dst, const TCannyData* const VS_RESTRICT d, const VSAPI* vsapi) noexcept;
#endif
