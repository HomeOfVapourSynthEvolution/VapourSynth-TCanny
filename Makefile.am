warning_flags = -Wall -Wextra -Wno-unused-parameter -Wshadow
common_cflags = -O3 -ffast-math $(MFLAGS) $(warning_flags)
AM_CXXFLAGS = -std=c++14 $(common_cflags)

AM_CPPFLAGS = $(VapourSynth_CFLAGS)

lib_LTLIBRARIES = libtcanny.la

libtcanny_la_SOURCES = TCanny/TCanny.cpp \
                       TCanny/vectorclass/instrset.h \
                       TCanny/vectorclass/instrset_detect.cpp

if VS_TARGET_CPU_X86
libtcanny_la_SOURCES += TCanny/TCanny_SSE2.cpp \
                        TCanny/vectorclass/vectorclass.h \
                        TCanny/vectorclass/vectorf128.h \
                        TCanny/vectorclass/vectorf256.h \
                        TCanny/vectorclass/vectorf256e.h \
                        TCanny/vectorclass/vectori128.h \
                        TCanny/vectorclass/vectori256.h \
                        TCanny/vectorclass/vectori256e.h \
                        TCanny/vectorclass/vectormath_common.h \
                        TCanny/vectorclass/vectormath_trig.h


noinst_LTLIBRARIES = libavx.la libavx2.la


libavx_la_SOURCES = TCanny/TCanny_AVX.cpp

libavx_la_CXXFLAGS = $(AM_CXXFLAGS) -mavx


libavx2_la_SOURCES = TCanny/TCanny_AVX2.cpp

libavx2_la_CXXFLAGS = $(AM_CXXFLAGS) -mfma -mavx2


libtcanny_la_LIBADD = libavx.la libavx2.la
endif

libtcanny_la_LDFLAGS = -no-undefined -avoid-version $(PLUGINLDFLAGS)
