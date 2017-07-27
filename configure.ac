AC_INIT([TCanny], [8], [https://github.com/HomeOfVapourSynthEvolution/VapourSynth-TCanny/issues], [TCanny], [https://github.com/HomeOfVapourSynthEvolution/VapourSynth-TCanny/])

: ${CXXFLAGS=""}

AM_INIT_AUTOMAKE([foreign no-dist-gzip dist-xz subdir-objects no-define])
AM_SILENT_RULES([yes])

LT_INIT([disable-static win32-dll])

AC_PROG_CXX

AC_CANONICAL_HOST


X86="false"

AS_CASE(
        [$host_cpu],
        [i?86], [BITS="32" X86="true"],
        [x86_64], [BITS="64" X86="true"],
)

AS_CASE(
        [$host_os],
        [cygwin*|mingw*],
        [
         AS_IF(
               [test "x$BITS" = "x32"],
               [
                AC_SUBST([PLUGINLDFLAGS], ["-Wl,--kill-at"])
               ]
         )
        ]
)

AS_IF(
      [test "x$X86" = "xtrue"],
      [
       AC_DEFINE([VS_TARGET_CPU_X86])

       AC_SUBST([MFLAGS], ["-mfpmath=sse -msse2"])
      ]
)


AC_ARG_ENABLE([opencl], AS_HELP_STRING([--enable-opencl], [Enable the TCannyCL filter. (default=yes)]))
AS_IF(
      [test "x$enable_opencl" != "xno"],
      [
       AC_DEFINE([HAVE_OPENCL])

       AC_SUBST([OPENCLLDFLAGS], ["-lOpenCL"])
      ]
)


AM_CONDITIONAL([VS_TARGET_CPU_X86], [test "x$X86" = "xtrue"])
AM_CONDITIONAL([OPENCL], [test "x$enable_opencl" != "xno"])


PKG_CHECK_MODULES([VapourSynth], [vapoursynth])

AC_CONFIG_FILES([Makefile])
AC_OUTPUT
