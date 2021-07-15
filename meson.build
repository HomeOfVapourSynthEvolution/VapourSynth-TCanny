project('TCanny', 'cpp',
  default_options: ['buildtype=release', 'warning_level=2', 'b_lto=true', 'b_ndebug=if-release', 'cpp_std=c++17'],
  license: 'GPL-2.0-or-later',
  meson_version: '>=0.51.0',
  version: '12'
)

cxx = meson.get_compiler('cpp')

gcc_syntax = cxx.get_argument_syntax() == 'gcc'

if get_option('buildtype') == 'release'
  add_project_arguments(gcc_syntax ? ['-fno-math-errno', '-fno-trapping-math'] : ['/GS-', '/fp:except-'], language: 'cpp')
endif

if gcc_syntax
  vapoursynth_dep = dependency('vapoursynth').partial_dependency(compile_args: true, includes: true)
  install_dir = vapoursynth_dep.get_variable(pkgconfig: 'libdir') / 'vapoursynth'
else
  vapoursynth_dep = dependency('', required: false)
  install_dir = get_option('libdir') / 'vapoursynth'
endif

sources = [
  'TCanny/shared.hpp',
  'TCanny/TCanny.cpp',
  'TCanny/TCanny.hpp'
]

libs = []

if host_machine.cpu_family().startswith('x86')
  project_args = ['-DVS_TARGET_CPU_X86']
  if gcc_syntax
    project_args += ['-mfpmath=sse', '-msse2']
  endif
  add_project_arguments(project_args, language: 'cpp')

  sources += [
    'TCanny/TCanny_SSE2.cpp',
    'TCanny/vectorclass/instrset.h',
    'TCanny/vectorclass/instrset_detect.cpp',
    'TCanny/vectorclass/vectorclass.h',
    'TCanny/vectorclass/vectorf128.h',
    'TCanny/vectorclass/vectorf256.h',
    'TCanny/vectorclass/vectorf256e.h',
    'TCanny/vectorclass/vectori128.h',
    'TCanny/vectorclass/vectori256.h',
    'TCanny/vectorclass/vectori256e.h',
    'TCanny/vectorclass/vectormath_common.h',
    'TCanny/vectorclass/vectormath_trig.h'
  ]

  libs += static_library('avx', 'TCanny/TCanny_AVX.cpp',
    cpp_args: gcc_syntax ? ['-mavx'] : ['/arch:AVX'],
    dependencies: vapoursynth_dep,
    gnu_symbol_visibility: 'hidden'
  )

  libs += static_library('avx2', 'TCanny/TCanny_AVX2.cpp',
    cpp_args: gcc_syntax ? ['-mavx2', '-mfma'] : ['/arch:AVX2'],
    dependencies: vapoursynth_dep,
    gnu_symbol_visibility: 'hidden'
  )
endif

shared_module('tcanny', sources,
  dependencies: vapoursynth_dep,
  link_with: libs,
  install: true,
  install_dir: install_dir,
  gnu_symbol_visibility: 'hidden'
)
