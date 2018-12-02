project('TCanny', 'cpp',
  default_options : ['buildtype=release', 'b_ndebug=if-release', 'cpp_std=c++14'],
  meson_version : '>=0.48.0',
  version : '11'
)

add_project_arguments('-ffast-math', language : 'cpp')

sources = [
  'TCanny/TCanny.cpp',
  'TCanny/TCanny.hpp',
  'TCanny/vectorclass/instrset.h',
  'TCanny/vectorclass/instrset_detect.cpp'
]

vapoursynth_dep = dependency('vapoursynth').partial_dependency(compile_args : true, includes : true)

deps = [vapoursynth_dep]

libs = []

if host_machine.cpu_family().startswith('x86')
  add_project_arguments('-DVS_TARGET_CPU_X86', '-mfpmath=sse', '-msse2', language : 'cpp')

  sources += [
    'TCanny/TCanny_SSE2.cpp',
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

  if get_option('opencl')
    add_project_arguments('-DHAVE_OPENCL', language : 'cpp')

    sources += [
      'TCanny/TCannyCL.cpp'
    ]

    opencl_dep = dependency('OpenCL', required : false)
    if not opencl_dep.found()
      opencl_dep = meson.get_compiler('cpp').find_library('OpenCL')
    endif

    boost_dep = dependency('boost', modules : ['filesystem', 'system'])

    deps += [opencl_dep, boost_dep]
  endif

  libs += static_library('avx', 'TCanny/TCanny_AVX.cpp',
    dependencies : [vapoursynth_dep],
    cpp_args : ['-mavx'],
    gnu_symbol_visibility : 'hidden'
  )

  libs += static_library('avx2', 'TCanny/TCanny_AVX2.cpp',
    dependencies : [vapoursynth_dep],
    cpp_args : ['-mavx2', '-mfma'],
    gnu_symbol_visibility : 'hidden'
  )
endif

shared_module('tcanny', sources,
  dependencies : deps,
  link_with : libs,
  install : true,
  install_dir : join_paths(vapoursynth_dep.get_pkgconfig_variable('libdir'), 'vapoursynth'),
  gnu_symbol_visibility : 'hidden'
)
