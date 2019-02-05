Description
===========

Builds an edge map using canny edge detection.

Ported from AviSynth plugin http://bengal.missouri.edu/~kes25c/


Usage
=====

    tcanny.TCanny(clip clip[, float[] sigma=1.5, float[] sigma_v=sigma, float t_h=8.0, float t_l=1.0, int mode=0, int op=1, float gmmax=50.0, int opt=0, int[] planes=[0, 1, 2]])

* clip: Clip to process. Any planar format with either integer sample type of 8-16 bit depth or float sample type of 32 bit depth is supported.

* sigma: Standard deviation of horizontal gaussian blur. If a single `sigma` is specified, it will be used for all planes. If two `sigma` are given then the second value will be used for the third plane as well.

* sigma_v: Standard deviation of vertical gaussian blur.

* t_h: High gradient magnitude threshold for hysteresis.

* t_l: Low gradient magnitude threshold for hysteresis.

* mode: Sets output format.
  * -1 = gaussian blur only
  * 0 = thresholded edge map (2^bitdepth-1 for edge, 0 for non-edge)
  * 1 = gradient magnitude map

* op: Sets the operator for edge detection.
  * 0 = the operator used in tritical's original filter
  * 1 = the operator proposed by P. Zhou et al.
  * 2 = the Sobel operator
  * 3 = the Scharr operator

* gmmax: Used for scaling gradient magnitude into [0, 2^bitdepth-1] for `mode=1`.

* opt: Sets which cpu optimizations to use.
  * 0 = auto detect
  * 1 = use c
  * 2 = use sse2
  * 3 = use avx
  * 4 = use avx2

* planes: Sets which planes will be processed. Any unprocessed planes will be simply copied.

---

    tcanny.TCannyCL(clip clip[, float[] sigma=1.5, float[] sigma_v=sigma, float t_h=8.0, float t_l=1.0, int mode=0, int op=1, float gmmax=50.0, int device=-1, bint list_device=False, bint info=False, int[] planes=[0, 1, 2]])

* device: Sets target OpenCL device. Use `list_device` to get the index of the available devices. By default the default device is selected.

* list_device: Whether to draw the devices list on the frame.

* info: Whether to draw the OpenCL-related info on the frame.


Compilation
===========

Requires `Boost` unless specify `-Dopencl=false`.

```
meson build
ninja -C build
ninja -C build install
```
