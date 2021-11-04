# TCanny
Builds an edge map using canny edge detection.

Ported from AviSynth plugin http://bengal.missouri.edu/~kes25c/


## Usage
    tcanny.TCanny(vnode clip[, float[] sigma=1.5, float[] sigma_v=sigma, float t_h=8.0, float t_l=1.0, int mode=0, int op=1, float scale=1.0, int opt=0, int[] planes=[0, 1, 2]])

- clip: Clip to process. Any format with either integer sample type of 8-16 bit depth or float sample type of 32 bit depth is supported.

- sigma: Standard deviation of horizontal gaussian blur. Setting to 0 disables gaussian blur. If a single `sigma` is specified, it will be used for all planes. If two `sigma` are given then the second value will be used for the third plane as well.

- sigma_v: Standard deviation of vertical gaussian blur.

- t_h: High gradient magnitude threshold for hysteresis.

- t_l: Low gradient magnitude threshold for hysteresis.

- mode: Sets output format.
  - -1 = gaussian blur only
  - 0 = thresholded edge map (MAX_PIXEL_VALUE for edge, 0 for non-edge)
  - 1 = gradient magnitude map

- op: Sets the operator for edge detection.
  - 0 = the operator used in tritical's original filter
  - 1 = the Prewitt operator whose use is proposed by P. Zhou et al. [1]
  - 2 = the Sobel operator
  - 3 = the Scharr operator
  - 4 = the Kroon operator
  - 5 = the Kirsch operator

- scale: Multiplies the gradient by `scale`. This can be used to increase or decrease the intensity of edges in the output.

- opt: Sets which cpu optimizations to use.
  - 0 = auto detect
  - 1 = use c
  - 2 = use sse2
  - 3 = use avx2
  - 4 = use avx512

- planes: Sets which planes will be processed. Any unprocessed planes will be simply copied.


[1]: Zhou, P., Ye, W., & Wang, Q. (2011). An Improved Canny Algorithm for Edge Detection. Journal of Computational Information Systems, 7(5), 1516-1523.


## Compilation
```
meson build
ninja -C build
ninja -C build install
```
