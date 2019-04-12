# Copyright 2018 Rasmus Scholer Sorensen, <rasmusscholer@gmail.com>

"""

The code for this module was originally part of the `qpaint_analysis` project, by Rasmus S. Sorensen.

"""

import numpy as np


def norm_pix_values(frame, vmin=0, vmax=None, scale=None, dtype=None, output=None):
    """Normalize pixel values within a given value range.

    Args:
        frame: input numpy array
        vmin: lower bound of the input value range, i.e. "black" pixels.
            Values below `vmin` are truncated to `vmin`.
            Default is 0.
        vmax: upper bound of the input value range, i.e. "white" pixels.
            Values above `vmax` are truncated to `vmax`.
            Default is the maximum value of the input frame.
        scale: The maximum value of the output.
            Default is 1 for floating point arrays and np.iinfo(dtype).max for integers.
        dtype: dtype for the output array.
        output: Use this numpy array for the output.

    Returns:
        numpy array of same shape as the input frame.

    Examples:
        # Convert 16-bit uint input to 8-bit uint (typical for making images and video):
        >>> frame = np.random.rand(256, 256, dtype='uint16')
        >>> vmax = np.percentile(frame, 99)
        >>> norm_pix_values(frame, vmin=0, vmax=vmax, scale=None, dtype='uint8')

    Alternative normalization methods / implementations:
        cv2.convertScaleAbs()
        sklearn.preprocessing.scale()

    """
    if vmax is None:
        vmax = frame.max()
    if dtype is None:
        dtype = frame.dtype
    if scale is None and np.issubdtype(dtype, np.integer):
        scale = np.iinfo(dtype).max
        clip_below, clip_above = 0, scale
    else:
        scale = 1.
        clip_below, clip_above = 0., 1.
    if output is None:
        # output = np.asarray(frame, dtype=dtype)
        # Uh, asarray just ensures that input is array.
        # If input is already an array (with same dtype, if specified), input array is returned unchanged/uncopied.
        output = np.empty_like(frame, dtype=dtype)
    # Any way of doing this (↓) more efficiently for integers?
    # Lets say scale is 255 (uint8 max), vmin is 0, vmax = 10200, and max(frame) = 20000 (uint16 input).
    # `(frame - vmin) // vmax` would be 2 (extreme loss of precision.
    # `scale * (frame - vmin)` would give interger overflow (wrap around from 250000).
    # `scale // vmax` is 0.
    # `vmax // scale` is 40...
    # `(frame - vmin) // (vmax // scale)` should work, assuming vmax >> scale (e.g. vmax > 10*scale)
    # What about at fmax? `(fmax - vmin) // (vmax // scale)` is 20000 // 40 = 500
    if np.issubdtype(frame.dtype, np.integer) and vmax > 10*scale:
        print(" int", clip_below, clip_above, end=" ")
        output[:] = np.clip((frame - vmin) // (vmax // scale), clip_below, clip_above)
    else:
        output[:] = scale * np.clip((frame - vmin) / vmax, clip_below, clip_above)
    return output
