

"""

Module for reading and writing image data.

The code for this module was originally taken from the `qpaint_analysis` project, by Rasmus S. Sorensen.


Reading Nikon ND2 image data:
* `PIMS_ND2` - PIMS extension for reading Nikon ND2 data files.
    https://github.com/soft-matter/pims_nd2
* `ND2Reader` - Pure Python library for reading NIS Elements ND2 images and metadata.
    Note: New version also uses PIMS framework; old/original version did not.
    https://github.com/rbnvrw/nd2reader - active fork, see also http://www.lighthacking.nl/nd2reader/
    https://github.com/jimrybarski/nd2reader - original developer,
    > " This library is no longer being developed here. "
    > " I am no longer supporting this library, as my lab has discovered Micro-Manager
    >   and found it to be a far superior application for acquiring microscope data."
* `python-bioformats` - Python bindings for the ubiquitous BioFormats Java library.
    https://github.com/CellProfiler/python-bioformats/

Reading Olympus VSI/OEX/ETS image data:
* BioFormats - Java library, via javabridge.
* python-bioformats - Python wrapper around the BioFormats Java library.
* [OpenSlide](https://github.com/openslide/openslide) - C library from Carnegie Mellon University, see issue #143.
* https://biop.epfl.ch/TOOL_VSI_Reader.html
* https://github.com/pskeshu/microscoper
* https://github.com/matsojr22/vsi_to_tif_converter - ImageJ macro converting VSI to TIF.


Microscopy-specific image libraries:
* PIMS - Pythonn Image Sequence. Can use BioFormats via `jpype`.
    Built-in readers include: ImageSequence, TiffStack, Video, and Bioformats.



General image libraries:
* PIL / Pillow
* Tifffile.py by C. Gohlke
* PyLSM
* PyLibTiff - python ctypes wrapper around libtiff C library.
* imread
* PIMS (Python Image Sequence) - https://github.com/soft-matter/pims


"""

import os
import numpy as np
import PIL
import pims
from .image_utils import norm_pix_values


def print_nd2_import_error():
    print("Could not import either nd2reader libraries. ND2 import will not be available.")


def import_nd2reader():
    try:
        # Try to import pims_nd2 package:
        from pims import ND2_Reader
    except ImportError:
        # Try to import the pure-python nd2reader (still PIMS-based!)
        try:
            from nd2reader import ND2Reader as ND2_Reader
        except ImportError:
            print_nd2_import_error()
            ND2_Reader = None
    return ND2_Reader


def read_image_stream(filename, ext=None):
    """General-purpose image reader function.
    Currently mostly just a wrapper around PIMS, but here for reference and in case you want to provide
    ability to read custom formats not supported by PIMS (or PIMS extensions).
    Will determine image type from filename and select the best method to read the image data from file.

    Args:
        filename: Path of image/stack to read.
            Note: The image data is usually not read into memory straight away but rather opened as a file stream.
            The image stream can be read by iterating over the reader.

    Returns:
        frames_reader: An iterable stream of image frames as numpy 2D arrays.

    """
    basename = os.path.basename(filename)
    if ext is None:
        stem, ext = os.path.splitext(basename)
        ext = ext.strip('.').lower()
    if ext == 'nd2':
        reader = import_nd2reader()
    elif ext in ('tif', 'tiff'):
        reader = pims.TiffStack
    # Consider separate case for .RAW images
    else:
        # Just use `pims.open` and see what happens.
        # `pims.open` will try all registered readers one by one until one of them doesn't raise an exception.
        reader = pims.open
    return reader(filename)


def save_image_sequence(
    frames, *,
    fnfmt="image-%(i)05d.%(filetype)s", filetype='png', fmttype='%s',
    vmin=0, vmax=None, dtype='uint8',
    verbose=0
):
    """ Save a stack/stream of image frames as separate image files.

    Args:
        frames:
        fnfmt:
        filetype:
        fmttype:
        vmin:
        vmax:
        dtype:
        verbose:

    Returns:

    """
    vmin, vmax = 0, np.percentile(frames, q=99.9)
    if verbose:
        print("frames: length=%s, min=%s, max=%s:" % (len(frames), frames.min(), frames.max()))
        print(" - Using pixel value range: %s - %s" % (vmin, vmax))
    for i, frame in enumerate(frames):
        if verbose > 1:
            print("\n" if verbose > 2 else "\r", end="")
        # normalize:
        # print("  frame min=%s, max=%s, dtype=%s..." % (frame.min(), frame.max(), frame.dtype))
        frame = norm_pix_values(frame, vmin=vmin, vmax=vmax, dtype='uint8')  # frame.astype('uint8')  # np.uint8(frame)
        # print("  frame min=%s, max=%s, dtype=%s..." % (frame.min(), frame.max(), frame.dtype), end="")
        # fname = fnfmt.format(i=i, filetype=filetype)
        if "%" in fmttype:
            fname = fnfmt % dict(i=i, filetype=filetype)
        elif 'ffmpeg' in fmttype.lower():
            # same format as used for ffmpeg: `$ ffmpeg -i image-%05d.png`
            fname = fnfmt % (i, )
        else:
            fname = fnfmt.format(i=i, filetype=filetype)
        with open(fname, 'wb') as fd:
            if verbose > 1:
                print("Saving frame #%s to file %s..." % (i, fname), end='')
            # Using PIL:
            PIL.Image.fromarray(frame).save(fd, filetype)
    if verbose:
        print("\n - Done!")


class FrameRange:
    """A simple class that imitates a standard PIMS reader, but with a frame-range selection applied."""

    def __init__(self, reader, frame_range):
        self.reader = reader
        if not isinstance(frame_range, range):
            self.frame_range = range(*frame_range)
        else:
            self.frame_range = frame_range
        if self.frame_range.stop > len(reader):
            print("Note: Requested frame_range %s-%s, but frame stack is only has %s frames in total." %
                  (self.frame_range.start, self.frame_range.stop, len(reader)))
            self.frame_range = range(self.frame_range.start, len(reader), self.frame_range.step)
        self.n_frames = ((self.frame_range.stop - self.frame_range.start) // self.frame_range.step)

    def frame_generator(self):
        return (self.reader[i] for i in self.frame_range)

    def __len__(self):
        return self.n_frames

    def __iter__(self):
        return self.frame_generator()

    def min(self):
        return self.reader[0].min()

    def max(self):
        return self.reader[0].max()

    def percentile(self, q):
        return np.percentile(self.reader[0], q)

    def __getitem__(self, item):
        return self.reader[item]
