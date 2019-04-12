# Copyright 2018 Rasmus Scholer Sorensen, <rasmusscholer@gmail.com>

"""

Module for reading/decoding and writing/encoding video files/streams.

This file was originally part of the `qpaint_analysis` project, by Rasmus S. Sorensen.


Refs:
* http://stackoverflow.com/questions/4092927/generating-movie-from-python-without-saving-individual-frames-to-files
* https://en.wikibooks.org/wiki/FFMPEG_An_Intermediate_Guide/image_sequence
* http://scikit-image.org/docs/dev/user_guide/video.html - pretty good overview.
* https://ffmpeg.org/ffmpeg.html
* https://ffmpeg.org/ffmpeg-filters.html
* https://github.com/scikit-video/scikit-video/blob/master/skvideo/io/ffmpeg.py


More FFmpeg and codecs refs:
* https://trac.ffmpeg.org/wiki/Encode/H.264
* https://trac.ffmpeg.org/wiki/Encode/H.265
* https://trac.ffmpeg.org/wiki/Encode/VP9
* http://stackoverflow.com/questions/9706037/recommendation-on-the-best-quality-performance-h264-encoder
* https://superuser.com/questions/300897/what-is-a-codec-e-g-divx-and-how-does-it-differ-from-a-file
* http://slhck.info/video/2017/02/24/crf-guide.html


Terminology:
* Codec - a standard for encoding/decoding data. E.g. H.264.
* Encoder/decoder - the actual implementations responsible for encoding/decoding according to the standard, e.g. x264.
* format, or container - the standard for bringing together multiple data streams (e.g. audio and video).


Good codecs and encoders (bitstream encoding):
* VP8/VP9 (WEBM). -vcodec libvpx/libvpx-vp9
* x264 for H.264 (MPEG4 part10). -vcodec libx264
* x265 for H.265/HEVC (MPEG-H part2),
    the successor of H.264, enables extremely high quality video, about 2x better compression. -c:v libx265
* Old: DivX or Xvid for MPEG4-part2 (typically in AVI or MP4 container, or alone as .m4v bitstream).


Common container formats:
* AVI - the most basic container, from 1992. You really should NOT use this.
* MP4 (aka MPEG-4 Part 14), based on the Quicktime format.
* MKV - Free open source alternative to MP4.
* OGG - Free open source alternative to MP4.
* FLV - Flash video format.


Conclusion:
* Codec/encoder: Use x265 or libvpx-vp9 for high quality/compression.
* Container/format: Use MP4, MKV, or OGG for H.264/H.265 video, use WebM for VP9 video.
* Always use constant rate factor encoding (-crf 10 -b:v 0), or lossless (-lossless 1).


Note: H.264 is also known as "MPEG-4 Part 10".

To see supported formats/codecs/encoders/decoders for FFmpeg:
* Use `$ ffmpeg -codecs` and `$ ffmpeg -formats` to see all codecs and formats (muxers/demuxers).
* Use `$ ffmpeg -encoders` and `$ ffmpeg -decoders` to see all encoders/decoders.
* To see help for a particular muxer/demuxer: `$ ffmpeg -h muxer=matroska`

FFmpeg pixel formats:
* $ ffmpeg -pix_fmts
* Gray input formats: gray, gray(10/12/16)(be/le) e.g. gray16be.
* be = big-endian, le = little-endian.

You can use `-preset slow` or other defined presets to easily adjust speed vs quality.


Comparisons of different codecs and encoders:
* https://wideopenbokeh.com/AthenasFall/?p=172  - H.264 vs H.265
* https://blogs.gnome.org/rbultje/ - VP9 vs H.264
* http://goughlui.com/2016/08/27/video-compression-testing-x264-vs-x265-crf-in-handbrake-0-10-5/
* http://wp.xin.at/archives/3465/comment-page-1
* https://web.archive.org/web/20110719095845/http://www.tkn.tu-berlin.de/research/evalvid/EvalVid/vp8_versus_x264.html


Calling FFmpeg (or other external programs):
---------------------------------------------

Preferable method is probably the standard `subprocess` module:
* subprocess.run(args, **kwargs) [replaces the old subprocess.call() function]
* subprocess.Popen(), if you need to capture the output.

Alternatives:
* `pexpect` package
* `fabric` package
* `sh` package - https://github.com/amoffat/sh
* `plumbum`

Refs:
* http://stackoverflow.com/questions/89228/calling-an-external-command-in-python

Note: Make sure you compile/install FFmpeg with the codecs you wish to use enabled:
* --enable-libx264
* --enable-libx265
* --enable-libvpx
* --enable-libvpx-vp9
* --enable-gpl --enable-nonfree


A note about "YUV422":
* YAV defines a three-dimensional color space.
* Y = luminance,
* U/V = chrominance coordinate (color plane).
* YUV should technically only be used for analog streams, while YCbCr or similar is used for digital streams.
* Our eyes have many luminance-sensitive rod cells and fewer color-sensitive cone cells.
* We can compress/reduce the chrominance resolution/detail much more than we can compress luminance.
* This is called chroma subsampling.
* YAV 4:1:1 reduces the horizontal chrominance resolution by 1:4, but has full vertical resolution.
* YAV 4:2:2 reduces the horizontal chrominance resolution by 1:2, but has full vertical resolution.
* YAV 4:2:0 reduces the horizontal chrominance resolution by 1:2, and further cuts the vertical resolution in half.
* Since grayscale video has no chrominance data, it doesn't matter much which format you use (yav422 or yav420).


"""

from collections import deque
import sys
from subprocess import Popen, PIPE, STDOUT
import shlex
from timeit import default_timer
import inspect
import numpy as np
import click

from .image_utils import norm_pix_values
from .image_io import read_image_stream, FrameRange


def write_video_ffmpeg(
    frames, outputfn,
    vmin=0, vmax=None,
    framerate=25,  # n_frames=None,
    stdout=None, stderr=None,
    verbose=0,
    normalization_dtype='uint8',
    normalization_batching=1,  # 0=, 1/"each_frame", 2/"all_batch"
    o_pix_fmt="yuv420p",
):
    """ Write image frames to video file.
    Opens a ffmpeg process and feed it the frames, one by one, via stdin.
    Typically, `frames` will be an image reader, reading images from disk from a TIF-like file on disk,
    as a stream, such that there is no keep the whole input file in memory.

    Args:
        frames: Image frames, typically a stream/iterable of 2D numpy arrays.
        outputfn:
        vmin: Contrast range, lower threshold.
        vmax: Contrast range, lower threshold.
        normalization_dtype: Normalize data to this dtype before feeding it to ffmpeg.
        normalization_batching: How/when to compute the normalized values that are fed to ffmpeg.
            0 = Don't normalize.
                Note: ffmpeg usually expects uint8 values (0-255).
                If the image is float or uint16, the values *must* be normalized before feeding it to ffmpeg.
            1 = Normalize each frame separately, inside the loop.
            2 = Normalize all frames before the loop. This will load the full imagestack into memory.
        framerate: Tell ffmpeg to encode the video with this framerate (fps).
        stdout: Redirect ffmpeg stdout stream to this pipe/file-like object.
        stderr: Redirect ffmpeg stdout stream to this pipe/file-like object.
        verbose: How verbose to be when writing progress and status messages.
            0 = No prints
            1 = Start and end messages (file output, etc).
            2 = Process id and loop progress.
            3 = FFmpeg command, vmin/vmax, loop timings.
            4 = Loop progress printed one line per loop (WARNING: Not suited for >1000 frames!)
        o_pix_fmt: output pixel format, typically yuv420p.

    Returns:
        The file name to which the video was written (after string formatting/interpolation).

    Good bitstreams and container formats:
        h264/h265 + MP4 container
        VP9/AV1 + WebM container  # AV1 = AOMedia Video 1, the successor to VP9 by the Alliance for Open Media.

    Alternatives, using skvideo:
        skvideo.io.vwrite(fname, videodata, inputdict=None, outputdict=None, backend='ffmpeg', verbosity=0)
        # or, to customize:
        writer = FFmpegWriter(fname, inputdict=inputdict, outputdict=outputdict, verbosity=verbosity)
        # c.f. http://www.scikit-video.org/stable/io.html#writing

    Alternative Python libraries (most also just calling FFmpeg or Libav):
    * PyAV
    * MoviePy - moviepy.video.io.ffmpeg_writer
    * ImageIO
    * OpenCV
    * matplotlib.animation

    FFmpeg alternatives:
    * Libav (fork of FFmpeg) / `avconv` [note: all FFmpeg's libraries also start with "libav", e.g. "libavcodec"]
    * x265, open source encoder for H.265 bitstreams

    """
    # OBS: If you use direct writing, you must tell ffmpeg about the shape and size of the input data:
    # OBS: The order of command line arguments DOES matter.
    # Arguments BEFORE `-i` pertains to the input stream, arguments AFTER `-i` pertains to the output stream.
    # For instance, we can have multiple `-f` `-pix_fmt` arguments, one for input and another for the output.
    # NOTE: It is possible to use 16-bit image input, just use `-pix_fmt gray16be`.
    n_frames = len(frames)
    image_shape_str = "%sx%s" % frames[0].shape
    if vmin is None:
        vmin = 0
    if vmax is None:
        import types
        if isinstance(frames, types.GeneratorType):
            vmax = np.percentile(frames, q=99)
        if verbose >= 3:
            print("`vmax` pixel normalization parameter set to 99th percentile = {:0.02f}.")
    if normalization_batching == 2:
        if verbose > 2:
            print("Normalizing pixels for all frames, dtype=%s, vmin/vmax=%s/%s..." % (normalization_dtype, vmin, vmax))
        # Note: This inevitably requires reading all frames to find percentile values.
        frames = norm_pix_values(frames, vmin=vmin, vmax=vmax, dtype=normalization_dtype)
    outputfn = outputfn.format(
        fps=framerate,
        framerate=framerate,
        size=image_shape_str, shape=image_shape_str, image_shape=image_shape_str,
        nframes=n_frames,
        kframes=n_frames//1000,
        pix_fmt=o_pix_fmt,
        vmin=vmin, vmax=vmax,
        ext="webm"
    )
    # Make sure ffmpeg is installed and available on your PATH.
    cmd = ("ffmpeg"
           " -framerate {framerate}"
           " -s {shape[1]}x{shape[0]}"  # -s size, set frame size (WxH or abbreviation) [correct]
           " -f rawvideo"  # essential
           " -pix_fmt {i_pix_fmt}"  # must be gray for 1-channel input
           " -i pipe:"
           # " -pix_fmt {pix_fmt} -vf format=gray"
           " -pix_fmt {o_pix_fmt}"  # output pixel format, typically yuv420p
           " -vf format=gray"
           " -crf 10"  # Constant rate factor (quality)
           " -y"
           " {output}"  # output; container format and codec can be chosen automatically based on file extension.
           ).format(
        i_pix_fmt="gray",
        o_pix_fmt=o_pix_fmt,
        framerate=framerate,
        shape=frames[0].shape,  # image_shape_str
        output=outputfn,
    )
    # Using shlex to tokenize a command line string is more reliable than str.split:
    # 'ffmpeg -i "filename with space.avi" "output with space.webm'.split()  # Gives very wrong tokenization.
    cmd = shlex.split(cmd)
    if verbose:
        print("Writing %s frames (%s) to video file: %s" % (image_shape_str, len(frames), outputfn,))
        if verbose >= 2:
            print(" - FFmpeg command:\n    $", " ".join(cmd))
            if verbose >= 3:
                try:
                    fmin, fmax = frames.min(), frames.max()
                    print(" - vmin, vmax: %s - %s [frames input min/max: %s - %s]" % (vmin, vmax, fmin, fmax))
                except AttributeError:
                    pass
    t00 = default_timer()
    times = deque(maxlen=100)
    times.append(1.)
    video_frame_shape = frames[0].shape + (1, )
    # We re-use a single small 2d array to store all 'normalized output frame':
    # The frame that we feed to ffmpeg must have three dimensions (width, height, channels), with len(channels)==1.
    frame_uint8 = np.ndarray(video_frame_shape, dtype=normalization_dtype)
    loop_print_prefix =  "\n" if verbose > 3 else "\r"

    # TEST/DEBUGGING:
    img_fn = outputfn+'.png'
    with open(img_fn, 'wb') as fd:
        print("Saving first image to file:", img_fn)
        import PIL
        frame_type = type(frames[0])
        print(f"Normalizing pixel values for frame {frame_type} using vmin={vmin}, vmax={vmax}, dtype={normalization_dtype}")
        norm_pix_values(frames[0], vmin=vmin, vmax=vmax, dtype=normalization_dtype, output=frame_uint8[:, :, 0])
        print(f"Normalized frame_uint8: min=%s, max=%s, dtype=%s" % (frame_uint8.min(), frame_uint8.max(), frame_uint8.dtype))
        PIL.Image.fromarray(frame_uint8[:, :, 0]).save(fd, 'png')

    # return

    ffmpeg = Popen(cmd, stdin=PIPE, stdout=stdout, stderr=stderr)
    if verbose >= 1:
        print(" - FFmpeg process id:", ffmpeg.pid)

    for i, frame in enumerate(frames):
        t0 = default_timer()
        if normalization_batching == 1:
            # frame_type = type(frame)
            # print(f"Normalizing pixel values for frame {frame_type} using vmin={vmin}, vmax={vmax}, dtype={normalization_dtype}")
            norm_pix_values(frame, vmin=vmin, vmax=vmax, dtype=normalization_dtype, output=frame_uint8[:, :, 0])
        else:
            frame_uint8[:, :, 0] = frame  # ffmpeg expects input of 1-4 channels, shape = (H, W, 1..4)
        # frame is currently 2d array, should have shape (width, height, channels=1)
        # frame = frame[:, :, np.newaxis]
        assert frame_uint8.ndim == 3
        assert frame_uint8.shape[2] == 1
        # assert ffmpeg is not None  # Check status
        if verbose > 1:
            print("%s - Writing frame #%s of %s (%0.1f %%, %0.1f fps, min/max/mean/median = %s/%s/%0.01f/%s)... "
                  % (loop_print_prefix,
                     i+1, n_frames,
                     100*(i+1)/n_frames, len(times)/sum(times),  # fps = 1/seconds-per-frame
                     frame_uint8.min(), frame_uint8.max(), np.mean(frame_uint8), np.median(frame_uint8)),
                  end="")
        ffmpeg.stdin.write(frame_uint8.tobytes())
        # communicate will (1) provide input, (2) read output (stdout/stderr), and (3) wait for process to finish.
        # communicate() is generally preferable to avoid deadlocks, but is not suitable for very large memory:
        # "Note: The data read is buffered in memory, so do not use this method if the data size is large or unlimited."
        # ffmpeg.communicate(input=frame.tobytes())
        t1 = default_timer() - t0
        times.append(t1)
        if verbose > 2:
            print("done! (%0.04f s)" % (t1,), end="")
    ffmpeg.stdin.close()
    if verbose:
        print()
        # poll() returns None if process is still running, otherwise errorcode (integer)
        if verbose > 1:
            print(" - ffmpeg pool status (before awaiting exit):", ffmpeg.poll())
            print(" - Closing ffmpeg...")
    # Wait for the process to terminate, raises TimeoutExpired exception if it doesn't complete in time.
    ffmpeg.wait(timeout=30)
    exit_code = ffmpeg.poll()
    # if verbose:
    #     # poll() returns None if process is still running, otherwise errorcode (integer)
    #     print("ffmpeg pool status (error code):", exit_code)
    # ffmpeg.terminate()  # Will send OS-specific termination signal.
    if verbose:
        print(" - FFmpeg encoding of %s frames completed in %0.01f seconds with error code %s"
              % (n_frames, default_timer() - t00, exit_code))

    return outputfn


def convert_imagestack_file_to_video(
        inputfn, outputfn=None,
        vmin=0, vmax=None,
        frame_range=None,
        framerate=None,
        stdout=None,
        stderr=None,
        verbose=0, quiet=False,
        normalization_dtype='uint8',
        normalization_batching=1,  # 0=No normalization, 1="normalize each frame", 2="normalize all frames first"
        o_pix_fmt="yuv420p",
        outputfn_ext="webm"
):
    """ Wrapper around write_video_ffmpeg(), for making a CLI command.

    Args:
        inputfn: The input image/TIFF-stack file to read from.
        outputfn: Filename for the generated video output.
        vmin:
        vmax:
        frame_range:
        framerate:
        stdout:
        stderr:
        verbose:
        quiet:
        normalization_dtype:
        normalization_batching:
        o_pix_fmt:
        outputfn_ext:

    Returns:

    """

    from pprint import pprint
    pprint(locals())
    reader = read_image_stream(inputfn)
    # For frame/range selection, see e.g. `convert_nd2_to_raw` module.
    if frame_range:
        frames = FrameRange(reader=reader, frame_range=frame_range)
    else:
        frames = reader
    n_frames = len(reader)

    if framerate is None:
        framerate = 25
        print("No framerate specified; using default = %s fps" % (framerate,))

    if vmax is None:
        vmax = int(reader[0].max() * 0.9)

    if outputfn is None:
        import os
        basename = os.path.basename(inputfn)
        inputdir = os.path.dirname(inputfn)
        fn_stem, inputfn_ext = os.path.splitext(inputfn)
        inputfn_ext = inputfn_ext.strip('.').lower()
        outputfn = fn_stem + "_{fps}fps"
        if n_frames == 1000 or n_frames > 2000:
            outputfn += "{kframes}kf"
        else:
            outputfn += "{nframes}f"
        if vmin is not None and vmax is not None:
            outputfn += "{vmin}-{vmax}"
        outputfn += "." + outputfn_ext

    print("Writing %s frames from %s to file: %s" % (n_frames, inputfn, outputfn))
    write_video_ffmpeg(
        frames, outputfn,
        vmin=vmin, vmax=vmax,
        framerate=framerate,
        # stdout=stdout, stderr=stderr,
        verbose=verbose,
        normalization_dtype=normalization_dtype,
        normalization_batching=normalization_batching,
        o_pix_fmt=o_pix_fmt,
    )

