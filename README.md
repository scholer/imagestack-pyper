# imagestack-pyper
Python CLIs for reading and converting stacked image files (e.g. TIFF movies). Uses PIMS for reading images either to stdout or directly to ffmpeg for movie conversion.




## Installation:


Install with conda/anaconda:

    conda create -n imagestack -c conda-forge pims




## Packages for reading images:



PIMS: Python Image Sequence

* Python Image Sequence: Load video and sequential images in many formats with a simple, consistent interface.
* https://github.com/soft-matter/pims
* https://pypi.org/project/PIMS/
* http://soft-matter.github.io/pims/

pims_nd2:

* Python nd2 reader based on the ND2 SDK. For reading Nikon ND2 files.
* https://github.com/soft-matter/pims_nd2
* https://pypi.org/project/pims_nd2/


nd2reader:

* Pure Python library for reading NIS Elements ND2 images and metadata.
* https://github.com/rbnvrw/nd2reader
* http://www.lighthacking.nl/nd2reader

ImageIO:

* Python library for reading and writing image data
* https://imageio.github.io/
* https://imageio.readthedocs.io/en/stable/examples.html



Christian Gohlke's tifffile module:

* https://www.lfd.uci.edu/~gohlke/code/tifffile.py.html


pylibtiff:

* Wrapper to the libtiff library to Python using ctypes, and module for reading and writing TIFF files.
* https://github.com/pearu/pylibtiff


### Full-stack image apps:

PIMS viewer:

* A graphical user interface (GUI) for PIMS.
* https://github.com/soft-matter/pimsviewer
* https://pypi.org/project/pimsviewer/


ImagePy:

* Image process framework inspired by ImageJ.
* http://imagepy.org
* https://github.com/Image-Py/imagepy
* 



## Packages for reading and writing videos:

MoviePy:

* Video editing with Python using ffmpeg or ImageMagick (pipes to binaries).
* https://github.com/Zulko/moviepy
* http://zulko.github.io/moviepy
* Examples:
    * Concatenating videos: http://zulko.github.io/moviepy/getting_started/compositing.html
    * Reading TIFF files: http://zulko.github.io/moviepy/ref/VideoClip/VideoClip.html#moviepy.video.VideoClip.ImageClip

ImageIO:

* Aims to provide a single, standard package for reading images and video.
* Supports plugins, and e.g. uses Gohlke's tifffile for reading tiff files.
* Pipes to `ffmpeg` binary for reading and writing videos.


Scikit-video:

* Video Processing in Python. Using ffmpeg or livav.
* http://www.scikit-video.org/
* https://github.com/scikit-video/scikit-video




PyAV:

* Provides python bindings to ffmpeg. 
* (Despite the name, PyAV no longer supports LibAV, since everyone has returned to using ffmpeg.) 
