# Copyright 2018 Rasmus Scholer Sorensen, <rasmusscholer@gmail.com>

"""

Module with all click CLIs.

"""

import inspect
import click
from .video_io import convert_imagestack_file_to_video


# CLI for convert_stack_to_video_ffmpeg:
convert_imagestack_file_to_video_cli = click.Command(
    callback=convert_imagestack_file_to_video,
    name=convert_imagestack_file_to_video.__name__,
    help=inspect.getdoc(convert_imagestack_file_to_video),
    params=[
        click.Option(['--frame-range'], type=int, nargs=2),
        click.Option(['--framerate'], type=click.types.FLOAT),
        click.Option(['--vmin'], type=click.types.FLOAT, default=0),
        click.Option(['--vmax'], type=click.types.FLOAT),
        click.Option(['--o-pix-fmt'], default='yuv420p'),
        click.Option(['--normalization_dtype'], default='uint8'),
        click.Option(['--normalization_batching'], default=1, type=int),
        click.Option(['--verbose', '-v'], count=True),
        click.Option(['--quiet/--no-quiet']),
        click.Argument(['inputfn'], required=True, type=click.Path(dir_okay=False, file_okay=True, exists=True)),
        click.Argument(['outputfn'], required=False)
    ]
)
