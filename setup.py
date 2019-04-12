from setuptools import setup

setup(
    name='imagestack-pyper',
    version='0.1dev',
    packages=['imagestack_pyper'],
    url='https://github.com/scholer/imagestack-pyper',
    license='GPLv3',
    author='Rasmus Scholer Sorensen',
    author_email='rasmusscholer@gmail.com',
    description='Python CLIs for reading and converting stacked image files (e.g. TIFF movies).',
    entry_points={
        'console_scripts': [
            # Console scripts should have lower-case names, else you may get an error when uninstalling:
            'imagestack-pype-to-ffmpeg=imagestack_pyper.cli:convert_imagestack_file_to_video_cli',
        ],
    }
)
