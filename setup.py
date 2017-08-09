import os
from setuptools import setup
# Imports the __version__ variable
#  exec(open(os.path.join(os.path.dirname(__file__), 'version.py')).read())


# Utility function to read the README file.
# Used for the long_description.  It's nice, because now 1) we have a top level
# README file and 2) it's easier to type in the README file than to put a raw
# string in below ...
def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


# Read metadata from version file
def get_version():
    with open("plotters.py") as f:
        for line in f:
            if line.startswith("__version__"):
                return line[15:-2]
    raise Exception("Could not find version number")


# Read metadata from version file
classifiers = [
    "Programming Language :: Python :: 3",
    "Intended Audience :: Developers",
    "License :: OSI Approved :: MIT License",
    "Topic :: Software Development :: Libraries",
    "Topic :: Utilities",
]

setup(
    name='plotters',
    version=get_version(),
    author="Fergal Cotter",
    author_email="fbc23@cam.ac.uk",
    description=("Convenience Functions for Plotting in Matplotlib"),
    license="MIT",
    keywords="matplotlib, images, plotting",
    url="https://github.com/fbcotter/plotters.git",
    long_description=read('README.rst'),
    classifiers=classifiers,
    py_modules=["plotters"],
    install_requires=["numpy", "matplotlib"],
    tests_require=["pytest"],
    extras_require={
        'docs': ['sphinx', 'docutils']
    }
)
