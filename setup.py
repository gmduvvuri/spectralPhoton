# build script for spectralPhoton adapted from
# https://github.com/cython/cython/wiki/PackageHierarchy
import os
from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
import numpy


def scandir(dir, files=[]):
    for file in os.listdir(dir):
        path = os.path.join(dir, file)
        if os.path.isfile(path) and path.endswith(".pyx"):
            files.append(path.replace(os.path.sep, ".")[:-4])
        elif os.path.isdir(path):
            scandir(path, files)
    return files


def makeExtension(extName):
    extPath = extName.replace(".", os.path.sep) + ".pyx"
    return Extension(
        extName,
        [extPath],
        include_dirs=[
            numpy.get_include(),
            ".",
        ],  # adding the '.' to include_dirs is CRUCIAL!!
    )


# get the list of extensions
extNames = scandir("spectralPhoton")

# and build up the set of Extension objects
extensions = [makeExtension(name) for name in extNames]


setup(
    name="spectralPhoton",
    version="0.0.1",
    packages=["spectralPhoton", "spectralPhoton.rebin"],
    ext_modules=extensions,
    include_dirs=[numpy.get_include(), "."],
    cmdclass={"build_ext": build_ext},
)
