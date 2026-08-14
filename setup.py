from setuptools import Extension, setup
import numpy as np

ext = Extension(
    name="cksubgraph",
    sources=["cksubgraph.c"],
    include_dirs=[np.get_include()],
    extra_compile_args=["-O3"],
)

setup(
    name="cksubgraph",
    version="0.1.1",
    ext_modules=[ext],
)
