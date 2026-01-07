import sys
from setuptools import setup, Extension
import pybind11

ext_modules = [
    Extension(
        "core",
        sorted([
            "core/bindings.cpp",
            "core/dsp_engine.cpp",
            "core/buffer.cpp"
        ]),
        include_dirs=[
            pybind11.get_include(),
            "core"
        ],
        language="c++",
        extra_compile_args=["-std=c++14"],  # C++14 required for modern pybind11
    ),
]

setup(
    name="swag-dds",
    version="0.1.0",
    author="Your Name",
    description="Shockwave Acoustic Gunfire - Direction Detection System",
    ext_modules=ext_modules,
    packages=["ai", "io_mod", "ui"],  
    install_requires=[
        "numpy",
        "pybind11",
    ],
    zip_safe=False,
)