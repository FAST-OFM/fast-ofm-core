"""Build the optional dependency-free exhaustive NMI accelerator."""

from setuptools import Extension, setup

setup(
    ext_modules=[
        Extension(
            "fast_ofm_core.focus.rg._rg_nmi",
            ["src/fast_ofm_core/focus/rg/_rg_nmi.cpp"],
            language="c++",
            extra_compile_args=["-O3", "-std=c++17"],
            optional=True,
        )
    ]
)
