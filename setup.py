from setuptools import setup
from torch.utils import cpp_extension

setup(
    name="opal_ptx",
    version="0.1dev",
    packages=[
        "opal_ptx",
    ],
    author="Kuter Dinel",
    author_email="kuterdinel@gmail.com",
    description="Experimental ptx metaprogramming language.",
    ext_modules=[
        cpp_extension.CUDAExtension(
            "opal_ptx._C",
            ["opal_ptx/trampoline.cu"],
            extra_compile_args={"cxx": ["-O3"], "nvcc": ["-O3"]},
            extra_link_args=["-lnvrtc", "-lcuda"],
        )
    ],
    cmdclass={"build_ext": cpp_extension.BuildExtension},
)
