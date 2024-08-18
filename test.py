import transformer as kernel_transformer
from torch.utils.cpp_extension import load
import torch

trampoline = load(
    "extension",
    sources=["trampoline.cu"],
    extra_ldflags=["-lnvrtc", "-lcuda"],
    extra_cuda_cflags=["-g"],
)

@kernel_transformer.kernel()
def test(buffer: "u64", size: "s64"):
    buffer_global: u64

    tidx: u64 = u32("%tid.x") + u32("%ntid.x") * u32("%ctaid.x")
    buffer = buffer + tidx * 8
    ptx.cvta.to._global.u64(buffer_global, buffer)

    if tidx < size:
        ptx.st._global.u64([buffer_global], tidx)


kernel_builder = kernel_transformer.KernelBuilder()
test(kernel_builder)
kernel_code = kernel_builder.generate("sm_75")

kernel_file = open("test.ptx", "w")
kernel_file.write(kernel_code)
kernel_file.close()

result = torch.zeros(32, dtype=torch.uint64, device="cuda")

wrapper = trampoline.CuModuleWrapper()
wrapper.load_ptx_code(kernel_code)

wrapper.launch_kernel("test", (1, 1, 1), (32, 1, 1), (result.data_ptr(), 32))
print(result)