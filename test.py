import transformer as kernel_transformer
import pycuda.driver as cuda
import pycuda.autoinit
import pycuda.compiler as compiler
import pycuda.gpuarray as gpuarray
import numpy as np


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
kernel_code = kernel_builder.generate("sm_90")

kernel_file = open("test.ptx", "w")
kernel_file.write(kernel_code)
kernel_file.close()


cuda.init()
device = cuda.Device(0)
context = device.make_context()

try:
    module = cuda.module_from_buffer(kernel_code.encode())
    kernel_function = module.get_function("test")
    sample_count = 32 * 2
    data_gpu = gpuarray.zeros(sample_count, dtype=np.uint64)

    kernel_function(
        data_gpu, np.uint64(sample_count), block=(sample_count, 1, 1), grid=(1, 1, 1)
    )

    cuda.Context.synchronize()

    print(data_gpu)

finally:
    # Clean up and reset CUDA context
    context.pop()
    context.detach()
