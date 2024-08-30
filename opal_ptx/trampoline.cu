#include <torch/extension.h>
#include <cuda.h>
#include <iostream>
#include <string>
#include <stdexcept>


#define CHECK_CUDA(call) do { \
    CUresult result = (call); \
    if (result != CUDA_SUCCESS) { \
        const char *errStr; \
        cuGetErrorName(result, &errStr); \
        if (!errStr) { \
            errStr = "Unknown error"; \
        } \
        std::string errorMsg = "CUDA error: "; \
        errorMsg += errStr; \
        throw std::runtime_error(errorMsg); \
    } \
} while (0)

class CuModuleWrapper {
public:
    CuModuleWrapper() : cuModule(nullptr) {}

    ~CuModuleWrapper() {
        if (cuModule) {
            cuModuleUnload(cuModule);
        }
    }

    void load_ptx_code(const std::string &ptx_code) {
        // Define log buffers and their sizes

        const size_t buffer_size = 32768;
        char info_log_buffer[buffer_size];

        char error_log_buffer[buffer_size];

        CUmodule module;
        CUjit_option options[4];
        void *option_values[4];

        // Set up options for info log buffer size and buffer
        options[0] = CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES;
        option_values[0] = (void*)buffer_size;

        options[1] = CU_JIT_INFO_LOG_BUFFER;
        option_values[1] = (void*)info_log_buffer;

        // Set up options for error log buffer size and buffer
        options[2] = CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES;
        option_values[2] = (void*)buffer_size;

        options[3] = CU_JIT_ERROR_LOG_BUFFER;
        option_values[3] = (void*)error_log_buffer;

        CUresult res = cuModuleLoadDataEx(&cuModule, ptx_code.c_str(), 4, options, option_values);
        if (res != CUDA_SUCCESS) {
            const char *errStr;
            cuGetErrorName(res, &errStr);

            std::string errorMessage = "Failed to load PTX code! Error: ";
            errorMessage += errStr ? errStr : "Unknown error";
            errorMessage += "\nError: " + std::string(error_log_buffer) + " Info: " + std::string(info_log_buffer);

            throw std::runtime_error(errorMessage);
        }
    }

    void launch_kernel(const std::string &kernel_name,
                       pybind11::tuple grid_dim_tuple,
                       pybind11::tuple block_dim_tuple,
                       pybind11::tuple kernel_params_tuple, unsigned int shmemSize) {

        dim3 grid_dim(grid_dim_tuple[0].cast<unsigned int>(),
                      grid_dim_tuple[1].cast<unsigned int>(),
                      grid_dim_tuple[2].cast<unsigned int>());

        dim3 block_dim(block_dim_tuple[0].cast<unsigned int>(),
                       block_dim_tuple[1].cast<unsigned int>(),
                       block_dim_tuple[2].cast<unsigned int>());

        std::vector<int64_t> kernel_arg_data;
        for (const auto& item : kernel_params_tuple) {
            kernel_arg_data.push_back(item.cast<int64_t>());
        }

        void** kernel_params = new void*[kernel_params_tuple.size() + 1];
        for (size_t i = 0; i < kernel_params_tuple.size(); ++i) {
            kernel_params[i] = reinterpret_cast<void*>(&kernel_arg_data[i]);
        }
        kernel_params[kernel_params_tuple.size()] = 0;

        CUfunction kernel_func;
        CHECK_CUDA(cuModuleGetFunction(&kernel_func, cuModule, kernel_name.c_str()));

        CHECK_CUDA(cuLaunchKernel(kernel_func, grid_dim.x, grid_dim.y, grid_dim.z,
                           block_dim.x, block_dim.y, block_dim.z,
                           0, nullptr, kernel_params, nullptr));
        CHECK_CUDA(cuCtxSynchronize());
    }

private:
    CUcontext cuContext;
    CUmodule cuModule;
};

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    pybind11::class_<CuModuleWrapper>(m, "CuModuleWrapper")
        .def(pybind11::init<>())
        .def("load_ptx_code", &CuModuleWrapper::load_ptx_code, "Load PTX code into the CUDA module")
        .def("launch_kernel", &CuModuleWrapper::launch_kernel, "Launch a CUDA kernel with specified parameters");
}
