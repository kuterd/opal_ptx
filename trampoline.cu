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
        CUresult res = cuModuleLoadDataEx(&cuModule, ptx_code.c_str(), 0, nullptr, nullptr);
        if (res != CUDA_SUCCESS) {
            const char *errStr;
            cuGetErrorName(res, &errStr);

            std::string errorMessage = "Failed to load PTX code! Error: ";
            errorMessage += errStr ? errStr : "Unknown error";

            // Optionally include additional details
            errorMessage += ". PTX Code Size: " + std::to_string(ptx_code.size()) + " bytes.";

            throw std::runtime_error(errorMessage);
        }
    }

    void launch_kernel(const std::string &kernel_name,
                       pybind11::tuple grid_dim_tuple,
                       pybind11::tuple block_dim_tuple,
                       pybind11::tuple kernel_params_tuple) {

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
