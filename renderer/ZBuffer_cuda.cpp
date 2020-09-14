#include <torch/extension.h>

#include <vector>

// CUDA forward declarations

std::vector<torch::Tensor> ZBuffer_cuda_forward(
    torch::Tensor s2d,
    torch::Tensor tri,
    torch::Tensor vis,
    const int tri_num,
    const int vertex_num,
    const int img_sz);

// C++ interface

// NOTE: AT_ASSERT has become AT_CHECK on master after 0.4.
#define CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

std::vector<torch::Tensor> ZBuffer_forward(
    torch::Tensor s2d,
    torch::Tensor tri,
    torch::Tensor vis,
    const int tri_num,
    const int vertex_num,
    const int img_sz) {
    CHECK_INPUT(s2d);
    CHECK_INPUT(tri);
    CHECK_INPUT(vis);


    return ZBuffer_cuda_forward(s2d, tri, vis, tri_num, vertex_num, img_sz);
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &ZBuffer_forward, "ZBuffer forward (CUDA)");
}
