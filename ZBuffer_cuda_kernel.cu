#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>

#define max(a,b) \
   ({ __typeof__ (a) _a = (a); \
       __typeof__ (b) _b = (b); \
     _a > _b ? _a : _b; })

#define min(a,b) \
   ({ __typeof__ (a) _a = (a); \
       __typeof__ (b) _b = (b); \
     _a < _b ? _a : _b; })

#define min3(a,b,c) (min(min(a,b), c))
#define max3(a,b,c) (max(max(a,b), c))


namespace {

template <typename scalar_t>
__global__ void Initialize(
    scalar_t* __restrict__ out,
    scalar_t* __restrict__ zbuffer,
    size_t tri_num,
    size_t img_sz
    ) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < img_sz * img_sz; i += blockDim.x * gridDim.x) {
        out[i] = tri_num;
        zbuffer[i] = -INFINITY;
    }
}


template <typename scalar_t>
__global__ void ZBuffer_cuda_forward_kernel(
    const scalar_t* __restrict__ s2d,
    const scalar_t* __restrict__ tri,
    const scalar_t* __restrict__ vis,
    size_t tri_num,
    size_t vertex_num,
    scalar_t* __restrict__ out,
    scalar_t* __restrict__ zbuffer,
    size_t img_sz) {

    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < tri_num - 1; i += blockDim.x * gridDim.x) {
  	    if (vis[i]) {

	        int vt1 = tri[            i];
	        int vt2 = tri[	tri_num + i];
	        int vt3 = tri[2*tri_num + i];

	        double point1_u = s2d[             vt1];
	        double point1_v = s2d[vertex_num + vt1];

	        double point2_u = s2d[             vt2];
	        double point2_v = s2d[vertex_num + vt2];

	        double point3_u = s2d[             vt3];
	        double point3_v = s2d[vertex_num + vt3];

	        int umin =  int(ceil (double( min3(point1_u, point2_u, point3_u) )));
	        int umax =  int(floor(double( max3(point1_u, point2_u, point3_u) )));

	        int vmin =  int(ceil (double( min3(point1_v, point2_v, point3_v) )));
	        int vmax =  int(floor(double( max3(point1_v, point2_v, point3_v) )));

            double r = (s2d[2*vertex_num+vt1] + s2d[2*vertex_num+vt2] + s2d[2*vertex_num+vt3])/3;


            if (umax < img_sz && vmax < img_sz && umin >= 0 && vmin >= 0 ){
                for (int u = umin; u <= umax; u++){
                    for (int v = vmin; v <= vmax; v++){

                        bool flag;

                        double v0_u = point3_u - point1_u; //C - A
                        double v0_v = point3_v - point1_v; //C - A

                        double v1_u = point2_u - point1_u; //B - A
                        double v1_v = point2_v - point1_v; //B - A

                        double v2_u = u - point1_u;
                        double v2_v = v - point1_v;

                        double dot00 = v0_u * v0_u + v0_v * v0_v;
                        double dot01 = v0_u * v1_u + v0_v * v1_v;
                        double dot02 = v0_u * v2_u + v0_v * v2_v;
                        double dot11 = v1_u * v1_u + v1_v * v1_v;
                        double dot12 = v1_u * v2_u + v1_v * v2_v;

                        double inverDeno = 1 / (dot00 * dot11 - dot01 * dot01 + 1e-6);
                        double uu = (dot11 * dot02 - dot01 * dot12) * inverDeno;
                        double vv = 0;
                        if (uu < 0 or uu > 1){
                            flag = 0;

                        }
                        else {
                            vv = (dot00 * dot12 - dot01 * dot02) * inverDeno;
                            if (vv < 0 or vv > 1){
                                flag = 0;
                            }
                            else
                            {
                                flag = uu + vv <= 1;
                            }

                        }

                        if (flag){
                            if (zbuffer[u * img_sz + v] < r ){ // and triCpoint(np.asarray([u, v]), pt1, pt2, pt3)):
                                zbuffer[u * img_sz + v] = r;
                                out[u * img_sz + v ] = i;
                            }

                        }
                    }
                }
            }
        }
    }
}


template <typename scalar_t>
__global__ void ConvertToMask(
    scalar_t* __restrict__ zbuffer,
    size_t img_sz
    ) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < img_sz*img_sz; i += blockDim.x * gridDim.x) {
        if (zbuffer[i] == -INFINITY) {
    	zbuffer[i] = 0;
        }
        else {
            zbuffer[i] = 1;
        }
    }
}




} // namespace

std::vector<torch::Tensor> ZBuffer_cuda_forward(
    torch::Tensor s2d,
    torch::Tensor tri,
    torch::Tensor vis,
    const int tri_num,
    const int vertex_num,
    const int img_sz
    ) {

    auto options = torch::TensorOptions().device(torch::kCUDA, 0);
    auto output = torch::zeros({img_sz, img_sz}, options);
    auto zbuffer = torch::zeros({img_sz, img_sz}, options);



/*
  AT_DISPATCH_FLOATING_TYPES(s2d.type(), "ZBuffer_forward_cuda", ([&] {
    ZBuffer_cuda_forward_kernel<scalar_t><<<blocks, threads>>>(
        gates.packed_accessor<scalar_t,3,torch::RestrictPtrTraits,size_t>(),
        old_cell.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
        new_h.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
        new_cell.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
        input_gate.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
        output_gate.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
        candidate_cell.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>());
  }));
*/

    // Initialize<<32, 256>>(output, zbuffer, tri_num - 1, img_sz);
    AT_DISPATCH_FLOATING_TYPES(s2d.type(), "ZBuffer_forward_cuda", ([&] {
        Initialize<scalar_t><<<32, 224>>> (
            output.data<scalar_t>(),
            zbuffer.data<scalar_t>(),
            tri_num,
            img_sz
        );
    }));

    AT_DISPATCH_FLOATING_TYPES(s2d.type(), "ZBuffer_forward_cuda", ([&] {
        ZBuffer_cuda_forward_kernel<scalar_t><<<32, 224>>> (
            s2d.data<scalar_t>(),
            tri.data<scalar_t>(),
            vis.data<scalar_t>(),
            tri_num + 1,
            vertex_num,
            output.data<scalar_t>(),
            zbuffer.data<scalar_t>(),
            img_sz
        );
    }));

    AT_DISPATCH_FLOATING_TYPES(s2d.type(), "ZBuffer_forward_cuda", ([&] {
        ConvertToMask<scalar_t><<<32, 224>>> (
            zbuffer.data<scalar_t>(),
            img_sz
        );
    }));


    return {output, zbuffer};
}

