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
    int* __restrict__ out,
    scalar_t* __restrict__ zbuffer,
    const int tri_num,
    const int img_sz) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < img_sz * img_sz; i += blockDim.x * gridDim.x) {
        out[i] = tri_num;
        zbuffer[i] = -INFINITY;
    }
}


template <typename scalar_t>
__global__ void ZBuffer_cuda_forward_kernel(
    const scalar_t* __restrict__ s2d,
    const int* __restrict__ tri,
    const bool* __restrict__ vis,
    const int tri_num,
    const int vertex_num,
    int* __restrict__ out,
    scalar_t* __restrict__ zbuffer,
    const int img_sz) {

    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < tri_num - 1; i += blockDim.x * gridDim.x) {
  	if (vis[i]) {

	    int vt1 = tri[            i];
	    int vt2 = tri[	tri_num + i];
	    int vt3 = tri[2*tri_num + i];

	    float point1_u = s2d[             vt1];
	    float point1_v = s2d[vertex_num + vt1];

	    float point2_u = s2d[             vt2];
	    float point2_v = s2d[vertex_num + vt2];

	    float point3_u = s2d[             vt3];
	    float point3_v = s2d[vertex_num + vt3];

	    int umin =  int(ceil (double( min3(point1_u, point2_u, point3_u) )));
	    int umax =  int(floor(double( max3(point1_u, point2_u, point3_u) )));

	    int vmin =  int(ceil (double( min3(point1_v, point2_v, point3_v) )));
	    int vmax =  int(floor(double( max3(point1_v, point2_v, point3_v) )));

            float r = (s2d[2*vertex_num+vt1] + s2d[2*vertex_num+vt2] + s2d[2*vertex_num+vt3])/3;


	    if (umax < img_sz && vmax < img_sz && umin >= 0 && vmin >= 0 ){
	    	for (int u = umin; u <= umax; u++){
	    		for (int v = vmin; v <= vmax; v++){

				    bool flag;

				    float v0_u = point3_u - point1_u; //C - A
                                    float v0_v = point3_v - point1_v; //C - A

				    float v1_u = point2_u - point1_u; //B - A
				    float v1_v = point2_v - point1_v; //B - A

				    float v2_u = u - point1_u;
				    float v2_v = v - point1_v;

				    float dot00 = v0_u * v0_u + v0_v * v0_v;
				    float dot01 = v0_u * v1_u + v0_v * v1_v;
				    float dot02 = v0_u * v2_u + v0_v * v2_v;
				    float dot11 = v1_u * v1_u + v1_v * v1_v;
				    float dot12 = v1_u * v2_u + v1_v * v2_v;

				    float inverDeno = 1 / (dot00 * dot11 - dot01 * dot01 + 1e-6);
				    float uu = (dot11 * dot02 - dot01 * dot12) * inverDeno;
				    float vv = 0;
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
    const int img_sz
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
    torch::Device device(torch::cuda::is_available() ? torch::kCUDA : torch::kCPU);
    auto options = torch::TensorOptions().dtype(torch::kFloat64).device(device);
    auto output = torch::zeros({img_sz, img_sz}, torch::TensorOptions().dtype(torch::kInt32).device(device));
    auto zbuffer = torch::zeros({img_sz, img_sz}, torch::TensorOptions().dtype(torch::kFloat32).device(device));


    AT_DISPATCH_FLOATING_TYPES(s2d.type(), "ZBuffer_forward_cuda", ([&] {
        Initialize<scalar_t><<<32, 256>>> (
            output.data<int>(),
            zbuffer.data<scalar_t>(),
            tri_num,
            img_sz
        );
    }));

    AT_DISPATCH_FLOATING_TYPES(s2d.type(), "ZBuffer_forward_cuda", ([&] {
        ZBuffer_cuda_forward_kernel<scalar_t><<<1, 1>>> (
            s2d.data<scalar_t>(),
            tri.data<int>(),
            vis.data<bool>(),
            tri_num+1,
            vertex_num,
            output.data<int>(),
            zbuffer.data<scalar_t>(),
            img_sz
        );
    }));

    AT_DISPATCH_FLOATING_TYPES(s2d.type(), "ZBuffer_forward_cuda", ([&] {
        ConvertToMask<scalar_t><<<32, 256>>> (
            zbuffer.data<scalar_t>(),
            img_sz
        );
    }));


    return {output, zbuffer};
}

