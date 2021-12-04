#include <cmath>
#include <iostream>
#include "gpu-new-forward.h"

// Parameters and Declarations
#define TILE_WIDTH 2 // tiling consideration

/*
******************************************************************************************
OPTIMIZATION - Select an optimization number to execute forward convolution
The meaning of each number is as indicated:

0 - baseline (MP2)

- NOTE: All the following optimizations load the M*K*K kernels from the GPU constant memory (RUBRIC: 1)

1 - Tiled shared memory convolution (RUBRIC: 2)
2 - Shared memory matrix multiplication and input matrix unrolling (RUBRIC: 3) 
        RECOMMENDED TILE_WIDTH = 2 (change above)
3 - Kernel fusion for unrolling and matrix-multiplication (requires previous optimization) (RUBRIC: 2)
4 - Tuning with restrict and loop unrolling (RUBRIC: 3)
5 - 
******************************************************************************************
*/
#define OPTIMIZATION 2 // READ THE ABOVE DOCSTRING

// Declare constant memory for storing the kernel
// Implicit assumption that kernel dimension (K) < TILE WIDTH
__constant__ float device_constant_k[16*7*7];

__global__ void baseline_conv_forward_kernel(float *y, const float *x, const float *k, const int B, const int M, const int C, const int H, const int W, const int K)
{
    /*
    Modify this function to implement the forward pass described in Chapter 16.
    We have added an additional dimension to the tensors to support an entire mini-batch
    The goal here is to be correct AND fast.
    Function paramter definitions:
    y - output
    x - input
    k - kernel
    B - batch_size (number of images in x)
    M - number of output feature maps
    C - number of input feature maps
    H - input height dimension
    W - input width dimension
    K - kernel height and width (K x K)
    */

    const int H_out = H - K + 1;
    const int W_out = W - K + 1;

    // (void)H_out; // silence declared but never referenced warning. remove this line when you start working
    // (void)W_out; // silence declared but never referenced warning. remove this line when you start working

    // We have some nice #defs for you below to simplify indexing. Feel free to use them, or create your own.
    // An example use of these macros:
    // float a = y4d(0,0,0,0)
    // y4d(0,0,0,0) = a

#define y4d(i3, i2, i1, i0) y[(i3) * (M * H_out * W_out) + (i2) * (H_out * W_out) + (i1) * (W_out) + i0]
#define x4d(i3, i2, i1, i0) x[(i3) * (C * H * W) + (i2) * (H * W) + (i1) * (W) + i0]
#define k4d(i3, i2, i1, i0) k[(i3) * (C * K * K) + (i2) * (K * K) + (i1) * (K) + i0]

    // int H_grid = H_out / TILE_WIDTH;
    int W_grid = ceil(1.0 * W_out / TILE_WIDTH);
    int i3 = blockIdx.x;
    int i2 = blockIdx.y;
    int i1 = (blockIdx.z / W_grid) * TILE_WIDTH + threadIdx.y;
    int i0 = (blockIdx.z % W_grid) * TILE_WIDTH + threadIdx.x;

    float outputY = 0.0;
    if (i1 < H_out && i0 < W_out)
    {
        for (int c = 0; c < C; c++)
        {
            for (int p = 0; p < K; p++)
            {
                for (int q = 0; q < K; q++)
                {
                    outputY += x4d(i3, c, p + i1, q + i0) * k4d(i2, c, p, q);
                }
            }
        }
        y4d(i3, i2, i1, i0) = outputY;
    }


#undef y4d
#undef x4d
#undef k4d
}

__global__ void tiled_conv_forward_kernel(float *y, const float *x, const int B, const int M, const int C, const int H, const int W, const int K)
{
    /*
    Modify this function to implement the forward pass described in Chapter 16.
    We have added an additional dimension to the tensors to support an entire mini-batch
    The goal here is to be correct AND fast.

    Function paramter definitions:
    y - output
    x - input
    k - kernel
    B - batch_size (number of images in x)
    M - number of output feature maps
    C - number of input feature maps
    H - input height dimension
    W - input width dimension
    K - kernel height and width (K x K)
    */

    const int H_out = H - K + 1;
    const int W_out = W - K + 1;

    // (void)H_out; // silence declared but never referenced warning. remove this line when you start working
    // (void)W_out; // silence declared but never referenced warning. remove this line when you start working

    // We have some nice #defs for you below to simplify indexing. Feel free to use them, or create your own.
    // An example use of these macros:
    // float a = y4d(0,0,0,0)
    // y4d(0,0,0,0) = a

#define y4d(i3, i2, i1, i0) y[(i3) * (M * H_out * W_out) + (i2) * (H_out * W_out) + (i1) * (W_out) + i0]
#define x4d(i3, i2, i1, i0) x[(i3) * (C * H * W) + (i2) * (H * W) + (i1) * (W) + i0]
#define k4d(i3, i2, i1, i0) device_constant_k[(i3) * (C * K * K) + (i2) * (K * K) + (i1) * (K) + i0]

    // int H_grid = H_out / TILE_WIDTH;
    int W_grid = ceil(1.0 * W_out / TILE_WIDTH);
    int n = blockIdx.x;
    int m = blockIdx.y;
    int h_base = (blockIdx.z / W_grid) * TILE_WIDTH;
    int h0 = threadIdx.y;
    int w_base = (blockIdx.z % W_grid) * TILE_WIDTH;
    int w0 = threadIdx.x;

    int h = h_base + h0;
    int w = w_base + w0;
    int X_tile_width = TILE_WIDTH + K - 1;

    // Allocate shared memory for input kernel and image tiles
    extern __shared__ float shmem[];
    float* x_shared = &shmem[0];
    float* k_shared = &shmem[X_tile_width * X_tile_width];
    // pointing to shared memory pointer - this is already allocated before kernel invocation >>

    float outputY = 0.0;
    for (int c = 0; c < C; c++) {
        // Load kernel for this input feature map in the GPU shared memory >>
        if (h0 < K && w0 < K)
        {
            k_shared[h0*K+w0] = k4d(m, c, h0, w0);
        }
        __syncthreads();

        // Load the image pixels here into the shared memory >>
        for (int _p = h; _p < h_base + X_tile_width; _p += TILE_WIDTH) {
            for (int _q = w; _q < w_base + X_tile_width; _q += TILE_WIDTH) {
                if (_p < H && _q < W) {
                    x_shared[(_p - h_base)*X_tile_width + (_q - w_base)] = x4d(n, c, _p, _q);
                }
            }
        }
        __syncthreads();
        for (int p = 0; p < K; p++) {
            for (int q = 0; q < K; q++){
                outputY += x_shared[(h0+p)*X_tile_width+(w0+q)] * k_shared[p*K+q];
            }
        }
        __syncthreads();
    }

    if (h < H_out && w < W_out) {
        y4d(n, m, h, w) = outputY;
    }

#undef y4d
#undef x4d
#undef k4d
}

__global__ void naive_matrix_multiply(const float * B, float * C, int width, int P, int Q)
{
  int r = blockIdx.y * blockDim.y + threadIdx.y;
  int c = blockIdx.x * blockDim.x + threadIdx.x;
  // check boundry conditions
  if(r < P && c < Q){
    // do the multiplication for one row and col
    float value = 0.0;
    for(int k = 0; k < width; k++){
      value += device_constant_k[r * width + k] * B[k * Q + c];
    }
    // store the result
    C[r * Q + c] = value;
  }
}


__global__ void matrixMultiplyShared(const float * B, float * C,
                                     int numARows, int numAColumns,
                                     int numBRows, int numBColumns,
                                     int numCRows, int numCColumns) {

  //@@ Insert code to implement matrix multiplication here
  //@@ You have to use shared memory for this MP
  __shared__ float A_ds[TILE_WIDTH][TILE_WIDTH];
  __shared__ float B_ds[TILE_WIDTH][TILE_WIDTH];

  int Row = blockIdx.y*TILE_WIDTH + threadIdx.y;
  int Col = blockIdx.x*TILE_WIDTH + threadIdx.x;

  if (Row < numCRows && Col < numCColumns) {

    // Looping over tiles in A and B
    float pValue = 0.0;
    for (int i=0; i < ceil(1.0*numAColumns/TILE_WIDTH); i++) {

      if (TILE_WIDTH*i+threadIdx.x < numAColumns){
        A_ds[threadIdx.y][threadIdx.x] = device_constant_k[Row*numAColumns+ TILE_WIDTH*i + threadIdx.x];
      }
      if (TILE_WIDTH*i + threadIdx.y < numBRows) {
        B_ds[threadIdx.y][threadIdx.x] = B[(TILE_WIDTH*i + threadIdx.y)*numBColumns + Col];
      }
      __syncthreads();

      for (int k = 0; k < TILE_WIDTH; k++) {
        pValue += A_ds[threadIdx.y][k] * B_ds[k][threadIdx.x];
      }
      __syncthreads();
    }
    C[Row*numCColumns + Col] = pValue;
  }
}

__global__ void unroll_kernel(const float * device_x, float * device_unrolled_x, const int C, const int H, const int W, const int K) {
    // each thread retrieve and generate k*k elements in the unrolled_x
    int t = blockIdx.x*blockDim.x + threadIdx.x;
    int H_out = H - K + 1;
    int W_out = W - K + 1;
    int unrolledWidth = (H_out*W_out);

#define x3d(i2, i1, i0) device_x[(i2) * (H * W) + (i1) * (W) + i0]

    if (t < C*H_out*W_out) {

        int threadRow = t/unrolledWidth; // this row address of thread corresponds - c
        int threadCol = t%unrolledWidth; // Starting point is the same index in the X matrix
        int row = threadCol/W_out;  // Starting Row Number in X
        int col = threadCol%W_out;  // Starting Col Number in X

        // Thread will write data in the same col but rows shall offset by K*K (starting point = c*K*K) and increment by H_out x W_out
        int rowOffset = threadRow * K * K;
        int current_unroll_index = rowOffset*unrolledWidth + threadCol;
        for(int p = 0; p < K; p++) {
            for(int q = 0; q < K; q++) {
                device_unrolled_x[current_unroll_index] = x3d(threadRow, row + p, col + q);
                current_unroll_index += unrolledWidth;
            }
        }
    }
#undef x3d
}

__global__ void loop_unroll_restrict_conv_forward_kernel(float * __restrict y, const float * __restrict x, const int B, const int M, const int C, const int H, const int W, const int K)
{
    /*
    Modify this function to implement the forward pass described in Chapter 16.
    We have added an additional dimension to the tensors to support an entire mini-batch
    The goal here is to be correct AND fast.
    Function paramter definitions:
    y - output
    x - input
    k - kernel
    B - batch_size (number of images in x)
    M - number of output feature maps
    C - number of input feature maps
    H - input height dimension
    W - input width dimension
    K - kernel height and width (K x K)
    */

    const int H_out = H - K + 1;
    const int W_out = W - K + 1;

    // (void)H_out; // silence declared but never referenced warning. remove this line when you start working
    // (void)W_out; // silence declared but never referenced warning. remove this line when you start working

    // We have some nice #defs for you below to simplify indexing. Feel free to use them, or create your own.
    // An example use of these macros:
    // float a = y4d(0,0,0,0)
    // y4d(0,0,0,0) = a

#define y4d(i3, i2, i1, i0) y[(i3) * (M * H_out * W_out) + (i2) * (H_out * W_out) + (i1) * (W_out) + i0]
#define x4d(i3, i2, i1, i0) x[(i3) * (C * H * W) + (i2) * (H * W) + (i1) * (W) + i0]
#define k4d(i3, i2, i1, i0) device_constant_k[(i3) * (C * K * K) + (i2) * (K * K) + (i1) * (K) + i0]

    // int H_grid = H_out / TILE_WIDTH;
    int W_grid = ceil(1.0 * W_out / TILE_WIDTH);
    int i3 = blockIdx.x;
    int i2 = blockIdx.y;
    int i1 = (blockIdx.z / W_grid) * TILE_WIDTH + threadIdx.y;
    int i0 = (blockIdx.z % W_grid) * TILE_WIDTH + threadIdx.x;

    float outputY = 0.0;
    if (i1 < H_out && i0 < W_out)
    {
        #pragma unroll (16)
        for (int c = 0; c < C; c++)
        {
            #pragma unroll (7)
            for (int p = 0; p < K; p++)
            {
                #pragma unroll (7)
                for (int q = 0; q < K; q++)
                {
                    outputY += x4d(i3, c, p + i1, q + i0) * k4d(i2, c, p, q);
                }
            }
        }
        y4d(i3, i2, i1, i0) = outputY;
    }


#undef y4d
#undef x4d
#undef k4d
}


__host__ void GPUInterface::conv_forward_gpu_prolog(const float *host_y, const float *host_x, const float *host_k, float **device_y_ptr, float **device_x_ptr, float **device_k_ptr, const int B, const int M, const int C, const int H, const int W, const int K)
{

    const int H_out = H - K + 1;
    const int W_out = W - K + 1;

    int size_of_y = B * M * H_out * W_out;
    int size_of_x = B * C * H * W;
    int size_of_k = B * C * K * K;

    // Allocate memory and copy over the relevant data structures to the GPU
    cudaMalloc((void **)device_y_ptr, size_of_y * sizeof(float));
    cudaMalloc((void **)device_x_ptr, size_of_x * sizeof(float));
    cudaMalloc((void **)device_k_ptr, size_of_k * sizeof(float));


    // We pass double pointers for you to initialize the relevant device pointers,
    //  which are passed to the other two functions.
    cudaMemcpy(*device_x_ptr, host_x, size_of_x * sizeof(float), cudaMemcpyHostToDevice);
    
    // Avoiding unnecessary cuda-memcpy
    if (OPTIMIZATION == 0){
        cudaMemcpy(*device_k_ptr, host_k, size_of_k * sizeof(float), cudaMemcpyHostToDevice);
    }
    else {
        cudaMemcpyToSymbol(device_constant_k, host_k, 16*7*7*sizeof(float), 0, cudaMemcpyHostToDevice);
    }
        cudaDeviceSynchronize();
    // Useful snippet for error checking
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess)
    {
        std::cout << "CUDA error: " << cudaGetErrorString(error) << std::endl;
        exit(-1);
    }
}

__host__ void GPUInterface::conv_forward_gpu(float *device_y, const float *device_x, const float *device_k, const int B, const int M, const int C, const int H, const int W, const int K)
{

    // Set the kernel dimensions and call the kernel
    const int H_out = H - K + 1;
    const int W_out = W - K + 1;

    switch(OPTIMIZATION) {
        case 0: {

            // parallel for outer-nested loop
            int H_grid = ceil(1.0 * H_out / TILE_WIDTH);
            int W_grid = ceil(1.0 * W_out / TILE_WIDTH);
            int Z = H_grid * W_grid;
            dim3 blockDim(TILE_WIDTH, TILE_WIDTH, 1);
            dim3 gridDim(B, M, Z);

            // call the kernel
            baseline_conv_forward_kernel<<<gridDim, blockDim>>>(device_y, device_x, device_k, B, M, C, H, W, K);
        }

        case 1: {

            int H_grid = ceil(1.0 * H_out / TILE_WIDTH);
            int W_grid = ceil(1.0 * W_out / TILE_WIDTH);
            int Z = H_grid * W_grid;
            dim3 blockDim(TILE_WIDTH, TILE_WIDTH, 1);
            dim3 gridDim(B, M, Z);
            size_t sharedSize = sizeof(float)*((TILE_WIDTH+K-1)*(TILE_WIDTH+K-1) + K*K);
            
            // Kernel has been transferred to constant memory in prologue 

            // call the kernel
            tiled_conv_forward_kernel<<<gridDim, blockDim, sharedSize>>>(device_y, device_x, B, M, C, H, W, K);
            cudaDeviceSynchronize();
            break;

        }

        case 2: {
            // 1. Setup unroll kernel and perform unrolling
            //  1.1 W - already unrolled from input side
            //  1.2 Y - already unrolled from input side
            //  1.3 Kernel has been transferred to constant memory in prologue   

            //  1.3 Allocate Memory for unrolled X
            float * device_unrolled_x;
            int size_of_unrolled_x = C * K * K * H_out * W_out;
            cudaMalloc((void**)&device_unrolled_x, sizeof(float) * size_of_unrolled_x);

            // 2. Call the unrolling kernel for each sample image in the batch in a loop
            int CUDA_MAX_NUM_THREADS = 1024;
            // cudaDeviceGetAttribute(&CUDA_MAX_NUM_THREADS, cudaDevAttrMaxThreadsPerBlock);
            int num_threads_unroll = C*H_out*W_out;
            int num_blocks_unroll = ceil(1.0*(num_threads_unroll)/CUDA_MAX_NUM_THREADS);

            dim3 gridDim(ceil(1.0*H_out*W_out/TILE_WIDTH), ceil(1.0*M/TILE_WIDTH), 1);
            dim3 blockDim(TILE_WIDTH, TILE_WIDTH, 1);

            size_t sharedSize = sizeof(float)*(TILE_WIDTH*TILE_WIDTH*2);

            for (int n=0; n < B; n++) {
                unroll_kernel<<<num_blocks_unroll, CUDA_MAX_NUM_THREADS>>>(&device_x[n*(C * H * W)], device_unrolled_x, C, H, W, K);
                cudaDeviceSynchronize();
                matrixMultiplyShared<<<gridDim, blockDim, sharedSize>>>(device_unrolled_x, &device_y[n*(M*H_out*W_out)], M, K*K*C, K*K*C, H_out*W_out, M, H_out*W_out);
                //naive_matrix_multiply<<<gridDim, blockDim>>>(device_k, device_unrolled_x, &device_y[n*(M*H_out*W_out)], K*K*C, M, H_out*W_out);
                cudaDeviceSynchronize();
            }
            break;
        }

        case 3: {
            std::cout<<"Not Implemented!"<<std::endl;
            break;
        }
        case 4: {

            // parallel for outer-nested loop
            int H_grid = ceil(1.0 * H_out / TILE_WIDTH);
            int W_grid = ceil(1.0 * W_out / TILE_WIDTH);
            int Z = H_grid * W_grid;
            dim3 blockDim(TILE_WIDTH, TILE_WIDTH, 1);
            dim3 gridDim(B, M, Z);
            // Kernel has been transferred to constant memory in prologue
            // call the kernel
            loop_unroll_restrict_conv_forward_kernel<<<gridDim, blockDim>>>(device_y, device_x, device_k, B, M, C, H, W, K);
            break;
        }
        default: {
            std::cout<<"Invalid Optimization Number!"<<std::endl;
            exit(-1);
        }
    }
}

__host__ void GPUInterface::conv_forward_gpu_epilog(float *host_y, float *device_y, float *device_x, float *device_k, const int B, const int M, const int C, const int H, const int W, const int K)
{
    // Copy the output back to host
    const int H_out = H - K + 1;
    const int W_out = W - K + 1;

    int size_of_y = B * M * H_out * W_out;
    cudaMemcpy(host_y, device_y, size_of_y * sizeof(float), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(device_y);
    cudaFree(device_x);
    cudaFree(device_k);
}

__host__ void GPUInterface::get_device_properties()
{
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);

    for (int dev = 0; dev < deviceCount; dev++)
    {
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, dev);

        std::cout << "Device " << dev << " name: " << deviceProp.name << std::endl;
        std::cout << "Computational capabilities: " << deviceProp.major << "." << deviceProp.minor << std::endl;
        std::cout << "Max Global memory size: " << deviceProp.totalGlobalMem << std::endl;
        std::cout << "Max Constant memory size: " << deviceProp.totalConstMem << std::endl;
        std::cout << "Max Shared memory size per block: " << deviceProp.sharedMemPerBlock << std::endl;
        std::cout << "Max threads per block: " << deviceProp.maxThreadsPerBlock << std::endl;
        std::cout << "Max block dimensions: " << deviceProp.maxThreadsDim[0] << " x, " << deviceProp.maxThreadsDim[1] << " y, " << deviceProp.maxThreadsDim[2] << " z" << std::endl;
        std::cout << "Max grid dimensions: " << deviceProp.maxGridSize[0] << " x, " << deviceProp.maxGridSize[1] << " y, " << deviceProp.maxGridSize[2] << " z" << std::endl;
        std::cout << "Warp Size: " << deviceProp.warpSize << std::endl;
    }
}
