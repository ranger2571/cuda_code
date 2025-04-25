#include <cuda.h>
#include <stdio.h>
#include <sys/time.h>
#include <stdlib.h>
#include <assert.h>
#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>

double get_walltime()
{
    struct timeval tp;
    gettimeofday(&tp, NULL);
    return (double)(tp.tv_sec + tp.tv_usec * 1e-6);
}
void initCpu(float *hostA, float *hostB, int n)
{
    for (int i = 0; i < n; i++)
    {
        hostA[i] = 1;
        hostB[i] = 1;
    }
}
void addCpu(float *hostA, float *hostB, float *hostC, int n)
{
    for (int i = 0; i < n; i++)
    {
        hostC[i] = hostA[i] + hostB[i];
    }
}
__global__ void addKernel(float *deviceA, float *deviceB, float *deviceC, int n)
{
    int index = threadIdx.x + blockIdx.x * blockDim.x; // 计算全局索引
    if (index < n)
    {
        deviceC[index] = deviceA[index] + deviceB[index];
    }
}
__global__ void addKernel_float4(float *deviceA, float *deviceB, float *deviceC, int n)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int n_new = n / 4;
    int remainder = n % 4; // 需要后续处理余数

    if (index < n_new) {
        float4 a = reinterpret_cast<float4*>(deviceA)[index];
        float4 b = reinterpret_cast<float4*>(deviceB)[index];
        float4 c = {a.x+b.x, a.y+b.y, a.z+b.z, a.w+b.w};
        reinterpret_cast<float4*>(deviceC)[index] = c;
    }

    // 可选：使用单独线程处理余数
    if (remainder != 0 && index == 0) {
        for(int i = n - remainder; i < n; ++i) {
            deviceC[i] = deviceA[i] + deviceB[i];
        }
    }
}
int main()
{
    float *hostA, *hostB, *hostC, *serialC;
    int n = 10240000;

    hostA = (float *)malloc(n * sizeof(float));
    hostB = (float *)malloc(n * sizeof(float));
    hostC = (float *)malloc(n * sizeof(float));
    serialC = (float *)malloc(n * sizeof(float));
    initCpu(hostA, hostB, n);
    double stC, elaC;
    stC = get_walltime();
    addCpu(hostA, hostB, serialC, n);
    elaC = get_walltime() - stC;
    double st, ela;
    st = get_walltime();

    float *dA, *dB, *dC;
    cudaMalloc((void **)&dA, n * sizeof(float));
    cudaMalloc((void **)&dB, n * sizeof(float));
    cudaMalloc((void **)&dC, n * sizeof(float));

    cudaMemcpy(dA, hostA, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dB, hostB, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaEvent_t start, stop;
    float ker_time = 0;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);
    int BLOCK_DIM = 1024;
    int num_block_x = n / BLOCK_DIM;
    int num_block_y = 1;
    dim3 grid_dim(num_block_x, num_block_y, 1);
    dim3 block_dim(BLOCK_DIM, 1, 1);
    addKernel<<<grid_dim, block_dim>>>(dA, dB, dC, n);

    int num_block_x_float4 = n/4 / BLOCK_DIM;
    dim3 grid_dim_1(num_block_x_float4, 1, 1);
    dim3 block_dim_1(BLOCK_DIM, 1, 1);
    addKernel_float4<<<grid_dim_1, block_dim_1>>>(dA, dB, dC, n);
    
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    cudaEventElapsedTime(&ker_time, start, stop); // must float ker_time
    cudaMemcpy(hostC, dC, n * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);

    int errors = 0;
    double epsilon = 1e-5;
    for (int i = 0; i < n; i++)
    {
        if (fabs(serialC[i] - hostC[i]) > epsilon)
        {
            errors++;
            // 如果需要，可以输出第一个出错的情况
            printf("Mismatch at index %d: serial = %f, GPU = %f\n", i, serialC[i], hostC[i]);
            break; // 若只关注第一个误差，则退出循环
        }
    }

    ela = get_walltime() - st;
    printf("n = %d: \n CPU use time:%.4f\n GPU use time:%.4f\n kernel time:%.4f\n", n, elaC, ela, ker_time / 1000.0);
    free(hostA);
    free(hostB);
    free(hostC);
    free(serialC);
    return 0;
}