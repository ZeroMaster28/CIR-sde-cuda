#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>
#include <device_functions.h>
#pragma once
#ifdef __INTELLISENSE__
void __syncthreads();
long long int clock64();
#endif

#include <math.h>
#include <stdio.h>
#include <curand_kernel.h>

// additional helping macros
#define MAX_BLOCKS 65535
#define HANDLE_ERROR(status) if(status != cudaSuccess) printf("\n Error...")

//algorithm parameters i.e constants for scheme accuracy and parallel processing
#define THREADS 100
#define TRAJECTORIES 1000
#define n 100
#define N 1000

//parameters for SDE of the form 'dXt = a(t,Xt)*dt + b(t,Xt)dWt'
#define alpha 0.1209
#define mu 0.0423
#define sigma 0.1642 
#define T 1.0
#define x0 1.0

__device__ double func_a(double t, double x)
{
    return alpha*(mu - x);
}

__device__ double func_b(double t, double x)
{
    return sigma*sqrt(x);
}

//returns squared difference value between exact and approximated solutions
__device__ double scheme_error(double* exact, double* non_exact)
{
    double result = 0.0;
    int i;
    for (i = 0; i < n; i++)
    {
        result += pow(exact[i] - non_exact[i], 2);
    }
    return result;
}

//assigns vector of approximated realizations of XT to the solutions argument
__global__ void classic_euler(double* solutions)
{
    double X = x0;
    curandState_t state;
    curand_init(clock64() * blockDim.x, threadIdx.x, 0, &state);
    int j;
    double H = T / (n * N);
    for (j = 0; j < N; j++)
    {
        X += func_a(j*H, X) * H + func_b(j * H, X) * curand_normal(&state) * sqrt(H);
    }
    solutions[blockIdx.x] = X;
}

//prints given number of samples of the approximated solutions which are computed due to parallel processing
//additionally capable of saving it to the file if needed
void get_solution_samples(int samples, bool save_to_file=true, bool print=false)
{
    FILE* fptr = fopen("samples.txt", "w");;
    double* solutions = (double*) malloc(sizeof(double)*samples);
    double* solutions_dev;
    HANDLE_ERROR(cudaMalloc((void**)&solutions_dev, sizeof(double) * samples));
    if (samples < MAX_BLOCKS) {
        classic_euler<<<samples, 1 >>>(solutions_dev);
        HANDLE_ERROR(cudaMemcpy(solutions, solutions_dev, sizeof(double*) * samples, cudaMemcpyDeviceToHost));
    }
    HANDLE_ERROR(cudaFree(solutions_dev));
    int i;
    for (i = 0; i < samples; i++) {
        if(print) printf("%f\n", solutions[i]);
        if(save_to_file) fprintf(fptr, "%d,%f\n", i+1, solutions[i]);
    }
    free(solutions);
    fclose(fptr);
}

// partial_mean- mean value of the solutions for every block, partial_error - mean error of the solutions per every block
__global__ void euler_for_unknown(double* partial_mean, double* partial_error)
{
    __shared__ double error_for_thread[THREADS];
    __shared__ double solution_for_thread[THREADS];

    int index = threadIdx.x + blockDim.x * blockIdx.x;
    if (index < TRAJECTORIES)
    {
        curandState_t state;
        curand_init(clock64()*blockIdx.x, threadIdx.x, 0, &state);

        double temp = x0, TEMP = x0;
        double* x = (double*)malloc(sizeof(double) * n); // rozwi�zania na rzadkiej
        double* X = (double*)malloc(sizeof(double) * n); // rozwi�zania na g�stej

        double curr_W = 0.0;
        double prev_W = 0.0;
        double dW = 0.0;

        double h = T / n;
        double H = h / N;

        int i, j;
        for (i = 0; i < n; i++) {
            for (j = 0; j < N; j++)
            {
                dW = curand_normal(&state) * sqrt(H);
                TEMP += func_a(j * H, TEMP) * H + func_b(j * H, TEMP) * dW;
                curr_W += dW;
            }

            temp += func_a(i * h, temp) * h + func_b(i * h, temp) * (curr_W - prev_W);
            prev_W = curr_W;

            x[i] = temp;
            X[i] = TEMP;
        }

        solution_for_thread[threadIdx.x] = X[n - 1];
        error_for_thread[threadIdx.x] = scheme_error(X, x);
        free(x);
        free(X);
    }

    __syncthreads();

    if (threadIdx.x == 0)
    {
        //First thread is the one to manage reduction process
        double error_value = 0.0;
        double mean_value = 0.0;
        int i;
        for (i = 0; i < blockDim.x; i++)
        {
            if (i + blockDim.x * blockIdx.x < TRAJECTORIES) {
                error_value += error_for_thread[i] / blockDim.x;
                mean_value += solution_for_thread[i] / blockDim.x;
            }
        }
        partial_error[blockIdx.x] = error_value;
        partial_mean[blockIdx.x] = mean_value;
    }
}

//returns true if the device satisfies properties for parallel processing
bool is_enough_threads()
{
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("Simulation running on a device: %s\n", prop.name);
    int max_threads_per_block = prop.maxThreadsPerBlock;
    printf("Available threads: %d\n=============\n\n", max_threads_per_block);
    return THREADS < max_threads_per_block;
}

// prints overall mean value of XT as well as approximation error
void print_results(double* partial_mean, double* partial_error, int size)
{
    int i;
    double mean = 0.0, error = 0.0;
    for (i = 0; i < size; i++)
    {
        error += partial_error[i] * THREADS / TRAJECTORIES;
        mean += partial_mean[i] * THREADS / TRAJECTORIES;
    }
    printf("error: %f\nmean: %f", error, mean);
}

int main()
{
    //Validating if accuracy parameters aren't too demanding
    if (!is_enough_threads()) {
        printf("Requesting for too many threads: %d", THREADS);
        return -1;
    }

    const int BLOCKS = (TRAJECTORIES + THREADS - 1) / THREADS;
    if (BLOCKS > MAX_BLOCKS) {
        printf("Requesting for too many blocks: %d", BLOCKS);
        return -1;
    }

    //Memory allocation
    double* partial_mean = (double*) malloc(sizeof(double*)*BLOCKS);
    double* partial_mean_dev;

    double* partial_error = (double*)malloc(sizeof(double*)*BLOCKS);
    double* partial_error_dev;

    //Events initialization for measuring time performance during kernel evaluation
    cudaEvent_t start, stop;
    HANDLE_ERROR(cudaEventCreate(&start));
    HANDLE_ERROR(cudaEventCreate(&stop));

    HANDLE_ERROR(cudaEventRecord(start, 0));

    HANDLE_ERROR(cudaMalloc((void**)&partial_error_dev, BLOCKS * sizeof(double)));
    HANDLE_ERROR(cudaMalloc((void**)&partial_mean_dev, BLOCKS * sizeof(double)));

    //calculating partial errors and means for the samples of XT
    euler_for_unknown<<<BLOCKS,THREADS>>>(partial_mean_dev, partial_error_dev);

    HANDLE_ERROR(cudaMemcpy(partial_error, partial_error_dev, BLOCKS*sizeof(double), cudaMemcpyDeviceToHost));
    HANDLE_ERROR(cudaMemcpy(partial_mean, partial_mean_dev, BLOCKS*sizeof(double), cudaMemcpyDeviceToHost));

    cudaFree(partial_error_dev);
    cudaFree(partial_mean_dev);

    HANDLE_ERROR(cudaEventRecord(stop, 0));

    cudaEventSynchronize(stop);
    float time;
    HANDLE_ERROR(cudaEventElapsedTime(&time, start, stop));
    printf("GPU computations time: %3.1f ms\n", time);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    //CPU usage for solving reduced problem of partial errors and means
    print_results(partial_mean, partial_error, BLOCKS);

    free(partial_mean);
    free(partial_error);

    //printing and additionally saving given number of samples from XT
    //useful for exploring the probability distribution
    get_solution_samples(10000);

    return 0;
}