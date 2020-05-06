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

//Additional helper macros used along the code
#define MAX_BLOCKS 65535
#define HANDLE_ERROR(status) if(status != cudaSuccess) printf("\n Error...")
#define min(a,b) a<b?a:b
#define max(a,b) a>b?a:b

//All algorithm parameters i.e constants for scheme accuracy and parallel processing.
#define THREADS 100
#define TRAJECTORIES 1000
#define n 10 //rare grid density we begin with 
#define N 10000 //dense grid density for when we do not know the exact solution
#define cl_n 5000 //standard density used for sampling from XT

//Parameters for SDE of the form 'dXt = a(t,Xt)*dt + b(t,Xt)dWt'.
#define alpha 0.1209
#define mu 0.0423
#define sigma 0.1642 
#define T 1.0
#define x0 1.0

//Stochastic drift 'a = a(t,x)'.
__device__ double func_a(double t, double x)
{
    return alpha*(mu - x);
}

//Stochastic diffusion 'b = b(t,x)'.
__device__ double func_b(double t, double x)
{
    return sigma*sqrt(x);
}

//Metric type enumeration
enum MetricType { MEAN, MAX };

//Error metric function which returns difference value between given exact and non_exact vectors.
//Type parameter defines whether metric is mean of all error values within the vector coordinates or just the greatest one.
__device__ double scheme_error(double* exact, double* non_exact, int size, MetricType type=MAX, int power=2)
{
    double result = 0.0, ab;
    int i;
    for (i = 0; i < size; i++)
    {
        if (type == MAX) {
            ab = abs(exact[i] - non_exact[i]);
            if (ab > result) result = ab;
        }
        if (type == MEAN) {
            result += pow(abs(exact[i] - non_exact[i]), power);
        }
    }
    if(type==MEAN) return result/size;
    if (type == MAX) return result;
}

//Function samples from XT and assigns results to the solutions argument.
//Number of samples is based on the count parameter. Accuracy is set by changing cl_n macro value.
__global__ void classic_euler(double* solutions, int count, long seed)
{
    //Initializing random state for the generator with given seed
    curandState_t state;
    curand_init(seed + threadIdx.x + blockDim.x * blockIdx.x, 0, 0, &state);
    
    //Sampling the trajectories in single blocks
    int trajectory_index = blockIdx.x;
    while (trajectory_index < count) {
        double X = x0;
        int j;
        double H = T / cl_n;
        for (j = 0; j < cl_n; j++)
        {
            X += func_a(j * H, X) * H + func_b(j * H, X) * curand_normal(&state) * sqrt(H);
        }
        solutions[trajectory_index] = X;
        trajectory_index += gridDim.x;
    }
}

//This function is just a CPU member which performs the above kernel. Based on the arguments it might print
//the results or save it to the file. Reason for this is that it might be useful for examining the probability distribution of XT. 
//Samples are computed using at most half of the available blocks.
void get_solution_samples(int samples, bool save_to_file=true, bool print=false)
{
    if (samples >= MAX_BLOCKS)
    {
        printf("TOO MANY SAMPLES! - ERROR");
        return;
    }
     
    double* solutions = (double*) malloc(sizeof(double)*samples);
    double* solutions_dev;

    //Seed
    srand(time(NULL));
    int seed = rand();

    //Measuring performance
    cudaEvent_t start, stop;
    HANDLE_ERROR(cudaEventCreate(&start));
    HANDLE_ERROR(cudaEventCreate(&stop));
    HANDLE_ERROR(cudaEventRecord(start, 0));
    HANDLE_ERROR(cudaMalloc((void**)&solutions_dev, sizeof(double) * samples));

    //Choosing at most half of the available blocks
    classic_euler<<<min(samples, (MAX_BLOCKS/2)), 1 >>>(solutions_dev, samples, seed);

    HANDLE_ERROR(cudaMemcpy(solutions, solutions_dev, sizeof(double*) * samples, cudaMemcpyDeviceToHost));
    HANDLE_ERROR(cudaFree(solutions_dev));

    HANDLE_ERROR(cudaEventRecord(stop, 0));
    cudaEventSynchronize(stop);

    //Printing the performance of the kernel
    float time;
    HANDLE_ERROR(cudaEventElapsedTime(&time, start, stop));
    printf("classic_euler kernel: %3.1f ms\n", time);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    //Based on the input arguments it performs printing or saving to the file
    int i;
    double mean = 0.0;
    FILE* fptr = fopen("samples.txt", "w");
    if (save_to_file) fprintf(fptr, "sample,value\n");
    for (i = 0; i < samples; i++) {
        if(print) printf("%f\n", solutions[i]);
        if(save_to_file) fprintf(fptr, "%d,%f\n", i+1, solutions[i]);
        mean += solutions[i]/samples;
    }
    if (save_to_file) fprintf(fptr, "%mean:%f\n", mean);
    fclose(fptr);
    free(solutions);
}


// Function is based on the reduction process used in parallel processing. For defined stochastic differential eqation
// for which we do not know the exact solution it uses Euler-Maruyama scheme to find the approximate solutions on
// rare and dense grid. Then it calculates the average value of errors and the average value of samples obtained from XT.
// Average values are calculated per block and saved to the given pointer arguments. Later on the final step consists of 
// averaging those partial results.
__global__ void euler_for_unknown(double* partial_mean, double* partial_error, int _n, long seed, bool debug=false)
{
    //Shared memory for some thread computed trajectory values.
    //Notice that we suppose that kernel always runs with number 
    //of threads which is not greater than THREADS 
    __shared__ double error_for_thread[THREADS]; 
    __shared__ double solution_for_thread[THREADS];

    //Every calculated trajectory can identified by single thread that performs computations
    int index = threadIdx.x + blockDim.x * blockIdx.x;

    //Skipping if there are more threads than actually needed
    if (index < TRAJECTORIES)
    {
        //Initializing random state based on the given seed
        curandState_t state;
        curand_init(seed + index, 0, 0, &state);

        //Declaring and initializing other variables 
        double temp = x0, TEMP = x0;
        double* x = (double*)malloc(sizeof(double) * _n); // Here goes rare grid solution
        double* X = (double*)malloc(sizeof(double) * _n); // and dense grid solution

        double curr_W = 0.0;
        double prev_W = 0.0;
        double dW;

        //Number of samples of more accurate (almost exact) solution. Value of _N describes how 
        //many of these are located between every grid point of rare solution
        int _N = N / _n;

        //Algorithm incrementation steps
        double h = T / _n;
        double H = h / _N;

        //Two loops for iterating through both rare and dense solutions
        int i, j;
        for (i = 0; i < _n; i++) {
            for (j = 0; j < _N; j++)
            {
                dW = curand_normal(&state) * sqrt(H);
                TEMP += func_a(i*h + j*H, TEMP) * H + func_b(i*h + j*H, TEMP) * dW;
                curr_W += dW;
            }

            temp += func_a(i * h, temp) * h + func_b(i * h, temp) * (curr_W - prev_W);
            prev_W = curr_W;
            if(debug && (blockIdx.x==0 && threadIdx.x==0))
                printf("x_%d=%f, X_%d=%f, err=%f \n", i, temp, i, TEMP, abs(temp - TEMP));
            //Saving trajectories values
            x[i] = temp;
            X[i] = TEMP;
        }

        //Assuming that the dense grid solution is more accurate we save it
        solution_for_thread[threadIdx.x] = X[_n - 1];

        //Both dense and rare grid solutions are used to get the error value
        error_for_thread[threadIdx.x] = scheme_error(X, x, _n);

        //Printing single result if needed
        if (debug && (blockIdx.x == 0 && threadIdx.x == 0))
            printf("error=%f\n", error_for_thread[0]);
        free(x);
        free(X);
    }

    //Before doing the reduction it is important to synchronize all the threads within the block.
    //The reason is to make sure that shared memory is fully initialized.
    __syncthreads();

    if (threadIdx.x == 0)
    {
        //First thread is chosen to be the one that manages reduction process
        int i;
        double error_value = 0.0;
        double mean_value = 0.0;
        for (i = 0; i < blockDim.x; i++)
        {
            if (i + blockIdx.x * blockDim.x < TRAJECTORIES) {
                error_value += error_for_thread[i];
                mean_value += solution_for_thread[i];
            }
        }
        partial_error[blockIdx.x] = error_value/THREADS;
        partial_mean[blockIdx.x] = mean_value/THREADS;
    }
}

//Returns true if device satisfies properties for parallel processing
//i.e header defined macro values like THREADS are not too demanding.
bool is_enough_threads()
{
    //Structure containg GPU properties
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    //Additionally printing device name and what significant properties it has
    printf("Simulation running on a device: %s\n", prop.name);
    int max_threads_per_block = prop.maxThreadsPerBlock;
    printf("Available threads: %d\n=============\n\n", max_threads_per_block);
    //Returning whether it is capable of performing required computations
    return THREADS < max_threads_per_block;
}

//CPU member function which puprose is to finish the reduced problem of calculating mean value and error.
//Additionally it prints obtained results and returns the error value.
double print_mean_and_get_error(double* partial_mean, double* partial_error, int size)
{
    int i;
    double mean = 0.0, error = 0.0;
    for (i = 0; i < size; i++)
    {
        error += partial_error[i] * THREADS / TRAJECTORIES;
        mean += partial_mean[i] * THREADS / TRAJECTORIES;
    }
    printf("error: %f\nmean: %f\n", error, mean);
    return error;
}

//Returns approximation error value for (n+trials)-rare time grid and N-dense time grid as defined in the header.
//Computations are performed by kernel function which uses given number of BLOCKS.
double approximate(int BLOCKS, int trials)
{
    //Seed
    srand(time(NULL));
    int seed = rand();

    //Memory allocation
    double* partial_mean = (double*)malloc(sizeof(double*) * BLOCKS);
    double* partial_mean_dev;

    double* partial_error = (double*)malloc(sizeof(double*) * BLOCKS);
    double* partial_error_dev;

    //Events initialization for measuring time performance during kernels evaluations
    cudaEvent_t start, stop;
    HANDLE_ERROR(cudaEventCreate(&start));
    HANDLE_ERROR(cudaEventCreate(&stop));

    HANDLE_ERROR(cudaEventRecord(start, 0));

    HANDLE_ERROR(cudaMalloc((void**)&partial_error_dev, BLOCKS * sizeof(double)));
    HANDLE_ERROR(cudaMalloc((void**)&partial_mean_dev, BLOCKS * sizeof(double)));

    //Calculating partial errors and means for the samples of XT
    euler_for_unknown << <BLOCKS, THREADS >> > (partial_mean_dev, partial_error_dev, n+trials, seed);

    HANDLE_ERROR(cudaMemcpy(partial_error, partial_error_dev, BLOCKS * sizeof(double), cudaMemcpyDeviceToHost));
    HANDLE_ERROR(cudaMemcpy(partial_mean, partial_mean_dev, BLOCKS * sizeof(double), cudaMemcpyDeviceToHost));

    cudaFree(partial_error_dev);
    cudaFree(partial_mean_dev);

    HANDLE_ERROR(cudaEventRecord(stop, 0));
    cudaEventSynchronize(stop);

    //Measuring performance
    float time;
    HANDLE_ERROR(cudaEventElapsedTime(&time, start, stop));
    printf("euler_for_unknown kernel: %3.1f ms\n", time);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    //CPU usage for solving reduced problem of partial errors and means
    double error = print_mean_and_get_error(partial_mean, partial_error, BLOCKS);

    free(partial_mean);
    free(partial_error);

    //Returning obtained error value
    return error;
}

int main()
{
    //Validating if the accuracy parameters aren't too demanding
    if (!is_enough_threads()) {
        printf("Requesting for too many threads: %d", THREADS);
        return -1;
    }

    //Finding number of blocks that together with threads are fine to calculate all trajectories
    const int BLOCKS = (TRAJECTORIES + THREADS - 1) / THREADS;
    if (BLOCKS > MAX_BLOCKS) {
        printf("Requesting for too many blocks: %d", BLOCKS);
        return -1;
    }

    //Getting distinct error values converted to log and saved to the file for future regression
    //analysis of convergence rate
    int i, _n;
    double log_error;
    FILE* fptr = fopen("log_error.txt", "w");
    fprintf(fptr, "log_n,log(err_n))\n");
    for (i = 0; i < 10; i++) {
        printf("\nn=%d, N=%d\n", n+10*i, N);
        log_error = log(approximate(BLOCKS, 10*i));
        fprintf(fptr, "%f,%f\n", log(n + 10 * i), log_error);
    }
    fclose(fptr);

    //Printing and additionally saving given number of samples from random variable XT
    //This is useful for example for exploring the probability distribution of XT
    get_solution_samples(1000);

    return 0;
}
