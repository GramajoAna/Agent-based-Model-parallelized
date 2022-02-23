// cronometro para procesos en GPU
// USO: gpu_timer T; T.tic();...calculo...;T.tac(); cout << T.ms_elapsed << "ms\n";

#pragma once
#ifdef __CUDACC__

///////////////////////////////// GPU TIMER ////////////////////////////////
// use CUDA's high-resolution timers when possible
/*
#include <cuda_runtime_api.h>
#include <thrust/system/cuda/error.h> //previous thrust releases
#include <thrust/system_error.h>
#include <string>
void HANDLE_ERROR(cudaError_t error, const std::string& message = "")
{
  if(error)
    throw thrust::system_error(error, thrust::cuda_category(), message);
}
*/

/*
 *
 *      From CUDA By Example An Introduction to General-Purpose GPU Programming”
 *  by Jason Sanders and Edward Kandrot, Addison-Wesley, Upper Saddle River, NJ, 2011
 *
 */

// Macro for handle errors
#include<stdio.h>
__host__ static void HandleError(cudaError_t err, const char *file, int line)
{
    if (err != cudaSuccess) {
        printf("%s in %s at line %d\n", cudaGetErrorString(err), file, line);
        exit(EXIT_FAILURE);
    }
}
#define HANDLE_NULL( a ) {if (a == NULL) { \
                            printf( "Host memory failed in %s at line %d\n", \
                                    __FILE__, __LINE__ ); \
                            exit( EXIT_FAILURE );}}
#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ))



/* Function to check for CUDA runtime errors */
static void checkCUDAError(const char* msg) {
	/* Get last error */
    cudaError_t err = cudaGetLastError();
    if( cudaSuccess != err) {
    	/* Print the error */
        printf("Cuda error: %s %s\n",msg, cudaGetErrorString( err));
        /* Abort the program */
        exit(EXIT_FAILURE);
    }
}


struct gpu_timer
{
  cudaEvent_t start;
  cudaEvent_t end;
  float ms_elapsed;
	
  gpu_timer(void)
  {
    HANDLE_ERROR(cudaEventCreate(&start));
    HANDLE_ERROR(cudaEventCreate(&end));
    tic();
  }

  ~gpu_timer(void)
  {
    HANDLE_ERROR(cudaEventDestroy(start));
    HANDLE_ERROR(cudaEventDestroy(end));
  }

  void tic(void)
  {
    HANDLE_ERROR(cudaEventRecord(start, 0));
  }

  double tac(void)
  {
    HANDLE_ERROR(cudaEventRecord(end, 0));
    HANDLE_ERROR(cudaEventSynchronize(end));

    HANDLE_ERROR(cudaEventElapsedTime(&ms_elapsed, start, end));
    return ms_elapsed;
  }

  double epsilon(void)
  {
    return 0.5e-6;
  }
};

#include<iostream>
#define CRONOMETRAR_GPU( X,VECES ) \
			    {  { \
                            gpu_timer t; \
			    float msacum=0.0;\
			    for(int n=0;n<VECES;n++){\
			    	t.tic();\
                            	X; t.tac();\
				msacum+=t.ms_elapsed;\
			    }\
			    std::cout << "GPU: " << (msacum) << \
			    " ms (" << VECES << " veces)\n"; \
                            }}

#endif

