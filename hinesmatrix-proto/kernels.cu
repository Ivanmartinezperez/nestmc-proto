#include <iostream>
#include <stdio.h>
#include "kernels.h"
#include "utils.hpp"
#include <omp.h>

int N;
int BlockSize;
int BlockInter;

    double init;

    double *a_device;
    double *b_device;
    double *d_device;
    double *rhs_device;
    int *p_device;

    double *a_interleave;
    double *b_interleave;
    double *d_interleave;
    double *rhs_interleave;
    double *rhs_interleave_input;
    double *rhs_interleave_test;
    int *p_interleave;

    double *rhs_interleave_device;

    double *a_host;
    double *b_host;
    double *d_host;
    //contains a non-modified version of the d array, used to copy the original data into the GPU
    double *d_input;
    //contains the value calculated for the last version executed (except sequential)
    double *rhs_host;
    //contains a non-modified version of the rhs array
    double *rhs_input;
    //contains the result of sequential cpu execution only
    double *rhs_host_test;

    int *p_host; 





void matrix_solve_flat_cpu(
            double *a,
            double *b,
            double *d,
            double *rhs,
            int *p,
            int ncells,
            int cell_size
    ) {

        double factor;
        
        for (int n = 0; n < ncells; ++n)
        {

            // get range of this thread's cell matrix
            int first = n*cell_size;
            int last  = first + cell_size;
            
            // backward sweep
            for(int i=last-1; i>first; --i) {
                factor = a[i] / d[i];
                d[p[i]+first]   -= factor * b[i];
                rhs[p[i]+first] -= factor * rhs[i];

            }

            rhs[first] /= d[first];

            // forward sweep
            for(int i=first+1; i<last; ++i) {
                rhs[i] -= b[i] * rhs[p[i]+first];
                rhs[i] /= d[i];
            }        
        }
    }

void matrix_solve_flat_multicore(
            double *a,
            double *b,
            double *d,
            double *rhs,
            int *p,
            int ncells,
            int cell_size
    ) {

        double factor;
        #pragma omp parallel for shared(a,b,d,rhs,p) private(factor)
        for (int n = 0; n < ncells; ++n)
        {

            // get range of this thread's cell matrix
            int first = n*cell_size;
            int last  = first + cell_size;

            
            // backward sweep
            for(int i=last-1; i>first; --i) {
                factor = a[i] / d[i];
                d[p[i]+first]   -= factor * b[i];
                rhs[p[i]+first] -= factor * rhs[i];
            }

            rhs[first] /= d[first];

            // forward sweep
            for(int i=first+1; i<last; ++i) {
                rhs[i] -= b[i] * rhs[p[i]+first];
                rhs[i] /= d[i];
            }        
        }

    }

__global__ void matrix_solve_flat(
            double *a,
            double *b,
            double *d,
            double *rhs,
            int *p,
            int ncells,
            int cell_size
    ) {
        int tid = threadIdx.x + blockDim.x*blockIdx.x;

        if(tid < ncells) {
            // get range of this thread's cell matrix
            int first = tid*cell_size;
            int last  = first + cell_size;
            double factor;

            // backward sweep
            for(int i=last-1; i>first; --i) {
                factor = a[i] / d[i];
                d[p[i]+first]   -= factor * b[i];
                rhs[p[i]+first] -= factor * rhs[i];
            }

            rhs[first] /= d[first];

            // forward sweep
            for(int i=first+1; i<last; ++i) {
                rhs[i] -= b[i] * rhs[p[i]+first];
                rhs[i] /= d[i];
            } 
            
            
        }
    }

__global__ void matrix_solve_batch(
            double *a, double *b, double *d, double *rhs,
            int *p,
            int ncells,
            int cell_size
    ) {

        int tid = threadIdx.x + blockDim.x*blockIdx.x;

        if(tid < ncells) {

            // get range of this thread's cell matrix
            int first = tid;
            int last  = ncells*(cell_size-1)+tid;
         
            
            double factor;
            // backward sweep
            for(int i=last; i>first; i-=ncells) {
                factor = a[i] / d[i];
                d  [p[i]*ncells+first] -= factor * b[i];
                rhs[p[i]*ncells+first] -= factor * rhs[i];
    
            }

            rhs[first] /= d[first];

            // forward sweep
            for(int i=first+ncells; i<=last; i+=ncells) {
                rhs[i] -= b[i] * rhs[p[i]*ncells+first];
                rhs[i] /= d[i];
            }
        }
        
    }


void execSeq(HinesMatrix const& params){


   init = bblas_wtime();
    matrix_solve_flat_cpu(
        a_host,
        b_host,
        d_host,
        rhs_host_test,
        p_host,
        N,
        params.cell_size
    );
    printf("        CPU SEQ %e\n", bblas_wtime()-init);

}

void execCpuMulti(HinesMatrix const& params){
    //Reset modified pointers
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < params.cell_size; ++j)
        {
            d_host[(i * params.cell_size) + j] = params.d[j];
            rhs_host[(i * params.cell_size) + j] = params.rhs[j];
            
        }
        


    init = bblas_wtime();
    matrix_solve_flat_multicore(
        a_host,
        b_host,
        d_host,
        rhs_host,
        p_host,
        N,
        params.cell_size
    );
   printf("        CPU MUL %e          ", bblas_wtime()-init);
}

void execGpuFlat(HinesMatrix const& params){

    cudaMalloc(&a_device,N*params.cell_size*sizeof(double));
    cudaMemcpy(a_device,a_host,N*params.cell_size*sizeof(double),cudaMemcpyHostToDevice);
    check();

    cudaMalloc(&b_device,N*params.cell_size*sizeof(double));
    cudaMemcpy(b_device,b_host,N*params.cell_size*sizeof(double),cudaMemcpyHostToDevice);
    check();

    cudaMalloc(&d_device,N*params.cell_size*sizeof(double));
    cudaMemcpy(d_device,d_input,N*params.cell_size*sizeof(double),cudaMemcpyHostToDevice);
    check();

    cudaMalloc(&rhs_device,N*params.cell_size*sizeof(double));
    cudaMemcpy(rhs_device,rhs_input,N*params.cell_size*sizeof(double),cudaMemcpyHostToDevice);
    check();

    cudaMalloc(&p_device,N*params.cell_size*sizeof(int));
    cudaMemcpy(p_device,p_host,N*params.cell_size*sizeof(int),cudaMemcpyHostToDevice);
    check();


    init = bblas_wtime();

    matrix_solve_flat<<<N/BlockSize, BlockSize>>>
    (
        a_device, b_device, d_device, rhs_device,
        p_device,
        N,
        params.cell_size
    );
    check();

    cudaDeviceSynchronize();
    printf("        FLAT %e             ", bblas_wtime()-init);



    cudaMemcpy(rhs_host,rhs_device,N*params.cell_size*sizeof(double),cudaMemcpyDeviceToHost);
}

void execGpuInter(HinesMatrix const& params){

    cudaMemcpy(a_device,a_interleave,N*params.cell_size*sizeof(double),cudaMemcpyHostToDevice);
    check();

    cudaMemcpy(b_device,b_interleave,N*params.cell_size*sizeof(double),cudaMemcpyHostToDevice);
    check();

    cudaMemcpy(d_device,d_interleave,N*params.cell_size*sizeof(double),cudaMemcpyHostToDevice);
    check();

    cudaMalloc(&rhs_interleave_device,N*params.cell_size*sizeof(double));
    cudaMemcpy(rhs_interleave_device,rhs_interleave_input,N*params.cell_size*sizeof(double),cudaMemcpyHostToDevice);
    check();

    cudaMemcpy(p_device,p_interleave,N*params.cell_size*sizeof(int),cudaMemcpyHostToDevice);
    check();

    init = bblas_wtime();
    matrix_solve_batch<<<N/BlockSize, BlockSize>>>
    (
        a_device, b_device, d_device, rhs_interleave_device,
        p_device,
        N,
        params.cell_size
    );
    check();

    
    cudaDeviceSynchronize();
    printf("        BATCH %e            ", bblas_wtime()-init);
    cudaMemcpy(rhs_host,rhs_interleave_device,N*params.cell_size*sizeof(double),cudaMemcpyDeviceToHost);
}


void allocateGlobalVariables(HinesMatrix const& params){
    a_host = (double*) malloc(N*params.cell_size*sizeof(double));
    b_host = (double*) malloc(N*params.cell_size*sizeof(double));
    d_host = (double*) malloc(N*params.cell_size*sizeof(double));
    d_input = (double*) malloc(N*params.cell_size*sizeof(double));
    rhs_host = (double*) malloc(N*params.cell_size*sizeof(double));
    rhs_input = (double*) malloc(N*params.cell_size*sizeof(double));
    rhs_host_test = (double*) malloc(N*params.cell_size*sizeof(double));
    p_host = (int*) malloc(N*params.cell_size*sizeof(int));

    a_interleave = (double*) malloc(N*params.cell_size*sizeof(double));
    b_interleave = (double*) malloc(N*params.cell_size*sizeof(double));
    d_interleave = (double*) malloc(N*params.cell_size*sizeof(double));
    rhs_interleave = (double*) malloc(N*params.cell_size*sizeof(double));
    rhs_interleave_input = (double*) malloc(N*params.cell_size*sizeof(double));
    rhs_interleave_test = (double*) malloc(N*params.cell_size*sizeof(double));
    p_interleave = (int*) malloc(N*params.cell_size*sizeof(int));
}

void initSimpleGlobalVariables(HinesMatrix const& params){

    for (int i = 0; i < N; ++i)
    {
        for (int j = 0; j < params.cell_size; ++j)
        {
            a_host[(i * params.cell_size) + j] = params.a[j];
            b_host[(i * params.cell_size) + j] = params.b[j];
            d_host[(i * params.cell_size) + j] = params.d[j];
            d_input[(i * params.cell_size) + j] = params.d[j];
            rhs_host[(i * params.cell_size) + j] = params.rhs[j];
            rhs_input[(i * params.cell_size) + j] = params.rhs[j];
            rhs_host_test[(i * params.cell_size) + j] = params.rhs[j];
            p_host[(i * params.cell_size) + j] = params.p[j];
            
        }
        
    }

}

void initInterleavedGlobalVariables(HinesMatrix const& params){
    for (int i = 0; i < params.cell_size; ++i)
    {
        for (int j = 0; j < N; ++j)
        {
            a_interleave[i*N+j] = a_host[j*params.cell_size+i];
            b_interleave[i*N+j] = b_host[j*params.cell_size+i];
            d_interleave[i*N+j] = d_input[j*params.cell_size+i];
            rhs_interleave[i*N+j] = rhs_input[j*params.cell_size+i];
            p_interleave[i*N+j] = p_host[j*params.cell_size+i];
            rhs_interleave_test[i*N+j] = rhs_host_test[j*params.cell_size+i];
            rhs_interleave_input[i*N+j] = rhs_input[j*params.cell_size+i];
                
        }
        
    }
}


void matrix_solve(HinesMatrix const& params,int Systems,int CudaBlockSize) {
    
    N=Systems;
    BlockSize=CudaBlockSize;

    cudaFuncSetCacheConfig(matrix_solve_flat, cudaFuncCachePreferL1);
    cudaFuncSetCacheConfig(matrix_solve_batch, cudaFuncCachePreferL1);

    allocateGlobalVariables(params);
    initSimpleGlobalVariables(params);

    /////////////////////// CPU SEQ ///////////////////////
    execSeq(params);


    ///////////////////// CPU MULTI //////////////////////////
    execCpuMulti(params);
    calcError(rhs_host,rhs_host_test,N*params.cell_size);


    ////////////////////////// GPU FLAT //////////////////////////////////////////////////////
    execGpuFlat(params);
    calcError(rhs_host,rhs_host_test,N*params.cell_size);


    ////////////////////// INTERLEAVE /////////////////
    initInterleavedGlobalVariables(params);
    execGpuInter(params);
    calcError(rhs_host,rhs_interleave_test,N*params.cell_size);


    cudaDeviceReset();
    free(a_host);  
    free(b_host);
    free(d_host);
    free(d_input); 
    free(rhs_host); 
    free(rhs_input); 
    free(rhs_host_test); 
    free(p_host); 

    free(a_interleave); 
    free(b_interleave);
    free(d_interleave); 
    free(rhs_interleave);
    free(rhs_interleave_input); 
    free(rhs_interleave_test);
    free(p_interleave);
    //check();
}


