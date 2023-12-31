#include "utils.cuh"
/*
__global__ void fill(float *a , float x, int N)
{
   int index =  blockIdx.x * blockDim.x + threadIdx.x;
   int stride = blockDim.x * gridDim.x;

   for(int i = index; i < N; i += stride)
   {
       a[i] = x;
   }
}
*/

void fill_vector(float *target, int size, float val){
    for(int i = 0; i < size; ++i){
        target[i] = val;
    }
}

cudaDeviceProp getDetails(int deviceId)
{
        cudaDeviceProp props;
            cudaGetDeviceProperties(&props, deviceId);
                return props;
}

void generate_random_vector(float* target, int n){
    for(int i = 0; i < n; ++i){
        float tmp = (float)(rand() % 5) + rand() % 5 * 0.01;
        tmp = (rand() % 2 == 0) ? tmp : tmp * (-1.);
        target[i] = tmp;
    }
}

void generate_random_matrix(float* target, int n){
    for(int i = 0; i < n; ++i){
	for(int j = 0; j < n; ++j){
        float tmp = (float)(rand() % 5) + rand() % 5 * 0.01;
        tmp = (rand() % 2 == 0) ? tmp : tmp * (-1.);     
   	target[i * n + j] = tmp;
	}
    }
}


void copy_vector(float *src, float *dest, int n){
    int i;
    for (i = 0; src + i && dest + i && i < n; i++) *(dest + i) = *(src + i);
    if (i != n) printf("copy failed at %d while there are %d elements in total.\n", i, n);
}

void copy_matrix(float *src, float *dest, int n){
    int i;
    for (i = 0; src + i && dest + i && i < n * n; i++) *(dest + i) = *(src + i);
    if (i != n * n) printf("copy failed at %d while there are %d elements in total.\n", i, n * n);
}


bool verify_vector(float *vec1, float *vec2, int n){
    double diff = 0.0;
    int i;
    for (i = 0; vec1 + i && vec2 + i && i < n; i++){
        diff = fabs( (double)vec1[i] - (double)vec2[i] );
        if (diff / double(vec1[i]) > 5e-3) {
            printf("error. %5.2f,%5.2f,%d\n", vec1[i], vec2[i],i);
            // return false;
            // printf("asdadsad");
            return false;
        }
    }
    return true;
}

bool verify_matrix(float *mat1, float *mat2, int n){
    double diff = 0.0;
    int i, j;
    for (i = 0; mat1 + i * n && mat2 + i * n && i < n; ++i){
        for(j = 0; mat1 + i * n + j && mat2 + i * n + j && j < n; ++j){
	    diff = fabs( (double)mat1[i * n + j] - (double)mat2[i * n + j] );
        double denominator = fabs(mat1[i * n  + j]) ;
        if (denominator < 1e-3)denominator += 1;
        // if (diff / denominator > 1e-4) {
        if (diff > 1e-2){
            printf("error is %8.5f, relateive error is %8.5f,  %8.5f,%8.5f. id: %d, %d\n",diff, (diff / denominator), mat1[i * n + j], mat2[i * n + j], i, j);
            return false;
        }
        }
    }
    return true;
}

void cpu_gemm(float alpha, float beta, float *mat1, float *mat2, int n, float *mat3){
    int i = 0, j = 0, k  = 0;
    for(i = 0; i < n; ++i){
        for(j = 0; j < n; ++j){
            float temp = 0;
	    for(k = 0; k < n; ++k)
		temp += mat1[i * n + k] * mat2[k * n + j];
            mat3[i * n + j] = alpha * temp + beta * mat3[i * n + j];
	}
    }
} 

void print_matrix(float* mat, int N){
    for(int i = 0; i < N; ++i){
        for(int j = 0; j < N; ++j){
            printf("%8.5f  ", mat[j * N + i]);
        }
        printf("\n");
    }
    fflush(stdout);
}
