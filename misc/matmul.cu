#include <stdio.h>

__global__ void matMulKernel(float *A, float *B, float *C, size_t m, size_t n, size_t k) {
	
	// calculate row and column indices for this thread
	int row = blockIdx.x * blockDim.x + threadIdx.x;
	int col = blockIdx.y * blockDim.y + threadIdx.y;

	if (row >= m || col >= k) {
		return;
	}

	float sum = 0.0f;
	for (int i = 0; i < n; ++i) {
		sum += A[row * n + i] * B[i * k + col]; 
	}

	C[row * k + col] = sum;
}

void matMul(float *A, float *B, float *C, size_t m, size_t n, size_t k) {

	// allocate device memory for input and output matrices 
	float *A_d, *B_d, *C_d;

	size_t A_size = m * n * sizeof(float);
	size_t B_size = n * k * sizeof(float);
	size_t C_size = m * k * sizeof(float);

	cudaMalloc((void **) &A_d, A_size); 
	cudaMalloc((void **) &B_d, B_size); 
	cudaMalloc((void **) &C_d, C_size); 

	// copy input matrices to device
	cudaMemcpy(A_d, A, A_size, cudaMemcpyHostToDevice);
	cudaMemcpy(B_d, B, B_size, cudaMemcpyHostToDevice);

	// declare grid/block dimensions
	int n_threads = 16;
	int grid_rows = (m + n_threads - 1) / n_threads;
    	int grid_cols = (k + n_threads - 1) / n_threads;
    
    	dim3 grid(grid_rows, grid_cols);
    	dim3 block(n_threads, n_threads);

	// call matMul kernel
    	matMulKernel<<<grid, block>>>(A_d, B_d, C_d, m, n, k);

	// copy output matrix to host
	cudaMemcpy(C, C_d, C_size, cudaMemcpyDeviceToHost);

	// free device memory
	cudaFree(A_d);
	cudaFree(B_d);
	cudaFree(C_d);
}

int main(int argc, char* argv[]) {

	// assert number of command line arguments
	if (argc != 4) {
		printf("must provide three arguments\n");
		return 1;
	}

	// read CLI args into matrix dimensions
	size_t m = atoi(argv[1]); 
	size_t n = atoi(argv[2]); 
	size_t k = atoi(argv[3]); 

	// declare new matrices
	float *A = new float[m * n];
	float *B = new float[n * k];
	float *C = new float[m * k];

	int e;

	printf("Matrix A: \n");
	for (int i = 0; i < m; ++i) {
		for (int j = 0; j < n; ++j) {
			e = (i*n) + j;
			A[e] = (float)e;
			printf("%f ", (float)e);
		}
		printf("\n");
	}

	printf("Matrix B: \n");
	for (int i = 0; i < n; ++i) {
		for (int j = 0; j < k; ++j) {
			e = (i*k) + j;
			B[e] = (float)e;
			printf("%f ", (float)e);
		}
		printf("\n");
	}

	// call matMul function
	matMul(A, B, C, m, n, k);

	printf("Matrix C: \n");
	for (int i = 0; i < m; ++i) {
		for (int j = 0; j < k; ++j) {
			printf("%f ", C[(i*k) + j]);
		}
		printf("\n");
	}

	// free memory
	delete[] A;
	delete[] B;
	delete[] C;

	return 0;

}
