#include "device_launch_parameters.h"
#include <cuda.h>
#include <curand.h>
#include <cstdlib>
#include <stdio.h>
#include <curand.h>
#include <iostream>
#include <ctime>
#include <curand_kernel.h>
#include <math.h>
#include <assert.h>
#include <device_functions.h>
#include <vector>
// количество примеров
#define N 100
// длинна последовательности
#define M 8
// размер одного элемента последовательности
#define K 11
// количество классов
#define ClassLength 3
const int THREADS_PER_BLOCK = 16;
#define TILE_DIM 1

/* Using updated (v2) interfaces to cublas and cusparse */
#include <cuda_runtime.h>
#include <cublas_v2.h>
using namespace std;
/**
* f - tanh
* g - softmax
* Прямой проход: для каждого вектора последовательности {x(1),…x(n)} :
* вычисляем состояния скрытого слоя {s(1),…s(n)} и выходы скрытого слоя {h(1),…h(n)}
* s(t)=V?x(t)+U?h(t?1)+a
* h(t)=f(s(t))
* вычисляем выход сети y
* y(n)=g(W?h(n)+b)
* 2. Обратный проход: вычисляем ошибку выходного слоя ?o
* ?o=y?d
* вычисляем ошибку скрытого слоя в конечном состоянии ?h(n)
* ?h(n)=WT??o?f?(s(n))
* вычисляем ошибки скрытого слоя в промежуточных состояниях ?h(t) (t=1,…n)
* ?h(t)=UT??h(t+1)?f?(s(n))
* 3. Вычисляем изменение весов:
* ?W=?o?(h(n))T
* ?by=??o
* ?V=?t?h(t)?(x(t))T
* ?U=?t?h(t)?(h(t?1))T
* ?bh=??t?h(t)
*/


void print_matrix(const float *A, int nr_rows_A, int nr_cols_A) {

	for (int i = 0; i < nr_rows_A; ++i) {
		for (int j = 0; j < nr_cols_A; ++j) {
			std::cout << A[j * nr_rows_A + i] << " ";
		}
		std::cout << std::endl;
	}
	std::cout << std::endl;
}

void GPU_fill_rand(float *A, int n) {
	curandGenerator_t prng;
	curandCreateGenerator(&prng, CURAND_RNG_PSEUDO_DEFAULT);
	curandSetPseudoRandomGeneratorSeed(prng, (unsigned long long) clock());
	curandGenerateUniform(prng, A, n);
}

__global__ void setup_kernel(curandState *state) {

	int idx = threadIdx.x + blockDim.x*blockIdx.x;
	curand_init(1234, idx, 0, &state[idx]);
}

__global__ void generate_kernel(curandState *my_curandstate, const unsigned int n, const unsigned *max_rand_int, const unsigned *min_rand_int, unsigned int *result) {

	int idx = threadIdx.x + blockDim.x*blockIdx.x;

	int count = 0;
	while (count < n) {
		float myrandf = curand_uniform(my_curandstate + idx);
		myrandf *= (max_rand_int[idx] - min_rand_int[idx] + 0.999999);
		myrandf += min_rand_int[idx];
		int myrand = (int)truncf(myrandf);

		assert(myrand <= max_rand_int[idx]);
		assert(myrand >= min_rand_int[idx]);
		result[myrand - min_rand_int[idx]]++;
		count++;
	}
}

// C(m,n) = A(m,k) * B(k,n)
void gpu_blas_mmul(const float *A, const float *B, float *C, const int m, const int k, const int n) {
	int lda = m, ldb = k, ldc = m;
	const float alf = 1;
	const float bet = 0;
	const float *alpha = &alf;
	const float *beta = &bet;

	cublasHandle_t handle;
	cublasCreate(&handle);

	cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);

	cublasDestroy(handle);
}

__global__ void cuda_tanh(float *a, int n)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < n)
	{
		a[idx] = tanhf(a[idx]);
	}
}
__global__ void dTanh(float *a, int n)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < n)
	{
		a[idx] = 1 - tanhf(a[idx])*tanhf(a[idx]);
	}
}

__global__ void repeat_vector_tomatrix(const float* vector, const unsigned vlen, float* matrix, const unsigned mdim, const unsigned col_major = 0) {
	if (col_major) {
		int idx = threadIdx.x + blockIdx.x*mdim;
		float myval = vector[blockIdx.x];
		while (idx < ((blockIdx.x + 1)*mdim)) {
			matrix[idx] = myval;
			idx += blockDim.x;
		}
	}
	else {
		int idx = threadIdx.x + blockDim.x * blockIdx.x;
		float myval = vector[idx%vlen];
		while (idx < mdim*vlen) {
			matrix[idx] = myval;
			idx += gridDim.x*blockDim.x;
		}
	}
}

__global__ void add_matrixes(float *a, float *b, float *c, int m, int n) {
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	if (row >= m || col >= n)return;
	c[row + col] = a[row + col] + b[row + col];
}

void generateData(float *h_X, float*h_y)
{
	float *d_X;
	cudaMalloc(&d_X, K * M * N * sizeof(float));
	GPU_fill_rand(d_X, K * M * N);
	cudaMemcpy(h_X, d_X, K * M * N * sizeof(float), cudaMemcpyDeviceToHost);
	
	
	for (int i = 0; i < N; i++)
		h_y[i] = 0 + (rand() % static_cast<int>(ClassLength + 1));

	cudaFree(d_X);
}

/**
* @param h_V (M-1)*K
* @param h_U (M-1)*(M-1)
* @param h_a (M-1)*1
* @param h_W C*(M-1)
* @param h_b C*1
* @param netSize K,M-1,ClassLength
*/
void generateWeight(float*h_V, float*h_U, float*h_a, float*h_W, float*h_b, int*netSize)
{
	float*d_V, *d_U, *d_a, *d_W, *d_b;;
	cudaMalloc(&d_V, netSize[1] * netSize[0] * sizeof(float));
	GPU_fill_rand(d_V, netSize[1] * netSize[0]);
	cudaMemcpy(h_V, d_V, netSize[1] * netSize[0] * sizeof(float), cudaMemcpyDeviceToHost);

	cudaMalloc(&d_U, netSize[1] * netSize[1] * sizeof(float));
	GPU_fill_rand(d_U, netSize[1] * netSize[1]);
	cudaMemcpy(h_U, d_U, netSize[1] * netSize[1] * sizeof(float), cudaMemcpyDeviceToHost);

	cudaMalloc(&d_a, netSize[1] * sizeof(float));
	GPU_fill_rand(d_a, netSize[1]);
	cudaMemcpy(h_a, d_a, netSize[1] * sizeof(float), cudaMemcpyDeviceToHost);

	cudaMalloc(&d_W, netSize[2] * netSize[1] * sizeof(float));
	GPU_fill_rand(d_W, netSize[2] * netSize[1]);
	cudaMemcpy(h_W, d_W, netSize[2] * netSize[1] * sizeof(float), cudaMemcpyDeviceToHost);

	cudaMalloc(&d_b, netSize[2] * sizeof(float));
	GPU_fill_rand(d_b, netSize[2]);
	cudaMemcpy(h_b, d_b, netSize[2] * sizeof(float), cudaMemcpyDeviceToHost);

	cudaFree(d_V);
	cudaFree(d_U);
	cudaFree(d_b);
	cudaFree(d_W);
	cudaFree(d_a);
}

__global__ void count_P(const float*O, const float*Y, float* P, int n)
{
	int idx = threadIdx.x + blockDim.x*blockIdx.x;
	if (idx < n) {
		P[idx] = O[idx + int(Y[idx])<ClassLength? int(Y[idx]):0];
	}
}

__global__ void cuda_log(float* d_p, int n)
{
	int idx = threadIdx.x + blockDim.x*blockIdx.x;
	if (idx < n) {
		d_p[idx] = -1 * logf(d_p[idx]);
	}
}
void trans2(float *A, const int m, const int n)
{
	float* clone;
	cudaMalloc(&clone, m*n * sizeof(float));
	cudaMemcpy(clone, A, m*n * sizeof(float), cudaMemcpyDeviceToDevice);
	float const alpha(1.0);
	float const beta(0.0);
	cublasHandle_t handle;
	cublasCreate(&handle);
	cublasSgeam(handle, CUBLAS_OP_T, CUBLAS_OP_N, m, n, &alpha, clone, n, &beta, clone, m, A, m);
	cublasDestroy(handle);
}

__global__ void multiplyMatrixPoint(float*A, float*B, float*C)
{
	__shared__ float tile[TILE_DIM][TILE_DIM];

	int x = blockIdx.x * TILE_DIM + threadIdx.x;
	int y = blockIdx.y * TILE_DIM + threadIdx.y;
	int width = gridDim.x * TILE_DIM;

	for (int j = 0; j < TILE_DIM; j += THREADS_PER_BLOCK)
		tile[threadIdx.y + j][threadIdx.x] = A[(y + j)*width + x] * B[(y + j)*width + x];

	__syncthreads();

	x = blockIdx.y * TILE_DIM + threadIdx.x; 
	y = blockIdx.x * TILE_DIM + threadIdx.y;

	for (int j = 0; j < TILE_DIM; j += THREADS_PER_BLOCK)
		C[(y + j)*width + x] = tile[threadIdx.x][threadIdx.y + j];
}

void softmax(float* v, int n, int m)
{

	for (int i = 0; i<m; i++)
	{
		float m = v[i*n];
		for (int j = 0; j<n; j++)
		{
			m = fmax(m, v[i*n + j]);
		}
		float s = 0;
		for (int j = 0; j<n; j++)
		{
			s += expf(v[i*n + j] - m);
		}
		for (int j = 0; j<n; j++)
		{
			v[i*n + j] = expf(v[i*n + j] - m) / s;
		}
	}

}

float cross_entropy(float* h_O, const float* h_y, int cl_n, int n)
{
	float *d_O, *d_Y;
	float *d_P;
	float *h_P = (float*)malloc(n * sizeof(float));

	cudaMalloc(&d_O, ClassLength*N * sizeof(float));
	cudaMalloc(&d_Y, N * sizeof(float));
	cudaMalloc(&d_P, N * sizeof(float));
	cudaMemcpy(d_O, h_O, ClassLength*N * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_Y, h_y, N * sizeof(float), cudaMemcpyHostToDevice);
	count_P << <(N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, THREADS_PER_BLOCK >> > (d_O, d_Y, d_P, N);
	cuda_log << <(N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, THREADS_PER_BLOCK >> > (d_P, N);
	cudaMemcpy(h_P, d_P, N * sizeof(float), cudaMemcpyDeviceToHost);
	float s = 0;
	for (int i = 0; i < n; i++) {
		s += h_P[i] / n;
	}
	free(h_P);
	cudaFree(d_O);
	cudaFree(d_P);
	cudaFree(d_Y);
	return s;
}

void forward(float* h_X, float* h_V, float* h_U, float* h_a, float* h_W, float* h_b, int* netSize, float* Out, float**sT, float**Ht)
{
	float* h = (float*)malloc((M - 1)*N * sizeof(float));
	memset(h, 0, (M - 1)*N * sizeof(float));

	float *d_V, *d_X, *d_VX, *d_U, *d_h, *d_Uh, *d_a, *d_VX_Uh, *temp_H;
	cudaMalloc(&d_V, (M - 1) * K * sizeof(float));
	cudaMalloc(&d_X, K*N * sizeof(float));
	cudaMemcpy(d_V, h_V, (M - 1) * K * sizeof(float), cudaMemcpyHostToDevice);
	cudaMalloc(&d_VX, (M - 1) * N * sizeof(float));
	cudaMalloc(&d_VX_Uh, (M - 1) * N * sizeof(float));
	cudaMalloc(&d_U, (M - 1) * (M - 1) * sizeof(float));
	cudaMalloc(&temp_H, (M - 1) *N * sizeof(float));
	cudaMalloc(&d_h, (M - 1) *N * sizeof(float));
	cudaMalloc(&d_Uh, (M - 1) * sizeof(float));
	cudaMemcpy(d_U, h_U, (M - 1) * (M - 1) * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_h, h, (M - 1)*N * sizeof(float), cudaMemcpyHostToDevice);
	cudaMalloc(&d_a, (M - 1) * sizeof(float));
	cudaMemcpy(d_a, h_a, (M - 1) * sizeof(float), cudaMemcpyHostToDevice);
	for (int t = 0; t<M; t++)
	{
		float* Xt = (float*)malloc(K*N * sizeof(float));
		memcpy(Xt, h_X + t * K*N, K*N * sizeof(float));
		cudaMemcpy(d_X, Xt, K*N * sizeof(float), cudaMemcpyHostToDevice);

		gpu_blas_mmul(d_V, d_X, d_VX, M - 1, K, N);
		cudaFree(d_X);
		free(Xt);

		gpu_blas_mmul(d_U, d_h, d_Uh, M - 1, M - 1, N);

		// Vx+Uh
		add_matrixes << <(N + (THREADS_PER_BLOCK - 1)) / THREADS_PER_BLOCK, THREADS_PER_BLOCK >> >(d_VX, d_Uh, d_VX_Uh, M - 1, N);

		// h = Vx + Uh+a
		repeat_vector_tomatrix << <(N + (THREADS_PER_BLOCK - 1)) / THREADS_PER_BLOCK, THREADS_PER_BLOCK >> >(d_a, M - 1, temp_H, N);
		add_matrixes << <(N + (THREADS_PER_BLOCK - 1)) / THREADS_PER_BLOCK, THREADS_PER_BLOCK >> >(d_VX_Uh, temp_H, d_h, M - 1, N);
		cudaMemcpy(sT[t], d_h, (M - 1) *N * sizeof(float), cudaMemcpyDeviceToHost);
		// h = tanh(h)
		cuda_tanh << <1, (M - 1)*N >> >(d_h, (M - 1)*N);
		cudaMemcpy(Ht[t], d_h, (M - 1) *N * sizeof(float), cudaMemcpyDeviceToHost);
	}
	cudaFree(d_V);
	cudaFree(d_X);
	cudaFree(d_VX);
	cudaFree(d_Uh);
	cudaFree(d_a);
	cudaFree(d_VX_Uh);
	cudaFree(temp_H);

	// Out
	float *d_W, *d_b, *d_Wh, *d_O, *temp_o;
	cudaMalloc(&d_W, ClassLength*(M - 1) * sizeof(float));
	cudaMemcpy(d_W, h_W, ClassLength*(M - 1) * sizeof(float), cudaMemcpyHostToDevice);
	cudaMalloc(&d_b, ClassLength * sizeof(float));
	cudaMemcpy(d_b, h_b, ClassLength * sizeof(float), cudaMemcpyHostToDevice);
	cudaMalloc(&d_Wh, ClassLength*N * sizeof(float));
	cudaMalloc(&d_O, ClassLength*N * sizeof(float));
	cudaMalloc(&temp_o, ClassLength*N * sizeof(float));
	gpu_blas_mmul(d_W, d_h, d_Wh, ClassLength, M - 1, N);
	// o = Wh+b
	repeat_vector_tomatrix << <ClassLength, N >> >(d_b, ClassLength, temp_o, N);
	add_matrixes << <ClassLength, N >> >(d_Wh, temp_o, d_O, ClassLength, N);

	cudaMemcpy(Out, d_O, ClassLength*N * sizeof(float), cudaMemcpyDeviceToHost);
	softmax(Out, ClassLength, N);
	cudaFree(d_W);
	cudaFree(d_b);
	cudaFree(d_Wh);
	cudaFree(d_O);
	cudaFree(temp_o);
}

__global__ void valid_out(float*v, float*y, int n)
{
	int idx = threadIdx.x + blockDim.x*blockIdx.x;
	if (idx < n) {
		v[idx + int(y[idx])] = 1;
	}
}

__global__ void error_out(float*v, float*y, float*err, int n)
{
	int idx = threadIdx.x + blockDim.x*blockIdx.x;
	if (idx < n) {
		err[idx] = v[idx] - y[idx];
	}
}

void bptt_backward(float* h_W, float* h_U, const float*y, float*o, float*err1, float**err2, float**sT)
{
	float* d, *d_Y, *d_O;
	cudaMalloc(&d, ClassLength * N * sizeof(float));
	cudaMalloc(&d_Y, N * sizeof(float));
	cudaMemcpy(d_Y, y, N * sizeof(float), cudaMemcpyHostToDevice);
	cudaMalloc(&d_O, ClassLength * N * sizeof(float));
	cudaMemcpy(d_O, o, ClassLength * N * sizeof(float), cudaMemcpyHostToDevice);
	valid_out << <ClassLength, N >> > (d, d_Y, ClassLength * N);

	float*d_err1, *d_err2, *d_W, *d_TE, *d_dS, *d_U;
	cudaMalloc(&d_err1, ClassLength * N * sizeof(float));
	error_out << <1, ClassLength * N >> > (d_O, d, d_err1, ClassLength * N);
	cudaMemcpy(err1, d_err1, ClassLength * N * sizeof(float), cudaMemcpyDeviceToHost);

	cudaMalloc(&d_err2, (M - 1) *N * sizeof(float));
	cudaMalloc(&d_W, ClassLength * (M - 1) * sizeof(float));
	cudaMemcpy(d_W, h_W, ClassLength * N * sizeof(float), cudaMemcpyHostToDevice);
	cudaMalloc(&d_dS, (M - 1) *N * sizeof(float));
	cudaMalloc(&d_TE, (M - 1) * N * sizeof(float));
	cudaMalloc(&d_U, (M - 1) * (M - 1) * sizeof(float));
	cudaMemcpy(d_U, h_U, (M - 1) * (M - 1) * sizeof(float), cudaMemcpyHostToDevice);

	dim3 dimGrid(ClassLength / TILE_DIM, (M - 1) / TILE_DIM, 1);
	dim3 dimGridU((M - 1) / TILE_DIM, (M - 1) / TILE_DIM, 1);
	dim3 dimBlock(TILE_DIM, THREADS_PER_BLOCK, 1);
	trans2(d_W, (M - 1), ClassLength);
	trans2(d_U, (M - 1), (M - 1));

	gpu_blas_mmul(d_W, d_err1, d_TE, M - 1, ClassLength, N);
	cudaMemcpy(d_dS, sT[M - 1], (M - 1) *N * sizeof(float), cudaMemcpyHostToDevice);

	dTanh << <1, (M - 1) *N >> >(d_dS, (M - 1) *N);

	multiplyMatrixPoint << <dimGridU, dimBlock >> >(d_dS, d_TE, d_err2);
	cudaMemcpy(err2[M - 1], d_err2, (M - 1) *N * sizeof(float), cudaMemcpyDeviceToHost);

	for (int m = M - 2; m >= 0; m--)
	{
		cudaMemcpy(d_dS, sT[m], (M - 1) *N * sizeof(float), cudaMemcpyHostToDevice);
		gpu_blas_mmul(d_U, d_err2, d_TE, M - 1, ClassLength, N);
		dTanh << <1, (M - 1) *N >> >(d_dS, (M - 1) *N);
		multiplyMatrixPoint << <dimGridU, dimBlock >> >(d_dS, d_TE, d_err2);
		cudaMemcpy(err2[m], d_err2, (M - 1) *N * sizeof(float), cudaMemcpyDeviceToHost);
	}

	cudaFree(d);
	cudaFree(d_Y);
	cudaFree(d_O);
	cudaFree(d_err1);
	cudaFree(d_err2);
	cudaFree(d_W);
	cudaFree(d_TE);
	cudaFree(d_U);
	cudaFree(d_dS);
}

__global__ void multiScalaronMatrix(float scalar,float*m,int n)
{
	int idx = threadIdx.x + blockDim.x*blockIdx.x;
	if (idx < n) {
		m[idx] *= scalar;
	}
}

float norm(vector<float>v,int n)
{
	float s = 0;
	for (int i = 0; i < n; i++)
		s += v[i]*v[i] / n/n;
	return n*sqrt(s);
}

/**
* @param h_V (M-1)*K
* @param h_U (M-1)*(M-1)
* @param h_a (M-1)*1
* @param h_W C*(M-1)
* @param h_b C*1
* @param netSize K,M-1,ClassLength
*/
void bptt_gradient(float* h_x, float* h_v, float* h_u, float* h_a, float* h_w, float* h_b,
                  float* err1, float** err2, float** st, float** ht, float a)
{
	float* d_dW, *d_h,*d_err1,*d_w;
	cudaMalloc(&d_h, (M - 1)*N * sizeof(float));
	cudaMemcpy(d_h, ht[M - 1], (M - 1)*N * sizeof(float), cudaMemcpyHostToDevice);
	cudaMalloc(&d_w, ClassLength*(M - 1) * sizeof(float));
	cudaMemcpy(d_w, h_w, ClassLength*(M - 1)* sizeof(float), cudaMemcpyHostToDevice);
	cudaMalloc(&d_err1, ClassLength*N * sizeof(float));
	cudaMemcpy(d_err1, err1, ClassLength*N * sizeof(float), cudaMemcpyHostToDevice);
	cudaMalloc(&d_dW, ClassLength*(M - 1) * sizeof(float));
	trans2(d_h, (M - 1), N);
	gpu_blas_mmul(d_err1, d_h, d_dW, ClassLength, N, (M - 1));

	float dB = 0;
	for (int i = 0; i < ClassLength*N; i++)
		dB += err1[i];
	//for (int i = 0; i < ClassLength; i++)
	//	h_b[i] -= a * s;

	float* d_x, *d_err2, *d_V, *d_dV, *d_U, *d_dU, *temp_V,*temp_h;
	cudaMalloc(&temp_h, (M - 1)*N * sizeof(float));
	cudaMalloc(&d_x, K*N * sizeof(float));
	cudaMalloc(&d_err2, (M - 1) *N * sizeof(float));
	cudaMemcpy(d_err2, err2[0], (M - 1) *N * sizeof(float), cudaMemcpyHostToDevice);
	cudaMalloc(&d_V, (M - 1) *K * sizeof(float));
	cudaMemcpy(d_V, h_v, (M - 1) *K * sizeof(float), cudaMemcpyHostToDevice);
	cudaMalloc(&d_dV, (M - 1) *K * sizeof(float));
	cudaMalloc(&temp_V, (M - 1) *K * sizeof(float));
	cudaMalloc(&d_dU, (M - 1) *(M - 1) * sizeof(float));
	cudaMalloc(&d_U, (M - 1) *(M - 1) * sizeof(float));
	cudaMemcpy(d_U, h_u, (M - 1) *(M - 1) * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemset(d_dU, 0, (M - 1) *(M - 1) * sizeof(float));

	float* Xt = (float*)malloc(K*N * sizeof(float));
	memcpy(Xt, h_x , K*N * sizeof(float));
	cudaMemcpy(d_x, Xt, K*N * sizeof(float), cudaMemcpyHostToDevice);
	trans2(d_x, K, N);
	gpu_blas_mmul(d_err2, d_x, d_dV, (M - 1), N, K);
	free(Xt);
	float sA = 0;
	for(int j=0;j<N;j++)
	{
		sA+= err2[0][j];
	}

	for (int t = 1; t<M; t++)
	{
		cudaMemcpy(d_err2, err2[t], (M - 1) *N * sizeof(float), cudaMemcpyHostToDevice);
		for (int j = 0; j<N; j++)
		{
			sA += err2[t][j];
		}
		Xt = (float*)malloc(K*N * sizeof(float));
		memcpy(Xt, h_x, K*N * sizeof(float));
		cudaMemcpy(d_x, Xt+t* K*N, K*N * sizeof(float), cudaMemcpyHostToDevice);
		trans2(d_x, K, N);
		gpu_blas_mmul(d_err2, d_x, temp_V, (M - 1), N, K);
		add_matrixes << <M - 1, K >> >(d_dV, temp_V, d_dV, M - 1, K);
		free(Xt);

		cudaMemcpy(temp_h, ht[t - 1], (M - 1)*N * sizeof(float), cudaMemcpyHostToDevice);
		trans2(temp_h, (M - 1), N);
		gpu_blas_mmul(d_err2, temp_h, temp_V, (M - 1), N, M-1);
		add_matrixes << <(M - 1), (M - 1) >> >(d_dU, temp_h, d_dU, M - 1, (M - 1));
	}

	vector<float> v;
	v.push_back(sA);
	v.push_back(dB);
	float *h_dW, *h_dV, *h_dU;
	h_dW = (float*)malloc(ClassLength*(M - 1) * sizeof(float));
	cudaMemcpy(h_dW, d_dW, ClassLength*(M - 1) * sizeof(float), cudaMemcpyDeviceToHost);

	h_dV = (float*)malloc((M - 1)* K * sizeof(float));
	cudaMemcpy(h_dV, d_dV, (M - 1)* K * sizeof(float), cudaMemcpyDeviceToHost);

	h_dU = (float*)malloc((M - 1)* (M - 1) * sizeof(float));
	cudaMemcpy(h_dU, d_dU, (M - 1)* (M - 1) * sizeof(float), cudaMemcpyDeviceToHost);
	for (int i = 0; i < ClassLength*(M - 1); i++)
		v.push_back(h_dW[i]);
	for (int i = 0; i < (M - 1)* K; i++)
		v.push_back(h_dV[i]);
	for (int i = 0; i < (M - 1)* (M - 1); i++)
		v.push_back(h_dU[i]);
	free(h_dW);
	free(h_dV);
	free(h_dU);
	float nv = norm(v,v.size());
	if (isinf(nv))
		nv = v.size();
	v.clear();
	sA /= nv;
	dB /= nv;
	for(int i=0;i<M-1;i++)
		h_a[i] -= a * sA;
	for (int i = 0; i<ClassLength; i++)
		h_b[i] -= a * dB;
	multiScalaronMatrix << <ClassLength, (M - 1) >> >(-a/ nv, d_dW, ClassLength*(M - 1));
	add_matrixes << <ClassLength, (M - 1) >> >(d_w, d_dW, d_w, ClassLength, M - 1);

	multiScalaronMatrix << <(M - 1), K >> >(-a/ nv, d_dV, (M - 1)* K);
	add_matrixes << <M - 1, K >> >(d_V,d_dV, d_V, M - 1, K);

	multiScalaronMatrix << <(M - 1), (M - 1) >> >(-a/ nv, d_dU, (M - 1)* (M - 1));
	add_matrixes << <(M - 1), (M - 1) >> >(d_U, d_dU, d_U, M - 1, (M - 1)* (M - 1));


	cudaMemcpy(h_w, d_w, ClassLength*(M - 1) * sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(h_v, d_V, (M - 1) *K * sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(h_u, d_U, (M - 1)* (M - 1) * sizeof(float), cudaMemcpyDeviceToHost);

	cudaFree(d_dW);
	cudaFree(d_h);
	cudaFree(d_err1);
	cudaFree(d_w);
	cudaFree(d_x);
	cudaFree(d_err2);
	cudaFree(d_V);
	cudaFree(temp_V);
	cudaFree(temp_h);
}

void fit(float* h_V, float* h_U, float* h_a, float* h_W, float* h_b, int* netSize, float* h_X, const float* h_y)
{
	float a = 0.05;
	float er_max = 0.07; // порог допустимой ошибки
	int n = 4e3; // максимальное количество эпох обучения
	float* er_hist = new float[n]; // история изменения ошибки
	for (int i = 0; i<n; i++)
	{
		float *O = (float*)malloc(ClassLength * N * sizeof(float));
		float **Ht = reinterpret_cast<float **>(malloc(M * sizeof(float*)));
		float **St = reinterpret_cast<float **>(malloc(M * sizeof(float*)));
		float **err2 = reinterpret_cast<float **>(malloc(M * sizeof(float*)));
		for (int t = 0; t < M; t++) {
			Ht[t] = reinterpret_cast<float *>(malloc((M - 1) *N * sizeof(float)));
			St[t] = reinterpret_cast<float *>(malloc((M - 1) *N * sizeof(float)));
			err2[t] = reinterpret_cast<float *>(malloc((M - 1) *N * sizeof(float)));
		}

		forward(h_X, h_V, h_U, h_a, h_W, h_b, netSize, O, St, Ht);
		er_hist[i] = cross_entropy(O, h_y, ClassLength, N);
		printf("epoph %i/%i, error: %f/%f\n", i, n, er_hist[i], er_max);

		if (er_hist[i] < er_max) break;
		float *err1 = (float*)(malloc(ClassLength * N * sizeof(float)));

		bptt_backward(h_W, h_U, h_y, O, err1, err2, St);
		bptt_gradient(h_X, h_V, h_U, h_a, h_W, h_b, err1, err2, St, Ht,a);

		free(O);
		free(Ht);
		free(St);
		free(err2);
		free(err1);
	}

}

int main()
{
	float *X = (float*)malloc(M* K * N * sizeof(float));
	float *y = (float*)malloc(N * sizeof(float));
	generateData(X, y);
	print_matrix(y, N, 1);
	int*netSize = new int[3];
	netSize[0] = K; netSize[1] = M - 1; netSize[2] = ClassLength;

	float *V = (float*)malloc(netSize[1] * netSize[0] * sizeof(float));
	float *U = (float*)malloc(netSize[1] * netSize[1] * sizeof(float));
	float *a = (float*)malloc(netSize[1] * sizeof(float));
	float *W = (float*)malloc(netSize[2] * netSize[1] * sizeof(float));
	float *b = (float*)malloc(netSize[2] * sizeof(float));
	generateWeight(V, U, a, W, b, netSize);

	fit(V, U, a, W, b, netSize, X, y);
    return 0;
}
