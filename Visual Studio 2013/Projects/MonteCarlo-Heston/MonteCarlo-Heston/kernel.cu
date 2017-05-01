
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <curand.h>
#include <curand_kernel.h>


#include <iostream>
#include <cmath>
#include <vector>
#include "dev_array.h"



#include <stdio.h>
#include <vector>
#include <time.h>
#include <math.h>
#include <iostream>
#include <time.h>
#include <cuda_runtime.h>

#define mu 0.05f
#define sigma .2f
#define timespan 252.0f


#define TRIALS 10000
#define numThreads 512


using namespace std;
//#include <random>

__global__ void europeanOption(
	int size, int iterations,
	float *d_price, float initialPrice, float strikePrice,
	curandState_t *d_state, float * d_normals)
{
	int tid = threadIdx.x + blockDim.x * blockIdx.x;
	/*std::vector<double> correlated;
	double rho = .6;*/

	float spot_normals;
	int temp = 0;
	
	/*std::random_device rd;
	std::mt19937 e2(rd());
	std::normal_distribution<> dist(0, 1);*/
	double spot_draws[100];

	double S_0 = 100.0;    // Initial spot price
	double K = 100.0;      // Strike price
	double r = 0.0319;     // Risk-free rate
	double v_0 = 0.010201; // Initial volatility 
	double T = 1.00;       // One year until expiry

	double rho = -0.7;     // Correlation of asset and volatility
	double kappa = 6.21;   // Mean-reversion rate
	double theta = 0.019;  // Long run average volatility
	double xi = 0.61;      // "Vol of vol"

	if (tid < size)
	{
		size_t vec_size = iterations;
		double dt = T ;
		for (int i = 0; i < iterations; i++)
		{
			initialPrice *= 1 + mu / timespan + curand_normal(&d_state[tid])*sigma / sqrt(timespan); // initial code
			dt = d_normals[tid];
			
			//Vol Path
			/*for (int i = 1; i<vec_size; i++) {
				double v_max = std::max(vol_path[i - 1], 0.0);
				vol_path[i] = vol_path[i - 1] + kappa * dt * (theta - v_max) +
					xi * sqrt(v_max * dt) * vol_draws[i - 1];*/


			/*for (int n = 0; n < 100; ++n) {
				temp = std::round(dist(e2));
			}*/
			/*
			1) Use the correlated brownian motion to create the vol_path
			2) Use that vol path to create the spot_path
			3) Use the vol_path and spot_path to create asset path
			4)
			*/
				

			//correlated[i] = rho * (curand_normal(&d_state[tid]))[i] + correlated[i] * sqrt(1 - rho*rho);

			/*for (int i = 0; i<vals; i++) {
				correlated[i] = rho * (spot_normals)[i] + correlated[i] * sqrt(1 - rho*rho);
			}*/
		}

		d_price[tid] = initialPrice - strikePrice;
		if (d_price[tid] < 0)
		{
			d_price[tid] = 0;
		}
	}

}

__global__ void init(
	unsigned int seed,
	curandState_t *d_state)
{
	curand_init(
		seed,
		threadIdx.x + blockDim.x * blockIdx.x,
		0,
		&d_state[threadIdx.x + blockDim.x * blockIdx.x]);
}


int main()
{
	int iterations = 1000;
	dev_array<float> d_normals(iterations);

	curandGenerator_t curandGenerator;
	curandCreateGenerator(&curandGenerator, CURAND_RNG_PSEUDO_MTGP32);
	curandSetPseudoRandomGeneratorSeed(curandGenerator, 1234ULL);
	curandGenerateNormal(curandGenerator, d_normals.getData(), 10000, 0.0f, 1); // acutally sqrt of dt instead of 1 as variance


	float *h_prices, *d_prices;

	h_prices = new float[TRIALS];
	cudaMalloc((void**)&d_prices, TRIALS*sizeof(float));

	curandState_t *d_state;
	cudaMalloc((void**)&d_state, TRIALS * sizeof(curandState_t));

	init << < (TRIALS - numThreads - 1) / numThreads, numThreads >> >(time(0), d_state);

	europeanOption << <(TRIALS - numThreads - 1) / numThreads, numThreads >> >(
		TRIALS, 252,
		d_prices, 100.0f, 100.0f,
		d_state, d_normals.getData());

	cudaMemcpy(h_prices, d_prices, TRIALS*sizeof(float), cudaMemcpyDeviceToHost);

	float price = 0;

	int count = 0;

	for (int i = 0; i < TRIALS; i++)
	{
		price += h_prices[i];
		if (h_prices[i] != 0)
		{
			count += 1;
		}
	}

	price /= TRIALS;

	std::cout << "The Theoretical Price of the Option is " << price << "." << std::endl;

	std::cout << "Count is " << count << "." << std::endl;

	delete[] h_prices;
	cudaFree(d_state); cudaFree(d_prices);

	cudaDeviceReset();

	int i;
	std::cin >> i;
	return 0;
}