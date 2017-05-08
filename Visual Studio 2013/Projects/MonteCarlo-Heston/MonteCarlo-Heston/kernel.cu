//------------------------------------------------------------------------------------------------------------------------
// Author      : Saeed Rahman
// Title       : Vanila Call Option Pricing using Heston SDE using Monte Carlo Simulation in GPU and CPU. 
// Description : Project done as part of FE-529 "GPU Computing in Finance" Project.
// Date		   : 5/8/2017
//------------------------------------------------------------------------------------------------------------------------
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <curand.h>
#include <curand_kernel.h>
#include <algorithm>
#include <iostream>
#include <cmath>
#include <random>
#include <time.h>

#define TRIALS 100000
#define numThreads 512

// Kernel Implementation-------------------------------------------------------------------------------------------------
__global__ void europeanOption(
	int iterations,int size,
	float *d_price,
	curandState_t *d_state)
{
	int tid = threadIdx.x + blockDim.x * blockIdx.x;

	double S_0 = 100;
	double K = 100;
	double r = .0319;
	double v_0 = .010201;
	double T = 1;
	double dt = T / iterations;
	double rho = -.7;
	double kappa = 6.21;
	double theta = .019;
	double xi = .61;

	float random1=0;
	float random2=0;
	float corr_random = 0;

	double vol_path = v_0;
	double spot_path = S_0;
	double v_max=0;

	//  variable to store the random variable generated using curand
	float2 random;

	if (tid < iterations)
	{
		vol_path = v_0;
		spot_path = S_0;
		random1 = 0;
		random2 = 0;
		corr_random = 0;
		for (int i = 0; i < size; i++)
		{
			if (vol_path > 0)
				v_max = vol_path;
			else
				v_max = 0;

			vol_path = vol_path + kappa*dt*(theta - v_max) + xi*sqrt(v_max*dt)*random1;

			spot_path = spot_path*exp((r - .5*v_max)*dt + sqrt(v_max*dt)*corr_random);

			// creating two independent standard normal random values
			random = curand_normal2(&d_state[tid]);
			random1 = random.x;
			random2 = random.y;
			// creating the correlated brownian motion
			corr_random = rho * random1 + sqrt(1 - rho * rho) * random2;

		}
		// calculating the payoff
		d_price[tid] = spot_path - K;
		if (d_price[tid] < 0)
		{
			d_price[tid] = 0;

		}
	}

}


// setting the init function to set the seed for curand
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


// CPU implementation of the Heston MC----------------------------------------------------------------------------------------------------------
void CPU(int iterations,int size,  float *d_price)
{
	double S_0 = 100;
	double K = 100;
	double strikePrice = K;
	double r = .0319;
	double v_0 = .010201;
	double T = 1;
	double dt = T / iterations;
	double rho = -.7;
	double kappa = 6.21;
	double theta = .019;
	double xi = .61;

	float random1 = 0;
	float random2 = 0;
	float corr_random = 0;

	double vol_path = v_0;
	double spot_path = S_0;
	double v_max;

	double initialPrice = 0;

	//using the mersenne twister engine to create the random number
	std::random_device rd;
	std::mt19937 e2(rd());
	std::normal_distribution<>dist(0, 1);

	for (int j = 0; j < iterations; j++)
	{
		
		vol_path = v_0;
		spot_path = S_0;
		random1 = 0;
		random2 = 0;
		corr_random = 0;
		for (int i = 0; i < size; i++)
		{
			
			v_max = std::max(vol_path, 0.0);
			vol_path = vol_path + kappa*dt*(theta - v_max) + xi*sqrt(v_max*dt)*random1;

			spot_path = spot_path*exp((r - .5*v_max)*dt + sqrt(v_max*dt)*corr_random);

			// creating the independent and the correlated brownian motion
			random1 = dist(e2);
			random2 = dist(e2);
			corr_random = rho * random1 + sqrt(1 - rho * rho) * random2;
			/*std::cout << "random " << random1<< random2 << corr_random << std::endl;*/
			

		}
		// calculating the payoff
		initialPrice = spot_path;
		d_price[j] = initialPrice - strikePrice;
		if (d_price[j] < 0)
		{
			d_price[j] = 0;
		}
	}


}
int main()
{

	int number_of_steps = 252;
	float *h_prices, *d_prices;

	clock_t beginGPU = clock();

	h_prices = new float[TRIALS];
	cudaMalloc((void**)&d_prices, TRIALS*sizeof(float));

	curandState_t *d_state;

	cudaMalloc((void**)&d_state, TRIALS * sizeof(curandState_t));

	clock_t beginGPUKernel = clock();

	init << < (TRIALS - numThreads - 1) / numThreads, numThreads >> >(time(0), d_state);

	// calling the kernel

	europeanOption << <(TRIALS - numThreads - 1) / numThreads, numThreads >> >(
		TRIALS, number_of_steps,
		d_prices,
		d_state);


	clock_t endGPUKernel = clock();

	cudaMemcpy(h_prices, d_prices, TRIALS*sizeof(float), cudaMemcpyDeviceToHost);

	clock_t endGPU = clock();

	double elapsed_secs_GPU = double(endGPU - beginGPU) / float(CLOCKS_PER_SEC);

	double elapsed_secs_GPU_kernel = double(endGPUKernel - beginGPUKernel) / float(CLOCKS_PER_SEC);

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

	std::cout << "The Theoretical Price of the Option simulated in GPU = " << price << "." << std::endl;

	std::cout << "Count is " << count << "." << std::endl;

	std::cout << "The time taken for the GPU Simulation including memory transfer=" << elapsed_secs_GPU << std::endl;

	std::cout << "The time taken for the GPU Kernel=" << elapsed_secs_GPU_kernel << std::endl;

	delete[] h_prices;
	cudaFree(d_state); cudaFree(d_prices);

	cudaDeviceReset();


	//CPU Implementation--------------------------------------------------------------------------------------------------------------

	
	clock_t beginCPU = clock();
	float cpu_price = 0;
	int cpu_count = 0;
	float *option_prices;
	option_prices = new float[TRIALS];

	CPU(TRIALS, number_of_steps, option_prices);
	
	clock_t endCPU = clock();
	double elapsed_secs_CPU = double(endCPU - beginCPU) / float(CLOCKS_PER_SEC);

	for (int i = 0; i < TRIALS; i++)
	{
		cpu_price += option_prices[i];
		if (option_prices[i] > 0)
		{
			cpu_count += 1;
		}
	}
	cpu_price /= TRIALS;

	

	std::cout <<std::endl<< "The Theoretical Price of the Option simulated in CPU =" << cpu_price << "." << std::endl;

	std::cout << "The time taken for the CPU Simulation=" << elapsed_secs_CPU << std::endl;

	std::cout << "count=" << cpu_count << std::endl;

	int i;
	std::cin >> i;
	return 0;
}
