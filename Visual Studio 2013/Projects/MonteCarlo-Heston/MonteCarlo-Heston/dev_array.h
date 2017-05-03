#ifndef _DEV_ARRAY_H_
#define _DEV_ARRAY_H_

#include <stdexcept>
#include <algorithm>
#include <cuda_runtime.h>

template <class T>
class dev_array
{
	// public functions
public:
	explicit dev_array()
		: start_(0),
		end_(0)
	{}

	// constructor
	explicit dev_array(size_t size)
	{
		allocate(size);
	}
	// destructor
	~dev_array()
	{
		free();
	}

	// resize the vector
	void resize(size_t size)
	{
		free();
		allocate(size);
	}

	// get the size of the array
	size_t getSize() const
	{
		return end_ - start_;
	}

	// get data
	const T* getData() const
	{
		return start_;
	}

	T* getData()
	{
		return start_;
	}

	// set
	void set(const T* src, size_t size)
	{
		size_t min = std::min(size, getSize());
		cudaError_t result = cudaMemcpy(start_, src, min * sizeof(T), cudaMemcpyHostToDevice);
		if (result != cudaSuccess)
		{
			throw std::runtime_error("failed to copy to device memory");
		}
	}
	// get
	void get(T* dest, size_t size)
	{
		size_t min = std::min(size, getSize());
		cudaError_t result = cudaMemcpy(dest, start_, min * sizeof(T), cudaMemcpyDeviceToHost);
		if (result != cudaSuccess)
		{
			throw std::runtime_error("failed to copy to host memory");
		}
	}


	// private functions
private:
	// allocate memory on the device
	void allocate(size_t size)
	{
		cudaError_t result = cudaMalloc((void**)&start_, size * sizeof(T));
		if (result != cudaSuccess)
		{
			start_ = end_ = 0;
			throw std::runtime_error("failed to allocate device memory");
		}
		end_ = start_ + size;
	}

	// free memory on the device
	void free()
	{
		if (start_ != 0)
		{
			cudaFree(start_);
			start_ = end_ = 0;
		}
	}

	T* start_;
	T* end_;
};

#endif
for (int j = 0; j < size; j++)
{

	vol_path = v_0;
	spot_path = S_0;
	for (int i = 0; i < iterations; i++)
	{

		v_max = std::max(vol_path, 0.0);

		//Vt = Vt - kappa_ * dt * (std::max(0.0, Vt) - theta_) + vol_ * sqrt(std::max(0.0, Vt)) * Vhold * sqdt;
		vol_path = vol_path + kappa*dt*(theta - v_max) + xi*sqrt(v_max*dt)*random1;

		spot_path = spot_path*exp((r - .5*v_max)*dt + sqrt(v_max*dt)*corr_random);


		random1 = dist(e2);
		random2 = dist(e2);
		corr_random = rho * random1 + sqrt(1 - rho * rho) * random2;

		//std::cout << "Vol="<<vol_path << std::endl;
		//std::cout << "spot=" << spot_path << std::endl;
	}
	initialPrice = spot_path;
	d_price[j] = initialPrice - strikePrice;
	if (d_price[j] < 0)
	{
		d_price[j] = 0;
	}
	/*std::cout << "Count-" << j << std::endl;
	std::cout << "Price=" << d_price[j] << std::endl;*/
}