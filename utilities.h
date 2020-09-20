#pragma once
#include <string>
#include <torch/torch.h>

struct average_meter
{
	average_meter(std::string name) : _name(name), _sum(0), _mean(0), _count(0) {}
	average_meter() : average_meter("") {}

	void reset()
	{
		_sum = 0;
		_mean = 0;
		_count = 0;
	}

	void update(long double value, bool incrementalUpdate = true, long long count = 1)
	{
		if (!incrementalUpdate) this->reset();

		_count += count;
		_sum += count * value;
		_mean = _sum / _count;
	}

	long double getMean() { return _mean; }
	long double getSum() { return _sum; }
	long double getCount() { return _count; }

private:
	std::string _name;
	long double _sum, _mean;
	long long _count;
};

/// returns the number of elements in the tensor
int get_element_count(torch::Tensor a);

/// <summary>
/// Different from is_same_size in that assert_equal_content_count checks 
///	 the number of elements by taking product along dimensions;
/// </summary>
/// <param name="a"></param>
/// <param name="b"></param>
void assert_equal_content_count(torch::Tensor a, torch::Tensor b);

double calculate_torch_accuracy(torch::Tensor output, torch::Tensor target);