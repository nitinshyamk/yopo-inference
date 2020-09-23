#pragma once

#include <memory>
#include <torch/torch.h>


class ITrainer
{
public:
	virtual void train_batch(torch::data::Example<> example) = 0;
	virtual std::pair<double, double> get_accuracies() = 0;
};