#pragma once
#include <memory>
#include <torch/torch.h>
#include "ITrainer.h"
#include "FastGradientSingleLayerTrainer.h"
#include "utilities.h"
#include "Loss.h"

template <typename NetworkType, typename LossModuleType>
class YOPOTrainer : public ITrainer
{
public:
	/// <summary>
	/// Creates a Trainer for YOPO-K-N2
	/// </summary>
	/// <typeparam name="NetworkType"></typeparam>
	/// <typeparam name="LossModuleType"></typeparam>
	YOPOTrainer(
		torch::nn::ModuleHolder<NetworkType> network,
		std::shared_ptr<torch::optim::Optimizer> optimizer,
		torch::nn::ModuleHolder<LossModuleType> loss,
		int K, 
		int N2,
		double sigma,
		double epsilon,
		c10::Device device = c10::kCPU) :
		_network(network), 
		_loss(loss),
		_optimizer(optimizer),
		_device(device),
		_layer_one_trainer(
			network->layer_one(),
			sigma,
			epsilon,
			N2),
		_K(K), 
		_epsilon(epsilon)
	{}

	void train_batch(torch::data::Example<> example)
	{
		auto data = example.data; data.to(_device);
		auto labels = example.target; labels.to(_device);

		auto eta = (torch::rand_like(data) - 0.5) * 2 * _epsilon;
		eta.to(_device);
		eta.requires_grad_();

		_optimizer->zero_grad();
		_layer_one_trainer.param_zero_grad();

		for (int j = 0; j < _K; ++j)
		{
			auto pred = _network(data + eta.detach());
			auto loss = _loss(pred, labels);

			// next line obtains p for the Hamiltonian
			auto p = -1.0 * _network->layer_one_output().grad();
			
			auto toggleConv1RequiresGrad = [&](bool requiresGrad) {
				this->_network->conv1()->named_parameters()["weight"].requires_grad_(requiresGrad);
			};
			toggleConv1RequiresGrad(false);
			loss.backward();
			toggleConv1RequiresGrad(true);
			 
			torch::Tensor yopo_input, eta;
			std::tie(yopo_input, eta) = _layer_one_trainer.step(data, p, eta);

			{	
				torch::NoGradGuard ngg;
				if (j == 0)
				{
					_clean_accuracy.update(calculate_torch_accuracy(pred, labels), false);
				}
				if (j == _K - 1)
				{
					auto yopo_pred = _network(yopo_input);
					_yopo_accuracy.update(calculate_torch_accuracy(yopo_pred, labels), false);
				}
			}
		}
		_optimizer->step();
		_layer_one_trainer.param_step();
		_optimizer->zero_grad();
		_layer_one_trainer.param_zero_grad();
	}

	std::pair<double, double> get_accuracies()
	{
		return std::make_pair(_clean_accuracy.getMean(), _yopo_accuracy.getMean());
	}

private:
	torch::nn::ModuleHolder<NetworkType> _network;
	FastGradientSingleLayerTrainer<StackSequentialImpl> _layer_one_trainer;
	torch::nn::ModuleHolder<LossModuleType> _loss;
	std::shared_ptr<torch::optim::Optimizer> _optimizer;
	int _K;
	double _epsilon;
	average_meter _clean_accuracy = average_meter("clean accuracy");
	average_meter _yopo_accuracy = average_meter("yopo accuracy");

	c10::Device _device;
};