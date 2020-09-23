#pragma once
#include <memory>
#include <torch/torch.h>
#include "ITrainer.h"
#include "utilities.h"
#include "Loss.h"


template <typename NetworkType, typename LossModuleType>
class StandardTrainer : public ITrainer
{
public:
	StandardTrainer(
		torch::nn::ModuleHolder<NetworkType> network,
		std::shared_ptr<IAttacker<NetworkType>> attacker,
		std::shared_ptr<torch::optim::Optimizer> optimizer,
		torch::nn::ModuleHolder<LossModuleType> loss,
		c10::Device device = c10::kCPU) :
		_network(network), _attacker(attacker), _device(device), _loss(loss), _optimizer(optimizer)
	{}

	void train_batch(torch::data::Example<> example)
	{
		auto data = example.data.to(_device);
		auto label = example.target.to(_device);
		(*_optimizer).zero_grad();

		if (_attacker->getType() != AttackType::Noop)
		{
			auto adversarial_input = (*_attacker)(_network, data, label);
			_optimizer->zero_grad();
			_network->train();
			auto prediction = _network(adversarial_input);
			auto loss = _loss(prediction, label);
			loss.backward();
			_adversarial_accuracy.update(calculate_torch_accuracy(prediction, label), false);
		}

		auto prediction = _network(data);
		auto loss = _loss(prediction, label);
		loss.backward();
		_optimizer->step();
		_clean_accuracy.update(calculate_torch_accuracy(prediction, label), false);
	}

	std::pair<double, double> get_accuracies()
	{
		return std::make_pair(_clean_accuracy.getMean(), _adversarial_accuracy.getMean());
	}

private:
	torch::nn::ModuleHolder<NetworkType> _network;
	std::shared_ptr<IAttacker<NetworkType>> _attacker;
	std::shared_ptr<torch::optim::Optimizer> _optimizer;
	torch::nn::ModuleHolder<LossModuleType> _loss;
	c10::Device _device;

	average_meter _clean_accuracy = average_meter("Clean accuracy");
	average_meter _adversarial_accuracy = average_meter("Adversarial accuracy");
};