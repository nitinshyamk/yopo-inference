#pragma once
#include <memory>
#include <torch/torch.h>
#include "utilities.h"


template <typename NetworkType>
class ITrainer
{
public:
	virtual void train_batch(torch::data::Example<> example) = 0;
};

template <typename NetworkType, typename LossModuleType>
class StandardTrainer : public ITrainer<NetworkType>
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
			auto adversarial_input = (*_attacker)(network, data, label);
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

	torch::nn::ModuleHolder<NetworkType> _network;
	std::shared_ptr<IAttacker<NetworkType>> _attacker;
	std::shared_ptr<torch::optim::Optimizer> _optimizer;
	torch::nn::ModuleHolder<LossModuleType> _loss;
	c10::Device _device;

	average_meter _clean_accuracy = average_meter("Clean accuracy");
	average_meter _adversarial_accuracy = average_meter("Adversarial accuracy");
};

template <typename LayerType> 
class FastGradientSingleLayerTrainer
{
public:
	FastGradientSingleLayerTrainer()
	std::pair<torch::Tensor, torch::Tensor> step(torch::Tensor data, torch::Tensor p, torch::Tensor eta)
	{

	}

	void param_zero_grad() {}
	void param_step() {}



};

template <typename NetworkType, typename LossModuleType>
class YOPOTrainer : public ITrainer<NetworkType>
{
public:
	YOPOTrainer(
		torch::nn::ModuleHolder<NetworkType> network,
		std::shared_ptr<torch::optim::Optimizer> optimizer,
		torch::nn::ModuleHolder<LossModuleType> loss,
		int _K, 
		int _InnerLayer,
		c10::Device device = c10::kCPU) :
		_network(network), _loss(loss), _optimizer(optimizer), _device(device)
	{}

	void train_batch(torch::data::Example<> example)
	{
		auto data = example.data; data.to(_device);
		auto labels = example.target; labels.to(_device);

		auto eta = (torch::rand_like(data) - 0.5) * 2 * _epsilon;
	}

private:
	torch::nn::ModuleHolder<NetworkType> _network;
	torch::nn::ModuleHolder<LossModuleType> _loss;
	std::shared_ptr<torch::optim::Optimizer> _optimizer;
	int _K;
	double _epsilon;

	c10::Device _device;
};