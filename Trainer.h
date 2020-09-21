#pragma once
#include <memory>
#include <torch/torch.h>
#include "utilities.h"
#include "Loss.h"


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
	FastGradientSingleLayerTrainer(
		torch::nn::ModuleHolder<LayerType> _layerone,
		std::shared_ptr<torch::optim::Optimizer> optimizer,
		double sigma,
		double epsilon,
		int N2) : _hamiltonian(_layerone), _optimizer(optimizer), _N2(N2), _sigma(sigma), _epsilon(epsilon) {}

	std::pair<torch::Tensor, torch::Tensor> step(torch::Tensor data, torch::Tensor p, torch::Tensor eta)
	{
		if (!data.is_same_size(eta)) throw std::invalid_argument("data and eta must be of the same size");
		p.detach_();
		for (int i = 0; i < _N2; ++i)
		{
			auto tmp_input = torch::clamp_(data + eta, 0, 1);
			auto H = _hamiltonian(tmp_input, p);
			auto eta_grad = torch::autograd::grad({ H }, { eta }, {}, retain_graph = false);
			if (eta_grad.size() < 1) throw std::invalid_argument("autograd::grad failed to compute expected gradients");
			auto eta_grad_sign = eta_grad[0].sign();
			eta = eta - eta_grad_sign * _sigma;
			eta = torch::clamp_(eta, -1 * _epsilon, _epsilon);
			eta = torch::clamp_(data + eta, 0.0, 1.0) - data;
			eta.detach_();
			eta.requires_grad_();
			eta.retain_grad();
		}

		auto yopo_input = torch::clamp(eta + data, 0, 1);
		auto loss = -1.0 * _hamiltonian(yopo_input, p);
		loss.backward();
		return make_pair(yopo_input, p);
	}

	void param_zero_grad() { this->_optimizer->zero_grad();  }
	void param_step() { this->_optimizer->step();  }

private:
	Hamiltonian<LayerType> _hamiltonian;
	std::shared_ptr<torch::optim::Optimizer> _optimizer;
	int _N2;
	double _epsilon;
	double _sigma;
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
		_network(network), _loss(loss), _optimizer(optimizer), _device(device),
		_layer_one_trainer(
			network->layer_one(),
			optimizer,
			0.008,
			0.03,
			_InnerLayer)
	{}

	void train_batch(torch::data::Example<> example)
	{
		auto data = example.data; data.to(_device);
		auto labels = example.target; labels.to(_device);

		auto eta = (torch::rand_like(data) - 0.5) * 2 * _epsilon;
		eta.to(_device);
		eta.requires_grad_();

		_optimizer->zero_grad();
	}

private:
	torch::nn::ModuleHolder<NetworkType> _network;
	FastGradientSingleLayerTrainer<StackSequentialImpl> _layer_one_trainer;
	torch::nn::ModuleHolder<LossModuleType> _loss;
	std::shared_ptr<torch::optim::Optimizer> _optimizer;
	int _K;
	double _epsilon;

	c10::Device _device;
};