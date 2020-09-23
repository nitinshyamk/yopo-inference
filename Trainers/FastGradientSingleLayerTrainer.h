#pragma once
#include <memory>
#include <torch/torch.h>
#include "ITrainer.h"
#include "utilities.h"
#include "Loss.h"

template <typename LayerType>
class FastGradientSingleLayerTrainer
{
public:
	FastGradientSingleLayerTrainer(
		torch::nn::ModuleHolder<LayerType> layerone,
		double sigma,
		double epsilon,
		int N2) :
		_hamiltonian(layerone),
		_optimizer(std::make_shared<torch::optim::SGD>(
			layerone->parameters(),
			torch::optim::SGDOptions(0.005).momentum(0.9).weight_decay(0.0005))),
		_N2(N2),
		_sigma(sigma),
		_epsilon(epsilon)
	{}

	std::pair<torch::Tensor, torch::Tensor> step(torch::Tensor data, torch::Tensor p, torch::Tensor eta)
	{
		if (!data.is_same_size(eta)) throw std::invalid_argument("data and eta must be of the same size");
		p.detach_();
		for (int i = 0; i < _N2; ++i)
		{
			auto tmp_input = torch::clamp_(data + eta, 0, 1);
			auto H = _hamiltonian(tmp_input, p);
			auto eta_grad = torch::autograd::grad({ H }, { eta }, {}, false);
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
		return std::make_pair(yopo_input, eta);
	}

	void param_zero_grad() { this->_optimizer->zero_grad(); }
	void param_step() { this->_optimizer->step(); }


private:
	Hamiltonian<LayerType> _hamiltonian;
	std::shared_ptr<torch::optim::Optimizer> _optimizer;
	int _N2;
	double _epsilon;
	double _sigma;
};
