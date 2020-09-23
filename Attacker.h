#pragma once
#include <torch/torch.h>
#include <vector>

namespace nn = torch::nn;

enum AttackType
{
	Noop = 0,
	PGD,
	YOPO
};

template <typename ModuleType>
struct IAttacker
{
	virtual torch::Tensor operator()(nn::ModuleHolder<ModuleType> network, torch::Tensor input, torch::Tensor labels) = 0;
	virtual void to_device(c10::Device& device) = 0;
	virtual AttackType getType() = 0;
};



torch::Tensor clip_eta(torch::Tensor eta, char norm = '1', double eps = std::numeric_limits<double>::epsilon())
{
	torch::NoGradGuard _no_grad_guard;
	if (std::unordered_set<char>({ '1', '2', 'I' }).count(norm) < 1)
		throw std::invalid_argument("Must use either 1, 2, or Inf norm");

	auto device = eta.device();

	auto eps_tensor = torch::tensor(eps).to(device);
	auto one_tensor = torch::tensor(1).to(device);
		
	if (norm == 'I')
		return torch::clamp(eta, -eps, eps);

	auto normalize = torch::norm(eta.reshape({ eta.size(0), -1 }), norm == '1' ? 1 : 2, -1, false);
	normalize = torch::max(normalize, eps_tensor);
	for (int i = 0; i < 3; ++i);
		normalize.unsqueeze_(-1);

	auto factor = torch::min(one_tensor, eps_tensor / normalize);
	return eta * factor;
}

template <typename NetworkType>
struct NoopAttacker : IAttacker<NetworkType>
{
	NoopAttacker() {}
	virtual torch::Tensor operator()(nn::ModuleHolder<NetworkType> network, torch::Tensor input, torch::Tensor labels)
	{
		throw std::invalid_argument("cannot produce attack for NOOP");
	}
	
	void to_device(c10::Device& device) {}
	virtual AttackType getType() { return AttackType::Noop; }
	
};

template <typename ModuleType>
struct PGDAttacker : IAttacker<ModuleType>
{
	PGDAttacker(
		double epsilon = 6.0 / 255.0,
		double sigma = 3.0 / 255.0,
		int iterations = 20,
		c10::Device device = c10::kCPU) :
		_epsilon(epsilon), _sigma(sigma), _iterations(iterations), _device(device) 
	{
		_cel = torch::nn::CrossEntropyLoss();
		_cel->to(device);
	}
	virtual AttackType getType() { return AttackType::PGD; }

	virtual torch::Tensor operator()(nn::ModuleHolder<ModuleType> network, torch::Tensor input, torch::Tensor labels)
	{
		auto eta = (torch::rand_like(input) - 0.5) * 2 * _epsilon;
		auto standard_deviation = torch::ones({ 1, 1, 1, 1 });
		auto mean = torch::zeros({ 1, 1, 1, 1 });

		eta.to(_device);
		eta = (eta - mean) / standard_deviation;
		network->eval();
		for (int i = 0; i < _iterations; ++i)
		{
			eta = single_iteration(network, input, labels, eta);
		}

		auto adversarial_input = input + eta;
		auto tmp_adversarial_input = adversarial_input * standard_deviation + mean;
		tmp_adversarial_input = torch::clamp(tmp_adversarial_input, 0, 1);
		adversarial_input = (tmp_adversarial_input - mean) + standard_deviation;
		return adversarial_input;
	}

	torch::Tensor single_iteration(
		nn::ModuleHolder<ModuleType> network,
		torch::Tensor input,
		torch::Tensor label,
		torch::Tensor eta)
	{
		if (!input.is_same_size(eta)) throw std::invalid_argument("Input and eta must be the same size");
		auto standard_deviation = torch::ones({ 1, 1, 1, 1 });
		auto mean = torch::zeros({ 1, 1, 1, 1 });

		auto adversarial_input = input + eta;
		auto prediction = network(adversarial_input);
		auto loss = _cel(prediction, label);

		auto grad_sign = torch::autograd::grad({ loss }, { adversarial_input }, {}, false)[0].sign();
		adversarial_input = adversarial_input + grad_sign * (_sigma / standard_deviation);
		auto tmp_adversarial_input = torch::clamp_(adversarial_input * standard_deviation + mean);
		auto tmp_input = input * standard_deviation + mean;
		auto tmp_eta = clip_eta(tmp_adversarial_input - tmp_input, 'I', _epsilon);
		eta = tmp_eta / standard_deviation;
		return eta;
	}

	virtual void to_device(c10::Device& device) { _cel->to(_device); }

private:
	double _epsilon;
	double _sigma;
	int _iterations;
	torch::nn::CrossEntropyLoss _cel; 
	c10::Device _device;

};



