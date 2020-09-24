#pragma once
#include <torch/torch.h>
#include <vector>

namespace nn = torch::nn;

template <typename ModuleType>
struct PGDAttacker : IAttacker<ModuleType>
{
	PGDAttacker(
		double epsilon,
		double sigma,
		int iterations,
		c10::Device device) :
		_epsilon(epsilon), _sigma(sigma), _iterations(iterations), _device(device)
	{
		_cel = torch::nn::CrossEntropyLoss();
		_cel->to(device);
	}
	virtual AttackType getType() { return AttackType::PGD; }

	virtual torch::Tensor operator()(nn::ModuleHolder<ModuleType> network, torch::Tensor input, torch::Tensor labels)
	{
		auto eta = (torch::rand_like(input) - 0.5) * 2 * _epsilon;
		double standard_deviation = 1; //torch::ones({ 1, 1, 1, 1 }); standard_deviation.to(_device);
		double mean = 0; // torch::zeros({ 1, 1, 1, 1 }); mean.to(_device);
		std::cout << torch::ones({ 1, 1, 1, 1 }) << std::endl;
		std::cout << torch::zeros({ 1, 1, 1, 1 }) << std::endl;
		//std::cout << eta << std::endl;
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
		input.to(_device);
		label.to(_device);
		eta.to(_device);
		if (!input.is_same_size(eta)) throw std::invalid_argument("Input and eta must be the same size");
		auto standard_deviation = torch::ones({ 1, 1, 1, 1 }); standard_deviation.to(_device);
		auto mean = torch::zeros({ 1, 1, 1, 1 }); mean.to(_device);
		auto adversarial_input = torch::Tensor(input + eta).to(_device);
		std::cout << mean.get_device() <<  " " << _device << " AI device " << adversarial_input.get_device() << std::endl;
		auto prediction = network(adversarial_input); prediction.to(_device);
		auto loss = _cel(prediction, label); loss.to(_device);
		std::cout << torch::Device("cuda:0") << std::endl;
		adversarial_input.to(torch::Device("cuda:0"));
		std::cout << adversarial_input.get_device() << std::endl;
		//if (input.get_device() != _device || adversarial_input.get_device() != _device) throw std::invalid_argument("");
		auto grad_sign_pre = torch::autograd::grad({ loss }, { adversarial_input }, {}, false);
		auto grad_sign = grad_sign_pre[0].sign(); grad_sign.to(_device);
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
