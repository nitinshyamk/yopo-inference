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


