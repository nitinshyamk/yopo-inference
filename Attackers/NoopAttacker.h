#pragma once
#include <torch/torch.h>
#include <vector>

namespace nn = torch::nn;

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