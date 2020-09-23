#pragma once
#include <torch/torch.h>

template <typename ModuleType>
struct Hamiltonian
{
	Hamiltonian(torch::nn::ModuleHolder<ModuleType> layer) : _layer(layer) {}

	torch::Tensor operator()(torch::Tensor x, torch::Tensor p)
	{
		auto y = _layer(x);
		auto H = torch::sum(y * p);
		return H;
	}

	torch::nn::ModuleHolder<ModuleType> _layer;
};


struct CrossEntropyWithWeightPenaltyImpl : torch::nn::Cloneable<CrossEntropyWithWeightPenaltyImpl>
{
	CrossEntropyWithWeightPenaltyImpl(torch::nn::Module& network, double penalty, c10::Device device = c10::kCPU) :
		_network(network), _penalty(penalty), _device(device), _loss(torch::nn::CrossEntropyLoss()) 
	{
		_network.to(device);
		_loss->to(device);
	}

	virtual void reset()
	{
		_loss->zero_grad();
	}

	torch::Tensor forward(const torch::Tensor& prediction, const torch::Tensor& target)
	{
		auto lossval = _loss(prediction, target);
		auto penaltyval = calculate_l2_norm_sum() * _penalty;
		if (get_element_count(penaltyval) != 1)
			throw std::invalid_argument("invalid computed penalty");
		return penaltyval[0];
	}

private:
	torch::Tensor calculate_l2_norm_sum()
	{
		auto loss = torch::tensor(0.0);
		auto params = _network.named_parameters();
		for (auto it = params.begin(); it != params.end(); ++it)
		{
			if (it->key() == "weight")
			{
				auto normv = torch::norm(it->value());
				loss += 0.5 * normv * normv;
			}
		}
		return loss;
	}

	torch::nn::Module _network;
	torch::nn::CrossEntropyLoss _loss;
	c10::Device _device;
	double _penalty;
};

TORCH_MODULE(CrossEntropyWithWeightPenalty);