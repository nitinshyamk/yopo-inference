#pragma once
#include <torch/torch.h>
#include "Attacker.h"
#include "utilities.h"

template <typename NetworkType>
class Evaluator
{
public:
	Evaluator(std::shared_ptr<IAttacker<NetworkType>> attacker, const c10::Device& device)  :  _device(device), _attacker(attacker) {}

	void evaluate_single_batch(torch::nn::ModuleHolder<NetworkType> network, torch::data::Example<>& example)
	{
		auto data = example.data;
		auto label = example.target; 
		auto device = data.device();
		network->to(device);

		{ torch::NoGradGuard _nogradguard;

			auto prediction = network(data);
			_clean_accuracy.update(calculate_torch_accuracy(prediction, label));

			if (_attacker->getType() != AttackType::Noop)
			{
				auto adversarial_input = (*_attacker)(network, data, label);
				auto adv_prediction = network(data);
				_adversarial_accuracy.update(calculate_torch_accuracy(adv_prediction, label));
			}
		}
	}

	std::pair<double, double> get_accuracies()
	{
		return std::make_pair(_clean_accuracy.getMean(), _adversarial_accuracy.getMean());
	}

	void reset()
	{
		_clean_accuracy.reset();
		_adversarial_accuracy.reset();
	}

	average_meter _clean_accuracy = average_meter("clean accuracy");
	average_meter _adversarial_accuracy = average_meter("adversarial accuracy");
	c10::Device _device;
	std::shared_ptr<IAttacker<NetworkType>> _attacker;
};