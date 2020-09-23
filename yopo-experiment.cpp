// yopo-experiment.cpp : Defines the entry point for the application.
//
#include "yopo-experiment.h"
#include <memory>
#include <deque>
#include "SmallCNN.h"
#include "Attacker.h"
#include "Evaluator.h"
#include "Loss.h"
#include "Trainers/StandardTrainer.h"
#include "Trainers/YOPOTrainer.h"
#include "ExperimentRunner.h"

namespace nn = torch::nn;

int main() {
	using OptimizerPtr = std::shared_ptr<torch::optim::Optimizer>;
	using TrainerPtr = std::shared_ptr<ITrainer>;
	using ExperimentRunnerPtr = std::shared_ptr<IExperimentRunner>;
	using namespace std;


	c10::Device DEVICE = c10::kCUDA;
	std::deque<ExperimentRunnerPtr> experiments;

	auto mnist_training = torch::data::datasets::MNIST("FIX ME")
		.map(torch::data::transforms::Normalize<>(0.5, 0.5))
		.map(torch::data::transforms::Stack<>());

	{
		std::string experimentName = "PGD-Adversarial-1";

		SmallCNN smcnn;

		OptimizerPtr optimizer = std::make_shared<torch::optim::Adam>(smcnn->parameters());

		shared_ptr<IAttacker<SmallCNNImpl>> pgdattacker = std::make_shared<PGDAttacker<SmallCNNImpl>>();

		TrainerPtr trainer = std::make_shared<StandardTrainer<SmallCNNImpl, nn::CrossEntropyLossImpl>>(
			smcnn, pgdattacker, optimizer, torch::nn::CrossEntropyLoss(), DEVICE);

		ExperimentRunnerPtr experiment = std::make_shared<ExperimentRunner<SmallCNNImpl, decltype(mnist_training)>>(
			experimentName, mnist_training, smcnn, trainer, 50, 100, DEVICE);
		experiments.push_back(experiment);
	};


}
