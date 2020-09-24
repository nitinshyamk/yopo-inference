// yopo-experiment.cpp : Defines the entry point for the application.
//
#include "yopo-experiment.h"
#include <memory>
#include <deque>
#include "SmallCNN.h"
#include "Attackers/IAttacker.h"
#include "Attackers/PGDAttacker.h"
#include "Evaluator.h"
#include "Loss.h"
#include "Trainers/StandardTrainer.h"
#include "Trainers/YOPOTrainer.h"
#include "ExperimentRunner.h"

namespace nn = torch::nn;
namespace dt = torch::data;

int main() {
	using OptimizerPtr = std::shared_ptr<torch::optim::Optimizer>;
	using TrainerPtr = std::shared_ptr<ITrainer>;
	using ExperimentRunnerPtr = std::shared_ptr<IExperimentRunner>;
	using namespace std;


	c10::Device DEVICE = c10::kCUDA;
	std::deque<ExperimentRunnerPtr> experiments;

	auto mnist_training = dt::datasets::MNIST("D:/Projects/data/mnist", dt::datasets::MNIST::Mode::kTrain)
		.map(dt::transforms::Normalize<>(0.5, 0.5))
		.map(dt::transforms::Stack<>());

	{
		std::string experimentName = "PGD-Adversarial-1";

		SmallCNN smcnn; smcnn->to(DEVICE);

		OptimizerPtr optimizer = std::make_shared<torch::optim::Adam>(smcnn->parameters());

		shared_ptr<IAttacker<SmallCNNImpl>> pgdattacker = std::make_shared<PGDAttacker<SmallCNNImpl>>(6.0 / 255.0, 3.0 / 255.0, 20, DEVICE);

		TrainerPtr trainer = std::make_shared<StandardTrainer<SmallCNNImpl, nn::CrossEntropyLossImpl>>(
			smcnn, pgdattacker, optimizer, torch::nn::CrossEntropyLoss(), DEVICE);

		ExperimentRunnerPtr experiment = std::make_shared<ExperimentRunner<SmallCNNImpl, decltype(mnist_training)>>(
			experimentName, mnist_training, smcnn, trainer, 50, 100, DEVICE);
		experiments.push_back(experiment);
	};


	for (auto experiment : experiments)
		experiment->Run();
}
