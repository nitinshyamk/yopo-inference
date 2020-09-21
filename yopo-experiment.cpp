// yopo-experiment.cpp : Defines the entry point for the application.
//

#include "yopo-experiment.h"
#include <memory>
#include "SmallCNN.h"
#include "Attacker.h"
#include "Evaluator.h"
#include "Loss.h"
#include "Trainer.h"

int main() {
	using namespace std;
	torch::Tensor tensor = torch::rand({ 2, 3 });
	cout << tensor << endl;

	SmallCNN smcnn;

	smcnn->train();
	auto tstinput = torch::rand({ 1, 1, 28, 28 });
	std::cout << tstinput.dim() << std::endl;
	for (int i = 0; i < tstinput.dim(); ++i)
	{
		std::cout << tstinput.size(i) << std::endl;
	}

	std::shared_ptr<torch::optim::Optimizer> optimizer = std::make_shared<torch::optim::Adam>(smcnn->parameters());
	std::shared_ptr<IAttacker<SmallCNNImpl>> pgdattacker = std::make_shared<PGDAttacker<SmallCNNImpl>>();

	StandardTrainer<SmallCNNImpl, torch::nn::CrossEntropyLossImpl> standardTrainer(smcnn, pgdattacker, optimizer, torch::nn::CrossEntropyLoss(), c10::kCUDA);

	YOPOTrainer<SmallCNNImpl, torch::nn::CrossEntropyLossImpl> yopoTrainer(smcnn, optimizer, torch::nn::CrossEntropyLoss(), 5, 10);
}
