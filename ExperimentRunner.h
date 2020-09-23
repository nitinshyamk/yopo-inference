#pragma once
#include <iostream>
#include <chrono>
#include <torch/torch.h>
#include "Trainers/ITrainer.h"
#include "Evaluator.h"

class IExperimentRunner
{
	virtual void Run() = 0;
};

struct ScopedBlockLabel
{
	ScopedBlockLabel(std::string msg) : _msg(msg)
	{
		using namespace std;
		auto now = chrono::system_clock::to_time_t(chrono::system_clock::now());
		cout << "Beginning " << msg << " at " << ctime(&now) << std::endl;
	}

	~ScopedBlockLabel()
	{
		using namespace std;
		auto now = chrono::system_clock::to_time_t(chrono::system_clock::now());
		std::cout << "Ending " << _msg << " at " << ctime(&now) << std::endl;
	}

	std::string _msg;
};

template <typename NetworkType, typename DatasetType>
class ExperimentRunner : public IExperimentRunner
{
	using DataLoader_t = std::unique_ptr<torch::data::StatelessDataLoader<DatasetType, torch::data::samplers::RandomSampler>>;
public:
	ExperimentRunner(
		std::string experimentName,
		DatasetType& dataset,
		torch::nn::ModuleHolder<NetworkType> network,
		std::shared_ptr<ITrainer> trainer,
		int numberOfEpochs,
		int batchSize,
		c10::Device device) :

		_experimentName(experimentName),
		_network(network),
		_trainer(trainer),
		_batchSize(batchSize),
		_numberOfEpochs(numberOfEpochs),
		_pgdAttacker(std::make_shared<PGDAttacker<NetworkType>>(
			/*epsilon*/ 6.0 / 255.0,
			/*sigma*/ 3.0 / 255.0,
			/*iterations*/ 20,
			/*device*/ device)),
		_device(device),
		_dataset(dataset)
	{}

	void Run() override
	{
		ScopedBlockLabel startExperiment("Experiment " + _experimentName);
		
		// Train
		Evaluator<NetworkType> evaluator(_pgdAttacker, _device);

		DataLoader_t dataloader = torch::data::make_data_loader(
			_dataset,
			torch::data::DataLoaderOptions().batch_size(_batchSize).workers(2));

		for (int epoch = 0; epoch < _numberOfEpochs; ++epoch)
		{
			ScopedBlockLabel startExperiment("epoch " + std::to_string(epoch + 1));

			// Training block
			for (torch::data::Example<> batch : *dataloader)
				_trainer->train_batch(batch);

			print_accuracies(_trainer->get_accuracies());

			if (epoch % 10)
			{
				this->evaluate(evaluator, dataloader);
			}
		}

		this->evaluate(evaluator, dataloader);
	}


private:
	void evaluate(Evaluator<NetworkType>& evaluator, DataLoader_t& dataset)
	{
		_network->eval();
		evaluator.reset();
		for (torch::data::Example<> batch : *dataset)
			evaluator.evaluate_single_batch(_network, batch);
		print_accuracies(evaluator.get_accuracies());
		_network->train();
	}

	void print_accuracies(std::pair<double, double> accuracies)
	{
		std::cout << "Clean accuracy " << accuracies.first << ", Adversarial accuracy " << accuracies.second << std::endl;
	}

	std::string _experimentName;
	DatasetType _dataset;
	torch::nn::ModuleHolder<NetworkType> _network;
	std::shared_ptr<ITrainer> _trainer;

	int _numberOfEpochs;
	int _batchSize;
	c10::Device _device;
	std::shared_ptr<PGDAttacker<NetworkType>> _pgdAttacker;

};